/*
 * Hemi.hpp
 *
 */

#ifndef PROTOCOLS_HEMI_HPP_
#define PROTOCOLS_HEMI_HPP_

#include "Hemi.h"
#include "ShareMatrix.h"
#include "HemiOptions.h"

#include "HemiMatrixPrep.hpp"
#include "HemiPrep.hpp"

template<class T>
Hemi<T>::~Hemi()
{
    for (auto& x : matrix_preps)
        delete x.second;
}

template<class T>
typename T::MatrixPrep& Hemi<T>::get_matrix_prep(const array<int, 3>& dims,
        SubProcessor<T>& processor)
{
    if (matrix_preps.find(dims) == matrix_preps.end())
        matrix_preps.insert({dims,
            new typename T::MatrixPrep(dims[0], dims[1], dims[2],
                    dynamic_cast<typename T::LivePrep&>(processor.DataF),
                    matrix_usage)});
    return *matrix_preps.at(dims);
}

template<class T>
void Hemi<T>::matmulsm(SubProcessor<T>& processor, CheckVector<T>& source,
        const Instruction& instruction, int a, int b)
{
    if (HemiOptions::singleton.plain_matmul
            or not OnlineOptions::singleton.live_prep)
    {
        processor.matmulsm(source, instruction, a, b);
        return;
    }

    auto& dim = instruction.get_start();
    auto& S = processor.get_S();
    auto C = S.begin() + (instruction.get_r(0));
    assert(C + dim[0] * dim[2] <= S.end());
    auto Proc = processor.Proc;
    assert(Proc);

    ShareMatrix<T> A(dim[0], dim[1]), B(dim[1], dim[2]);

    if (not T::real_shares(processor.P))
    {
        matrix_multiply(A, B, processor);
        return;
    }

    for (int i = 0; i < dim[0]; i++)
        for (int k = 0; k < dim[1]; k++)
        {
            auto kk = Proc->get_Ci().at(dim[4] + k);
            auto ii = Proc->get_Ci().at(dim[3] + i);
            A.entries.v.push_back(source.at(a + ii * dim[7] + kk));
        }

    for (int k = 0; k < dim[1]; k++)
        for (int j = 0; j < dim[2]; j++)
        {
            auto jj = Proc->get_Ci().at(dim[6] + j);
            auto ll = Proc->get_Ci().at(dim[5] + k);
            B.entries.v.push_back(source.at(b + ll * dim[8] + jj));
        }

    auto res = matrix_multiply(A, B, processor);

    for (int i = 0; i < dim[0]; i++)
        for (int j = 0; j < dim[2]; j++)
            *(C + i * dim[2] + j) = res[{i, j}];
}

template<class T>
ShareMatrix<T> Hemi<T>::matrix_multiply(const ShareMatrix<T>& A,
        const ShareMatrix<T>& B, SubProcessor<T>& processor)
{
    Beaver<ShareMatrix<T>> beaver(this->P);
    array<int, 3> dims = {{A.n_rows, A.n_cols, B.n_cols}};
    ShareMatrix<T> C(A.n_rows, B.n_cols);

    int max_inner = OnlineOptions::singleton.batch_size;
    int max_cols = OnlineOptions::singleton.batch_size;
    for (int i = 0; i < A.n_cols; i += max_inner)
    {
        for (int j = 0; j < B.n_cols; j += max_cols)
        {
            auto subdim = dims;
            subdim[1] = min(max_inner, A.n_cols - i);
            subdim[2] = min(max_cols, B.n_cols - j);
            auto& prep = get_matrix_prep(subdim, processor);
            beaver.init(prep, mc);
            beaver.init_mul();
            bool for_real = T::real_shares(processor.P);
            beaver.prepare_mul(A.from(0, i, subdim.data(), for_real),
                    B.from(i, j, subdim.data() + 1, for_real));
            if (for_real)
            {
                beaver.exchange();
                C.add_from_col(j, beaver.finalize_mul());
            }
        }
    }

    return C;
}

/**
 * Reduce convolution to matrix multiplication
 */
template<class T>
void Hemi<T>::conv2ds(SubProcessor<T>& processor,
        const Instruction& instruction)
{
    if (HemiOptions::singleton.plain_matmul
            or not OnlineOptions::singleton.live_prep)
    {
        processor.conv2ds(instruction);
        return;
    }

    auto& args = instruction.get_start();
    int output_h = args[0], output_w = args[1];
    int inputs_h = args[2], inputs_w = args[3];
    int weights_h = args[4], weights_w = args[5];
    int stride_h = args[6], stride_w = args[7];
    int n_channels_in = args[8];
    int padding_h = args[9];
    int padding_w = args[10];
    int batch_size = args[11];
    size_t r0 = instruction.get_r(0);
    size_t r1 = instruction.get_r(1);
    int r2 = instruction.get_r(2);
    int filter_stride_h = 1;
    int filter_stride_w = 1;
    if (stride_h < 0)
    {
        filter_stride_h = -stride_h;
        stride_h = 1;
    }
    if (stride_w < 0)
    {
        filter_stride_w = -stride_w;
        stride_w = 1;
    }

    auto& S = processor.get_S();
    array<int, 3> dim({{1, weights_h * weights_w * n_channels_in, batch_size * output_h * output_w}});
    ShareMatrix<T> A(dim[0], dim[1]), B(dim[1], dim[2]);

    if (not T::real_shares(processor.P))
    {
        matrix_multiply(A, B, processor);
        return;
    }

    A.entries.init();
    B.entries.init();

    for (int i_batch = 0; i_batch < batch_size; i_batch ++)
    {
        size_t base = r1 + i_batch * inputs_w * inputs_h * n_channels_in;
        assert(base + inputs_w * inputs_h * n_channels_in <= S.size());
        T* input_base = &S[base];
        for (int out_y = 0; out_y < output_h; out_y++)
            for (int out_x = 0; out_x < output_w; out_x++)
            {
                int in_x_origin = (out_x * stride_w) - padding_w;
                int in_y_origin = (out_y * stride_h) - padding_h;

                for (int filter_y = 0; filter_y < weights_h; filter_y++)
                {
                    int in_y = in_y_origin + filter_y * filter_stride_h;
                    if ((0 <= in_y) and (in_y < inputs_h))
                        for (int filter_x = 0; filter_x < weights_w; filter_x++)
                        {
                            int in_x = in_x_origin + filter_x * filter_stride_w;
                            if ((0 <= in_x) and (in_x < inputs_w))
                            {
                                T* pixel_base = &input_base[(in_y * inputs_w
                                        + in_x) * n_channels_in];
                                T* weight_base = &S[r2
                                        + (filter_y * weights_w + filter_x)
                                                * n_channels_in];
                                for (int in_c = 0; in_c < n_channels_in; in_c++)
//                                    protocol.prepare_dotprod(pixel_base[in_c],
//                                            weight_base[in_c])
                                {
                                    int i_inner = n_channels_in * (filter_x * weights_h + filter_y) + in_c;
                                    B[{i_inner, output_h * (output_w * i_batch + out_x) + out_y}] = pixel_base[in_c];
                                    A[{0, i_inner}] = weight_base[in_c];
                                }
                            }
                        }
                }
//
//                protocol.next_dotprod();
            }
    }

    auto C = matrix_multiply(A, B, processor);

    for (int i_batch = 0; i_batch < batch_size; i_batch ++)
    {
        size_t base = r0 + i_batch * output_h * output_w;
        assert(base + output_h * output_w <= S.size());
        T* output_base = &S[base];
        for (int out_y = 0; out_y < output_h; out_y++)
            for (int out_x = 0; out_x < output_w; out_x++)
            {
                output_base[out_y * output_w + out_x] = C[{0, output_h * (output_w * i_batch + out_x) + out_y}];
//                        protocol.finalize_dotprod(
//                                lengths[i_batch][out_y][out_x]);
            }
    }

}

#endif /* PROTOCOLS_HEMI_HPP_ */
