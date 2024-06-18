/*
 * VssField.h
 *
 */

#ifndef PROTOCOLS_VSSFIELD_H_
#define PROTOCOLS_VSSFIELD_H_

#include "Semi.h"
#include "ShareMatrix.h"
#include "HemiOptions.h"
#include "VssMatrixPrep.h"
#include "../Processor/Conv2dTuple.h"
#include "../Processor/MatmulsmTuple.h"
#include "VssBeaver.h"

template <class T>
class VssField : public Semi<T>
{
    typedef typename T::MAC_Check MAC_Check;
    map<array<int, 3>, VssMatrixPrep<T> *> matrix_preps;
    DataPositions matrix_usage;
    MatrixMC<T> mc;
    vector<vector<int>> public_matrix;
    vector<typename T::open_type> field_inv; // 恢复系数
public:
    VssField(Player &P) : Semi<T>(P)
    {
    }
    
    // 求矩阵的行列式
    Integer determinant(vector<vector<int>> &matrix)
    {
        int n = matrix.size();
        if (n == 2)
        {
            Integer det = (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]);
            return det;
        }
        Integer det = 0;
        bool sign = true;
        for (int i = 0; i < n; i++)
        {
            vector<vector<int>> submatrix(n - 1, vector<int>(n - 1));
            for (int j = 1; j < n; j++)
            {
                int col = 0;
                for (int k = 0; k < n; k++)
                {
                    if (k != i)
                    {
                        submatrix[j - 1][col] = matrix[j][k];
                        col++;
                    }
                }
            }
            if (sign == true)
                det = det + (determinant(submatrix) * matrix[0][i]);
            else
                det = det - (determinant(submatrix) * matrix[0][i]);
            sign = !sign;
        }
        return det;
    }

    // 求矩阵的伴随矩阵
    vector<vector<typename T::open_type>> adjointMatrix(vector<vector<int>> &matrix)
    {
        int n = matrix.size();
        vector<vector<typename T::open_type>> adj(n, vector<typename T::open_type>(n));
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                vector<vector<int>> submatrix(n - 1, vector<int>(n - 1));
                int subi = 0, subj = 0;
                for (int k = 0; k < n; k++)
                {
                    if (k != i)
                    {
                        subj = 0;
                        for (int l = 0; l < n; l++)
                        {
                            if (l != j)
                            {
                                submatrix[subi][subj] = matrix[k][l];
                                subj++;
                            }
                        }
                        subi++;
                    }
                }
                int sign = ((i + j) % 2 == 0) ? 1 : -1;
                adj[j][i] = Integer(sign) * determinant(submatrix);
            }
        }
        return adj;
    }

    T finalize_mul(int n =-1) override
    {
        (void)n;
        typename T::open_type masked[2];

        int public_matrix_row = this->P.num_players(); // n+nd
        // int public_matrix_col = P.num_players() - ndparties; // n
        int public_matrix_col = this->P.num_players(); // n+nd

        public_matrix.resize(public_matrix_row);
        field_inv.resize(public_matrix_col);

        for (int i = 0; i < public_matrix_row; i++)
        {
            public_matrix[i].resize(public_matrix_col);
        }
        for (int i = 0; i < public_matrix_row; i++)
        {
            int x = 1;
            public_matrix[i][0] = 1;
            for (int j = 1; j < public_matrix_col; j++)
            {
                x *= (i + 1);
                public_matrix[i][j] = x;
            }
        }

        vector<vector<int>> selected(public_matrix.begin(), public_matrix.begin() + public_matrix_col);
        typename T::open_type det = determinant(selected);                   // 行列式
        typename T::open_type det_inv = det.invert();                        // 行列式的逆
        vector<vector<typename T::open_type>> adj = adjointMatrix(selected); // 伴随矩阵
        for (int i = 0; i < public_matrix_col; i++)
        {
            field_inv[i] = adj[0][i] * det_inv; // 逆矩阵的第一行
        }

        T &tmp = (*this->triple)[2];
        for (int k = 0; k < 2; k++)
        {
            masked[k] = *this->it++; 
        }
        tmp += (masked[0] * (*this->triple)[1]);
        tmp += ((*this->triple)[0] * masked[1]);
        tmp += T::constant((masked[0] * masked[1]) / field_inv[0], this->P.my_num(), this->MC->get_alphai()); // P0的是正确的，P1和P2的是错误的

        this->triple++;
        return tmp;
    }

    void matmulsm(SubProcessor<T>& processor, CheckVector<T>& source,
            const Instruction& instruction)
    {
        if (HemiOptions::singleton.plain_matmul
                or not OnlineOptions::singleton.live_prep)
        {
            processor.matmulsm(source, instruction);
            return;
        }

        auto& args = instruction.get_start();
        vector<MatmulsmTuple> tuples;
        for (size_t i = 0; i < args.size(); i += 12)
            tuples.push_back(MatmulsmTuple(args, processor.Proc, i));
        for (auto& tuple : tuples)
            tuple.run_matrix(processor, source);
    }

    ShareMatrix<T> matrix_multiply(const ShareMatrix<T> &A,
                                   const ShareMatrix<T> &B, SubProcessor<T> &processor)
    {
        VssBeaver<ShareMatrix<T>> beaver(this->P);
        array<int, 3> dims = {{A.n_rows, A.n_cols, B.n_cols}};
        ShareMatrix<T> C(A.n_rows, B.n_cols);

        int max_inner = OnlineOptions::singleton.batch_size;
        int max_cols = OnlineOptions::singleton.batch_size;
        // cout << "A.n_cols" << A.n_cols << endl;
        // cout << "B.n_cols" << B.n_cols << endl;
        for (int i = 0; i < A.n_cols; i += max_inner)
        {
            for (int j = 0; j < B.n_cols; j += max_cols)
            {
                auto subdim = dims;
                subdim[1] = min(max_inner, A.n_cols - i);
                subdim[2] = min(max_cols, B.n_cols - j);

                auto &prep = get_matrix_prep(subdim, processor);
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

    VssMatrixPrep<T> &get_matrix_prep(const array<int, 3> &dims,
                                      SubProcessor<T> &processor)
    {
        if (matrix_preps.find(dims) == matrix_preps.end())
            matrix_preps.insert({dims,
                                 new VssMatrixPrep<T>(dims[0], dims[1], dims[2],
                                                      dynamic_cast<typename T::LivePrep &>(processor.DataF),
                                                      matrix_usage, this->P)});
        return *matrix_preps.at(dims);
    }

   void conv2ds(SubProcessor<T>& processor,
        const Instruction& instruction)
        {
            if (HemiOptions::singleton.plain_matmul
                    or not OnlineOptions::singleton.live_prep)
            {
                processor.conv2ds(instruction);
                return;
            }

            auto& args = instruction.get_start();
            vector<Conv2dTuple> tuples;
            for (size_t i = 0; i < args.size(); i += 16)
                tuples.push_back(Conv2dTuple(args, i));
            for (auto& tuple : tuples)
                tuple.run_matrix(processor);
        }
};

#endif