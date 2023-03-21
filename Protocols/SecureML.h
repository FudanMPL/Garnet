/*
 * Hemi.h
 *
 */

#ifndef PROTOCOLS_SML_H_
#define PROTOCOLS_SML_H_

#include "Semi.h"
#include "ShareMatrix.h"
#include "HemiOptions.h"

#include "SmlMatrixPrep.h"
// #include "HemiPrep.hpp"

/**
 * Matrix multiplication
 */
template<class T>
class SecureML : public Semi<T>
{
    map<array<int, 3>, SmlMatrixPrep<T>*> matrix_preps;
    DataPositions matrix_usage;

    MatrixMC<T> mc;

public:
    SecureML(Player& P) :
            Semi<T>(P)
    {
    }
    // ~SecureML();

    void matmulsm(SubProcessor<T> & processor, CheckVector<T>& source,
            const Instruction& instruction, int a, int b)
    { 
        cout<<"this uses matmulsm"<<endl;

        auto& dim = instruction.get_start();
        auto& S = processor.get_S();
        cout<<"mat_dim: "<<dim[0]<<' '<<dim[1]<<' '<<dim[2]<<endl;

        auto C = S.begin() + (instruction.get_r(0));
        assert(C + dim[0] * dim[2] <= S.end());

        auto Proc = processor.Proc;
        assert(Proc);
        
        ShareMatrix<T> A(dim[0], dim[1]), B(dim[1], dim[2]);

        cout<<"matrix A:"<<endl;
        for (int i = 0; i < dim[0]; i++){
            for (int k = 0; k < dim[1]; k++)
            {
                auto kk = Proc->get_Ci().at(dim[4] + k);
                auto ii = Proc->get_Ci().at(dim[3] + i);
                A.entries.v.push_back(source.at(a + ii * dim[7] + kk));
                cout<<source.at(a + ii * dim[7] + kk)<<' ';
            }
            cout<<endl;
        }
            

        for (int k = 0; k < dim[1]; k++){
            for (int j = 0; j < dim[2]; j++)
            {
                auto jj = Proc->get_Ci().at(dim[6] + j);
                auto ll = Proc->get_Ci().at(dim[5] + k);
                B.entries.v.push_back(source.at(b + ll * dim[8] + jj));
            }
        }
            

        auto res = matrix_multiply(A, B, processor);

        for (int i = 0; i < dim[0]; i++)
            for (int j = 0; j < dim[2]; j++)
                *(C + i * dim[2] + j) = res[{i, j}];

        // processor.matmulsm(source, instruction, a, b);

        
    }

    ShareMatrix<T> matrix_multiply(const ShareMatrix<T>& A,
            const ShareMatrix<T>& B, SubProcessor<T>& processor)
    {
        std::cout<<"## this uses matrix_multiply"<<endl;
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
                std::cout<<"## beaver's init finished"<<endl;
                
                beaver.init_mul();
                std::cout<<"## beaver's initmul finished"<<endl;

                bool for_real = T::real_shares(processor.P);
                beaver.prepare_mul(A.from(0, i, subdim.data(), for_real),
                        B.from(i, j, subdim.data() + 1, for_real));
                std::cout<<"## beaver's prepare_mul finished"<<endl;

                if (for_real)
                {
                    beaver.exchange();
                    std::cout<<"## beaver's exchange finished"<<endl;
                    C.add_from_col(j, beaver.finalize_mul());
                    std::cout<<"## beaver's finalize_mul finished"<<endl;
                }
            }
        }
        return C;
    }

    SmlMatrixPrep<T>& get_matrix_prep(const array<int, 3>& dims,
            SubProcessor<T>& processor)
    {
        if (matrix_preps.find(dims) == matrix_preps.end())
            matrix_preps.insert({dims,
                new SmlMatrixPrep<T>(dims[0], dims[1], dims[2],
                        dynamic_cast<typename T::LivePrep&>(processor.DataF),
                        matrix_usage)});
        return *matrix_preps.at(dims);
    }
    // void conv2ds(SubProcessor<T>& processor, const Instruction& instruction);
};

#endif /* PROTOCOLS_SML_H_ */
