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
#include "../Processor/Conv2dTuple.h"
#include "../Processor/MatmulsmTuple.h"
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

    ShareMatrix<T> matrix_multiply(const ShareMatrix<T>& A,
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
#endif /* PROTOCOLS_SML_H_ */
