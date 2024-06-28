#ifndef PROTOCOLS_VSSMATRIXPREP_H_
#define PROTOCOLS_VSSMATRIXPREP_H_

#include "ShareMatrix.h"
#include "ReplicatedPrep.h"
#include "Tools/Bundle.h"
#include "Processor/BaseMachine.h"
#include "Networking/Player.h"

template <class T>
class VssMatrixPrep : public BufferPrep<ShareMatrix<T>>
{
    typedef BufferPrep<ShareMatrix<T>> super;
    typedef typename T::LivePrep LivePrep;
    typedef typename T::Dtype Dtype;

    int n_rows, n_inner, n_cols;
    bool swapped;

    LivePrep *prep;
    Player *P;

public:
    VssMatrixPrep(int n_rows, int n_inner, int n_cols,
                  LivePrep &prep,
                  DataPositions &usage,
                  Player &P) : super(usage), n_rows(n_rows), n_inner(n_inner),
                               n_cols(n_cols), prep(&prep), P(&P)
    {
        swapped = n_rows > n_cols;
        if (swapped)
            std::swap(this->n_rows, this->n_cols);
        assert(this->n_cols >= this->n_cols);
    }

    void set_protocol(typename ShareMatrix<T>::Protocol &)
    {
    }

    ShareMatrix<T> toVSSMatrixTriples(int rows, int cols, ShareMatrix<T> X)
    {
        // cout<<"toVSSMatrixTriples"<<endl;
        octetStream os, oc;
        int n = P->num_players();
        AddableVector<ValueMatrix<Dtype>> my_share(n, {rows, cols});
        AddableVector<ValueMatrix<Dtype>> Vss_X(n, {rows, cols});
        // ShareMatrix<Dtype> Vss_X(rows, cols);
        ValueMatrix<Dtype> S(n, 1);
        ShareMatrix<T> res(rows, cols);
        SeededPRNG G;
        res.randomize(G);
        for (int i = 0; i < n; i++)
        {
            my_share[i].randomize(G);
            Vss_X[i].randomize(G);
            for (int j = 0; j < rows; j++)
                for (int k = 0; k < cols; k++)
                {
                    my_share[i][{j, k}] = 0;
                    Vss_X[i][{j, k}] = 0;
                }
        }
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                S.randomize(G);
                S[{0, 0}] = X[{i, j}];
                res[{i, j}] = 0;
                for (int k = 0; k < n; k++)
                {
                    for (int l = 0; l < n; l++)
                    {
                        Vss_X[k][{i, j}] += S[{l, 0}] * P->public_matrix[k][l];
                    }
                }
            }
        }
        for (int k = 0; k < n; k++)
        {
            os.reset_write_head();
            oc.reset_read_head();
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    if (P->my_num() != k)
                        Vss_X[k][{i, j}].pack(os);
                    else
                        my_share[k][{i, j}] = Vss_X[k][{i, j}];
                }
            }
            if (P->my_num() != k)
            {
                P->send_to(k, os);
                P->receive_player(k, oc);
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        my_share[k][{i, j}] = oc.get<Dtype>();
                    }
                }
            }
        }
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                for (int k = 0; k < n; k++)
                {
                    res[{i, j}] += my_share[k][{i, j}];
                }
            }
        }
        return res;
    }

    void buffer_triples()
    {
        int n_matrices = 1;
        AddableVector<ValueMatrix<Dtype>> A(n_matrices, {n_rows, n_inner});
        AddableVector<ValueMatrix<Dtype>> B(n_matrices, {n_inner, n_cols});
        AddableVector<ValueMatrix<Dtype>> C(n_matrices, {n_rows, n_cols});
        SeededPRNG G;
        for (int i = 0; i < n_matrices; i++)
        {
            A[i].randomize(G);
            B[i].randomize(G);
            C[i].randomize(G);
            // for (int j = 0; j < n_rows; j++)
            //     for (int k = 0; k < n_inner; k++)
            //         A[i][{j, k}] = 0;
            // for (int j = 0; j < n_inner; j++)
            //     for (int k = 0; k < n_cols; k++)
            //         B[i][{j, k}] = 0;
            for (int j = 0; j < n_rows; j++)
                for (int k = 0; k < n_cols; k++)
                    C[i][{j, k}] = 0;
        }
        // cout << "TRIPLE FOR MAT GENERATING" << endl;
        // TRIPLE FOR MAT GENERATING

        for (int i = 0; i < n_matrices; i++)
        {
            assert(prep);
            auto &nTriplesPerLoop = prep->triple_generator->nPreampTriplesPerLoop;
            // cout << "nTriplesPerLoop: " << nTriplesPerLoop << endl;
            int Loops = DIV_CEIL(n_rows * n_inner * n_cols, nTriplesPerLoop);
            // cout << "Loops: " << Loops << endl;
            // native generating
            // for (int r = 0; r < n_rows; r++)
            // {
            //     for (int c = 0; c < n_cols; c++)
            //     {
            //         for (int k = 0; k < n_inner; k++)
            //         {
            //             prep->triple_generator->generateMyTriples(
            //                 A[i][{r, k}], B[i][{k, c}]);
            //             prep->triple_generator->unlock();
            //             for (auto x : prep->triple_generator->plainTriples)
            //             {
            //                 C[i][{r, c}] += x[2];
            //             }
            //         }
            //     }
            // }
            // A[i] = toVSSMatrixTriples(n_rows, n_inner, A[i]);
            // B[i] = toVSSMatrixTriples(n_inner, n_cols, B[i]);
            // C[i] = toVSSMatrixTriples(n_rows, n_cols, C[i]);
            // vectorisze generating
            for (int k = 0; k < Loops; k++)
            {
                C[i] = prep->triple_generator->generateMatrixTriples(k, n_rows, n_inner, n_cols, A[i], B[i], C[i]);
                A[i] = toVSSMatrixTriples(n_rows, n_inner, A[i]);
                B[i] = toVSSMatrixTriples(n_inner, n_cols, B[i]);
                C[i] = toVSSMatrixTriples(n_rows, n_cols, C[i]);
                prep->triple_generator->unlock();
            }

            // cout << "========== matTriple debug A =========" << endl;
            // for (int j = 0; j < n_rows; j++)
            // {
            //     cout << "[";
            //     for (int k = 0; k < n_inner; k++)
            //     {
            //         cout << A[i][{j, k}] << ',';
            //     }
            //     cout << "]," << endl;
            // }
            // cout << "========== matTriple debug B =========" << endl;
            // for (int j = 0; j < n_inner; j++)
            // {
            //     cout << "[";
            //     for (int k = 0; k < n_cols; k++)
            //     {
            //         cout << B[i][{j, k}] << ',';
            //     }
            //     cout << "]," << endl;
            // }
            // cout << "========== matTriple debug C =========" << endl;
            // for (int j = 0; j < n_rows; j++)
            // {
            //     cout << "[";
            //     for (int k = 0; k < n_cols; k++)
            //     {
            //         cout << C[i][{j, k}] << ',';
            //     }
            //     cout << "]," << endl;
            // }

            if (swapped)
                this->triples.push_back(
                    {{B[i].transpose(), A[i].transpose(), C[i].transpose()}});
            else
                this->triples.push_back({{A[i], B[i], C[i]}});
        }
    }
};

#endif