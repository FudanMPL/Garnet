/*
 * SmlMatrixPrep.h
 *
 */

#ifndef PROTOCOLS_SMLMATRIXPREP_H_
#define PROTOCOLS_SMLMATRIXPREP_H_

#include "ShareMatrix.h"
#include "ReplicatedPrep.h"
#include "Tools/Bundle.h"
#include "Processor/BaseMachine.h"


// template<class T> class HemiPrep;
template<class V>
class MatrixRandMulJob : public ThreadJob
{
public:
    MatrixRandMulJob(vector<ValueMatrix<V>>& C,
            const vector<ValueMatrix<V>>& A,
            vector<ValueMatrix<V>>& B,
            bool local_mul)
    {
        type = MATRX_RAND_MULT_JOB;
        output = &C;
        input = &A;
        supply = &B;
        length = local_mul;
    }
};

template<class V>
inline void matrix_rand_mul(ThreadJob job, true_type = {})
{
    auto& C = *(vector<ValueMatrix<V>>*) job.output;
    auto& A = *(vector<ValueMatrix<V>>*) job.input;
    auto& B = *(vector<ValueMatrix<V>>*) job.supply;
    SeededPRNG G;
    for (int i = job.begin; i < job.end; i++)
    {
        A[i].randomize(G);
        B[i].randomize(G);
        if (job.length)
            C[i] = A[i] * B[i];
    }
}

/**
 * matrix triple generation using OT
 */
template<class T>
class SmlMatrixPrep : public BufferPrep<ShareMatrix<T>>
{
    typedef BufferPrep<ShareMatrix<T>> super;
    typedef typename T::LivePrep LivePrep;
    typedef typename T :: Dtype Dtype;

    int n_rows, n_inner, n_cols;
    bool swapped;

    LivePrep* prep;

public:
    SmlMatrixPrep(int n_rows, int n_inner, int n_cols, 
            LivePrep& prep,
            DataPositions& usage) :
            super(usage), n_rows(n_rows), n_inner(n_inner),
            n_cols(n_cols), prep(&prep)
    {
        swapped = n_rows > n_cols;
        if (swapped)
            std::swap(this->n_rows, this->n_cols);
        assert(this->n_cols >= this->n_rows);
    }

    void set_protocol(typename ShareMatrix<T>::Protocol&)
    {
    }

    void buffer_triples(){
        int n_matrices = 1;

        AddableVector<ValueMatrix<Dtype>> A(n_matrices, {n_rows, n_inner});
        AddableVector<ValueMatrix<Dtype>> B(n_matrices, {n_inner, n_cols});
        AddableVector<ValueMatrix<Dtype>> C(n_matrices, {n_rows, n_cols});
        SeededPRNG G;

        for (int i=0; i<n_matrices; i++){
            A[i].randomize(G);
            B[i].randomize(G);
            C[i].randomize(G);
            for (int j=0; j<n_rows; j++)
                for (int k=0; k<n_cols; k++)
                    C[i][{j,k}]=0;
        }
                
        // TRIPLE FOR MAT GENERATING
        
        for (int i = 0; i < n_matrices; i++){
            assert(prep);
            auto& nTriplesPerLoop = prep->triple_generator->nPreampTriplesPerLoop;
            cout<<"nTriplesPerLoop: "<<nTriplesPerLoop<<endl;
            int Loops =  DIV_CEIL(n_rows * n_inner * n_cols , nTriplesPerLoop);
            cout<<"Loops: "<<Loops<<endl;
            // // native generating
            // for (int r = 0; r < n_rows; r++){
            //     for (int c = 0; c < n_cols; c++){
            //         for (int k = 0; k < n_inner; k++){
            //             prep->triple_generator->generateMyTriples(
            //                     A[i][{r,k}], B[i][{k,c}]);
            //             prep->triple_generator->unlock();
            //             for (auto x:prep->triple_generator->plainTriples){
            //                 C[i][{r,c}]+=x[2];
            //             }
            //         }
            //     }
            // }

            // vectorisze generating
            for (int k = 0; k < Loops; k++){
                C[i] = prep->triple_generator->generateMatrixTriples(k, n_rows, n_inner, n_cols, A[i], B[i], C[i]);
                prep->triple_generator->unlock();
            }

            // cout<<"========== matTriple debug A ========="<<endl;
            //     for (int j=0; j<n_rows; j++){
            //         cout<<"[";
            //         for (int k=0; k<n_inner; k++){
            //             cout<<A[i][{j,k}]<<',';
            //         }
            //         cout<<"],"<<endl;
            //     }
            // cout<<"========== matTriple debug B ========="<<endl;
            //     for (int j=0; j<n_inner; j++){
            //         cout<<"[";
            //         for (int k=0; k<n_cols; k++){
            //             cout<<B[i][{j,k}]<<',';
            //         }
            //         cout<<"],"<<endl;
            //     }
            // cout<<"========== matTriple debug C ========="<<endl;
            //     for (int j=0; j<n_rows; j++){
            //         cout<<"[";
            //         for (int k=0; k<n_cols; k++){
            //             cout<<C[i][{j,k}]<<',';
            //         }
            //         cout<<"],"<<endl;
            //     } 


            if (swapped)
                this->triples.push_back(
                        {{B[i].transpose(), A[i].transpose(), C[i].transpose()}});
            else
                this->triples.push_back({{A[i], B[i], C[i]}});
        }
    }

    
};



#endif /* PROTOCOLS_SMLMATRIXPREP_H_ */

