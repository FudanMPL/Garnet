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
        cout<<"vsize:"<<A[i].entries.v.size()<<endl;
        cout<<"vsize:"<<B[i].entries.v.size()<<endl;
        A[i].randomize(G);
        B[i].randomize(G);
        
        cout<<"random finished!"<<endl;
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
        std::cout<<"## this uses buffer_triples"<<endl;

        int n_matrices = 1;

        AddableVector<ValueMatrix<Dtype>> A(n_matrices, {n_rows, n_inner});
        AddableVector<ValueMatrix<Dtype>> B(n_matrices, {n_inner, n_cols});
        AddableVector<ValueMatrix<Dtype>> C(n_matrices, {n_rows, n_cols});
        SeededPRNG G;

        // cout<<"========== mattriple-1 local_mul ========="<<endl;
        // MatrixRandMulJob<Dtype> job(C, A, B, T::local_mul);
        // if (BaseMachine::thread_num == 0 and BaseMachine::has_singleton())
        // {
        //     auto& queues = BaseMachine::s().queues;
        //     int start = queues.distribute(job, n_matrices);
        //     job.begin = start;
        //     job.end = n_matrices;
        //     matrix_rand_mul<Dtype>(job);
        //     if (start)
        //         queues.wrap_up(job);
        // }
        // else
        // {
        //     job.begin = 0;
        //     job.end = n_matrices;
        //     matrix_rand_mul<Dtype>(job);
        // }

        for (int i=0; i<n_matrices; i++){
            A[i].randomize(G);
            B[i].randomize(G);
            C[i].randomize(G);
        }
                
        cout<<"========== mattriple generating ========="<<endl;
        for (int i = 0; i < n_matrices; i++){
            assert(prep);
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
            for (int r = 0; r < n_rows; r++){
                for (int c = 0; c < n_cols; c++){
                        C[i][{r,c}] = prep->triple_generator->generateMatrixTriples(r, n_inner, c, A[i], B[i]);
                        prep->triple_generator->unlock();
                }
            }
            
            
            cout<<"========== mattriple debug A ========="<<endl;
                for (int j=0; j<n_rows; j++){
                    cout<<"[";
                    for (int k=0; k<n_inner; k++){
                        cout<<A[i][{j,k}]<<',';
                    }
                    cout<<"],"<<endl;
                }
            cout<<"========== mattriple debug B ========="<<endl;
                for (int j=0; j<n_inner; j++){
                    cout<<"[";
                    for (int k=0; k<n_cols; k++){
                        cout<<B[i][{j,k}]<<',';
                    }
                    cout<<"],"<<endl;
                }
            cout<<"========== mattriple debug C ========="<<endl;

                for (int j=0; j<n_rows; j++){
                    cout<<"[";
                    for (int k=0; k<n_cols; k++){
                        cout<<C[i][{j,k}]<<',';
                    }
                    cout<<"],"<<endl;
                } 


            if (swapped)
                this->triples.push_back(
                        {{B[i].transpose(), A[i].transpose(), C[i].transpose()}});
            else
                this->triples.push_back({{A[i], B[i], C[i]}});
        }
             

        std::cout<<"## triples size: "<<this->triples.size()<<endl;
    }

    
};



#endif /* PROTOCOLS_SMLMATRIXPREP_H_ */

