/*
 * Hemi.h
 *
 */

#ifndef PROTOCOLS_HEMI_H_
#define PROTOCOLS_HEMI_H_

#include "Semi.h"
#include "HemiMatrixPrep.h"
#include "../Processor/MatmulsmTuple.h"
/**
 * Matrix multiplication optimized with semi-homomorphic encryption
 */
template<class T>
class Hemi : public T::BasicProtocol
{
    map<array<int, 3>, typename T::MatrixPrep*> matrix_preps;
    DataPositions matrix_usage;

    MatrixMC<T> mc;



public:
    Hemi(Player& P) :
            T::BasicProtocol(P)
    {
    }
    ~Hemi();

    typename T::MatrixPrep& get_matrix_prep(const array<int, 3>& dimensions,
            SubProcessor<T>& processor);

    void matmulsm(SubProcessor<T>& processor, CheckVector<T>& source,
            const Instruction& instruction);
    void conv2ds(SubProcessor<T>& processor, const Instruction& instruction);
    
    ShareMatrix<T> matrix_multiply(const ShareMatrix<T>& A, const ShareMatrix<T>& B,
            SubProcessor<T>& processor);
};

#endif /* PROTOCOLS_HEMI_H_ */
