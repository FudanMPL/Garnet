/*
 * HemiMatrixPrep.h
 *
 */

#ifndef PROTOCOLS_HEMIMATRIXPREP_H_
#define PROTOCOLS_HEMIMATRIXPREP_H_

#include "ShareMatrix.h"
#include "ReplicatedPrep.h"

template<class T> class HemiPrep;

/**
 * Semi-honest matrix triple generation using semi-homomorphic encryption
 */
template<class T>
class HemiMatrixPrep : public BufferPrep<ShareMatrix<T>>
{
    typedef BufferPrep<ShareMatrix<T>> super;
    typedef typename T::LivePrep LivePrep;

    int n_rows, n_inner, n_cols;
    bool swapped;

    LivePrep* prep;

    HemiMatrixPrep(const HemiMatrixPrep&) = delete;

public:
    HemiMatrixPrep(int n_rows, int n_inner, int n_cols, LivePrep& prep,
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

    void buffer_triples();
};

#endif /* PROTOCOLS_HEMIMATRIXPREP_H_ */
