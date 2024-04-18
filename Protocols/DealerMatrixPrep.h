/*
 * DealerMatrixPrep.h
 *
 */

#ifndef PROTOCOLS_DEALERMATRIXPREP_H_
#define PROTOCOLS_DEALERMATRIXPREP_H_

#include "ShareMatrix.h"

template<class T>
class DealerMatrixPrep : public BufferPrep<ShareMatrix<T>>
{
    typedef BufferPrep<ShareMatrix<T>> super;
    typedef typename T::LivePrep LivePrep;

    int n_rows, n_inner, n_cols;

    LivePrep* prep;

public:
    DealerMatrixPrep(int n_rows, int n_inner, int n_cols,
            typename T::LivePrep&, DataPositions& usage);

    void set_protocol(typename ShareMatrix<T>::Protocol&)
    {
    }

    void buffer_triples();
};

#endif /* PROTOCOLS_DEALERMATRIXPREP_H_ */
