/*
 * HemiShare.h
 *
 */

#ifndef PROTOCOLS_HEMISHARE_H_
#define PROTOCOLS_HEMISHARE_H_

#include "SemiShare.h"

template<class T> class HemiPrep;
template<class T> class Hemi;
template<class T> class HemiMatrixPrep;

template<class T>
class HemiShare : public SemiShare<T>
{
    typedef HemiShare This;
    typedef SemiShare<T> super;

public:
    typedef SemiMC<This> MAC_Check;
    typedef DirectSemiMC<This> Direct_MC;
    typedef SemiInput<This> Input;
    typedef ::PrivateOutput<This> PrivateOutput;
    // matrix multiplication only with prime order field
    typedef typename conditional<T::prime_field, Hemi<This>, Beaver<This>>::type Protocol;
    typedef HemiPrep<This> LivePrep;

    typedef HemiMatrixPrep<This> MatrixPrep;
    typedef Semi<This> BasicProtocol;

    static const bool needs_ot = false;
    static const bool local_mul = true;
    static true_type triple_matmul;

    HemiShare()
    {
    }
    template<class U>
    HemiShare(const U& other) :
            super(other)
    {
    }

};

template<class T>
true_type HemiShare<T>::triple_matmul;

#endif /* PROTOCOLS_HEMISHARE_H_ */
