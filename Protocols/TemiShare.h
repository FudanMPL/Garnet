/*
 * TemiShare.h
 *
 */

#ifndef PROTOCOLS_TEMISHARE_H_
#define PROTOCOLS_TEMISHARE_H_

#include "HemiShare.h"

template<class T> class TemiPrep;
template<class T> class Hemi;

template<class T>
class TemiShare : public HemiShare<T>
{
    typedef TemiShare This;
    typedef HemiShare<T> super;

public:
    typedef SemiMC<This> MAC_Check;
    typedef DirectSemiMC<This> Direct_MC;
    typedef SemiInput<This> Input;
    typedef ::PrivateOutput<This> PrivateOutput;
    typedef typename conditional<T::prime_field, Hemi<This>, Beaver<This>>::type Protocol;
    typedef TemiPrep<This> LivePrep;

    typedef HemiMatrixPrep<This> MatrixPrep;
    typedef Semi<This> BasicProtocol;

    static const bool needs_ot = false;
    static const bool local_mul = false;

    TemiShare()
    {
    }
    template<class U>
    TemiShare(const U& other) :
            super(other)
    {
    }

};

#endif /* PROTOCOLS_TEMISHARE_H_ */
