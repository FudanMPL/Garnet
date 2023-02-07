/*
 * ChaiGearShare.h
 *
 */

#ifndef PROTOCOLS_CHAIGEARSHARE_H_
#define PROTOCOLS_CHAIGEARSHARE_H_

#include "Share.h"

template<class T> class ChaiGearPrep;

template<class T>
class ChaiGearShare : public Share<T>
{
    typedef ChaiGearShare This;
    typedef Share<T> super;

public:
    typedef MAC_Check_<This> MAC_Check;
    typedef Direct_MAC_Check<This> Direct_MC;
    typedef ::Input<This> Input;
    typedef ::PrivateOutput<This> PrivateOutput;
    typedef SPDZ<This> Protocol;
    typedef ChaiGearPrep<This> LivePrep;
    typedef Share<typename T::FD::T> prep_check_type;

    const static bool needs_ot = false;

    const static true_type covert;

    ChaiGearShare()
    {
    }

    template<class U>
    ChaiGearShare(const U& other) :
            super(other)
    {
    }
};

template<class T>
const true_type ChaiGearShare<T>::covert;

#endif /* PROTOCOLS_CHAIGEARSHARE_H_ */
