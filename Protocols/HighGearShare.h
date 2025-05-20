/*
 * HighGearShare.h
 *
 */

#ifndef PROTOCOLS_HIGHGEARSHARE_H_
#define PROTOCOLS_HIGHGEARSHARE_H_

#include "ChaiGearShare.h"

template<class T>
class HighGearShare : public ChaiGearShare<T>
{
    typedef HighGearShare This;
    typedef ChaiGearShare<T> super;

public:
    typedef MAC_Check_<This> MAC_Check;
    typedef Direct_MAC_Check<This> Direct_MC;
    typedef ::Input<This> Input;
    typedef ::PrivateOutput<This> PrivateOutput;
    typedef SPDZ<This> Protocol;
    typedef ChaiGearPrep<This> LivePrep;

    const static false_type covert;

    HighGearShare()
    {
    }

    template<class U>
    HighGearShare(const U& other) :
            super(other)
    {
    }
};

template<class T>
const false_type HighGearShare<T>::covert;

#endif /* PROTOCOLS_HIGHGEARSHARE_H_ */
