/*
 * LowGearShare.h
 *
 */

#ifndef PROTOCOLS_LOWGEARSHARE_H_
#define PROTOCOLS_LOWGEARSHARE_H_

#include "CowGearShare.h"

template<class T>
class LowGearShare : public CowGearShare<T>
{
    typedef LowGearShare This;
    typedef CowGearShare<T> super;

public:
    typedef MAC_Check_<This> MAC_Check;
    typedef Direct_MAC_Check<This> Direct_MC;
    typedef ::Input<This> Input;
    typedef ::PrivateOutput<This> PrivateOutput;
    typedef SPDZ<This> Protocol;
    typedef CowGearPrep<This> LivePrep;

    const static false_type covert;

    LowGearShare()
    {
    }

    template<class U>
    LowGearShare(const U& other) :
            super(other)
    {
    }
};

template<class T>
const false_type LowGearShare<T>::covert;

#endif /* PROTOCOLS_LOWGEARSHARE_H_ */
