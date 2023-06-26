/*
 * FakeMC.h
 *
 */

#ifndef PROTOCOLS_FAKEMC_H_
#define PROTOCOLS_FAKEMC_H_

#include "MAC_Check_Base.h"

template<class T>
class FakeMC : public MAC_Check_Base<T>
{
public:
    FakeMC(typename T::mac_key_type, int = 0, int = 0)
    {
    }

    void exchange(const Player&)
    {
        for (auto& x : this->secrets)
            this->values.push_back(x);
    }

    FakeMC& get_part_MC()
    {
        return *this;
    }
};

#endif /* PROTOCOLS_FAKEMC_H_ */
