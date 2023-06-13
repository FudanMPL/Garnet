/*
 * Rep4MC.h
 *
 */

#ifndef PROTOCOLS_REP4MC_H_
#define PROTOCOLS_REP4MC_H_

#include "MAC_Check_Base.h"

template<class T>
class Rep4MC : public MAC_Check_Base<T>
{
    Hash check_hash, receive_hash;

public:
    Rep4MC(typename T::mac_key_type = {}, int = 0, int = 0)
    {
    }

    void exchange(const Player& P);
    void Check(const Player& P);

    Rep4MC& get_part_MC()
    {
        return *this;
    }
};

#endif /* PROTOCOLS_REP4MC_H_ */
