/*
 * Rep4MC.hpp
 *
 */

#ifndef PROTOCOLS_REP4MC_HPP_
#define PROTOCOLS_REP4MC_HPP_

#include "Rep4MC.h"

template<class T>
void Rep4MC<T>::exchange(const Player& P)
{
    octetStream right, tmp;
    for (auto& secret : this->secrets)
    {
        secret[0].pack(right);
        secret[2].pack(tmp);
    }
    check_hash.update(tmp);
    P.pass_around(right, 1);
    this->values.resize(this->secrets.size());
    for (size_t i = 0; i < this->secrets.size(); i++)
    {
        typename T::open_type a, b;
        a.unpack(right);
        this->values[i] = this->secrets[i].sum() + a;
    }
    receive_hash.update(right);
}

template<class T>
void Rep4MC<T>::Check(const Player& P)
{
    octetStream left;
    check_hash.final(left);
    P.pass_around(left, -1);
    octetStream os;
    receive_hash.final(os);
    if (os != left)
        throw mac_fail();
}

#endif /* PROTOCOLS_REP4MC_HPP_ */
