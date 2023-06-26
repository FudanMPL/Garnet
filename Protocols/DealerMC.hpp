/*
 * DealerMC.hpp
 *
 */

#ifndef PROTOCOLS_DEALERMC_HPP_
#define PROTOCOLS_DEALERMC_HPP_

#include "DealerMC.h"

template<class T>
DealerMC<T>::DealerMC(typename T::mac_key_type, int, int) :
        DealerMC(*(new internal_type))
{
}

template<class T>
DirectDealerMC<T>::DirectDealerMC(typename T::mac_key_type) :
        DealerMC<T>(*(new DirectSemiMC<SemiShare<typename T::clear>>))
{
}

template<class T>
DealerMC<T>::DealerMC(internal_type& internal) :
        internal(internal), sub_player(0)
{
}

template<class T>
DealerMC<T>::~DealerMC()
{
    delete &internal;
    if (sub_player)
        delete sub_player;
}

template<class T>
void DealerMC<T>::init_open(const Player& P, int n)
{
    if (P.my_num() != P.num_players() - 1)
    {
        if (not sub_player)
            sub_player = new AllButLastPlayer(P);
        internal.init_open(P, n);
    }
}

template<class T>
void DealerMC<T>::prepare_open(const T& secret, int n_bits)
{
    if (sub_player)
        internal.prepare_open(secret, n_bits);
    else
    {
        if (secret != T())
            throw runtime_error("share for dealer should be 0");
    }
}

template<class T>
void DealerMC<T>::exchange(const Player&)
{
    if (sub_player)
        internal.exchange(*sub_player);
}

template<class T>
typename T::open_type DealerMC<T>::finalize_raw()
{
    if (sub_player)
        return internal.finalize_raw();
    else
        return {};
}

template<class T>
array<typename T::open_type*, 2> DealerMC<T>::finalize_several(int n)
{
    assert(sub_player);
    return internal.finalize_several(n);
}

#endif /* PROTOCOLS_DEALERMC_HPP_ */
