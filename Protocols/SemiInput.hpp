/*
 * SemiInput.cpp
 *
 */

#ifndef PROTOCOLS_SEMIINPUT_HPP_
#define PROTOCOLS_SEMIINPUT_HPP_

#include "SemiInput.h"

#include "ShamirInput.hpp"

template<class T>
SemiInput<T>::SemiInput(SubProcessor<T>* proc, PlayerBase& P) :
        PairwiseKeyInput<T>(proc, P), P(P)
{
    this->reset_all(P);
}

template<class T>
PairwiseKeyInput<T>::PairwiseKeyInput(SubProcessor<T>* proc, PlayerBase& P) :
        PrepLessInput<T>(proc)
{
    vector<octetStream> to_send(P.num_players()), to_receive;
    for (int i = 0; i < P.num_players(); i++)
    {
        send_prngs.push_back({});
        to_send[i].append(send_prngs.back().get_seed(), SEED_SIZE);
    }
    P.send_receive_all(to_send, to_receive);
    recv_prngs.resize(P.num_players());
    for (int i = 0; i < P.num_players(); i++)
        if (i != P.my_num())
            recv_prngs[i].SetSeed(to_receive[i].consume(SEED_SIZE));
}

template<class T>
void SemiInput<T>::reset(int player)
{
    if (player == P.my_num())
        this->shares.clear();
}

template<class T>
void SemiInput<T>::add_mine(const typename T::clear& input, int)
{
	auto& P = this->P;
	typename T::open_type sum, share;
	for (int i = 0; i < P.num_players(); i++)
	{
	    if (i != P.my_num())
	        sum += this->send_prngs[i].template get<typename T::open_type>();
	}
	this->shares.push_back(input - sum);
}

template<class T>
void SemiInput<T>::add_other(int, int)
{
}

template<class T>
void SemiInput<T>::exchange()
{
}

template<class T>
void SemiInput<T>::finalize_other(int player, T& target, octetStream&,
        int)
{
    target = this->recv_prngs[player].template get<T>();
}

template<class T>
T SemiInput<T>::finalize_mine()
{
    return this->shares.next();
}

#endif
