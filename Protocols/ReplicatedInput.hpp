/*
 * ReplicatedInput.cpp
 *
 */

#ifndef PROTOCOLS_REPLICATEDINPUT_HPP_
#define PROTOCOLS_REPLICATEDINPUT_HPP_

#include "ReplicatedInput.h"
#include "Processor/Processor.h"

#include "Processor/Input.hpp"

template<class T>
void ReplicatedInput<T>::reset(int player)
{
    InputBase<T>::reset(player);
    assert(P.num_players() == 3);
    if (player == P.my_num())
    {
        this->shares.clear();
        os.resize(2);
        for (auto& o : os)
            o.reset_write_head();
    }
    expect[player] = false;
}

template<class T>
inline void ReplicatedInput<T>::add_mine(const typename T::open_type& input, int n_bits)
{
    auto& shares = this->shares;
    shares.push_back({});
    T& my_share = shares.back();
    my_share[0].randomize(protocol.shared_prngs[0], n_bits);
    my_share[1] = input - my_share[0];
    my_share[1].pack(os[1], n_bits);
    this->values_input++;
}

template<class T>
void ReplicatedInput<T>::add_other(int player, int)
{
    expect[player] = true;
}

template<class T>
void ReplicatedInput<T>::send_mine()
{
    P.send_relative(os);
}

template<class T>
void ReplicatedInput<T>::exchange()
{
    bool receive = expect[P.get_player(1)];
    bool send = not os[1].empty();
    auto& dest =  InputBase<T>::os[P.get_player(1)];
    if (send)
        if (receive)
            P.pass_around(os[1], dest, -1);
        else
            P.send_to(P.get_player(-1), os[1]);
    else
        if (receive)
            P.receive_player(P.get_player(1), dest);
}

template<class T>
inline void ReplicatedInput<T>::finalize_other(int player, T& target,
        octetStream& o, int n_bits)
{
    int offset = player - this->my_num;
    if (offset == 1 or offset == -2)
    {
        typename T::value_type t;
        t.unpack(o, n_bits);
        target[0] = t;
        target[1] = 0;
    }
    else
    {
        target[0] = 0;
        target[1].randomize(protocol.shared_prngs[1], n_bits);
    }
}

template<class T>
T PrepLessInput<T>::finalize_mine()
{
    return this->shares.next();
}

#endif
