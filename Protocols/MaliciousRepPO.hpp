/*
 * MaliciousRepPO.cpp
 *
 */

#ifndef PROTOCOLS_MALICIOUSREPPO_HPP_
#define PROTOCOLS_MALICIOUSREPPO_HPP_

#include "MaliciousRepPO.h"

#include <assert.h>

template<class T>
MaliciousRepPO<T>::MaliciousRepPO(Player& P) : P(P)
{
    assert(P.num_players() == 3);
}

template<class T>
void MaliciousRepPO<T>::prepare_sending(const T& secret, int player)
{
    if (player == P.my_num())
        secrets.push_back(secret);
    else
        secret[2 - P.get_offset(player)].pack(to_send);
}

template<class T>
void MaliciousRepPO<T>::send(int player)
{
    if (P.get_offset(player) == 2)
        P.send_to(player, to_send);
    else if (P.my_num() != player)
        P.send_to(player, to_send.hash());
}

template<class T>
void MaliciousRepPO<T>::receive()
{
    for (int i = 0; i < 2; i++)
        P.receive_relative(i + 1, to_receive[i]);
    if (to_receive[0].hash() != to_receive[1])
        throw mac_fail("mismatch in private output");
}

template<class T>
typename T::clear MaliciousRepPO<T>::finalize(const T& secret)
{
    return secret.sum() + to_receive[0].template get<typename T::open_type>();
}

template<class T>
typename T::clear MaliciousRepPO<T>::finalize()
{
    return finalize(secrets.next());
}

#endif
