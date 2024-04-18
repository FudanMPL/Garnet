/*
 * ReplicatedPO.cpp
 *
 */

#ifndef PROTOCOLS_REPLICATEDPO_HPP_
#define PROTOCOLS_REPLICATEDPO_HPP_

#include "ReplicatedPO.h"

#include "MaliciousRepPO.hpp"

template<class T>
void ReplicatedPO<T>::send(int player)
{
    if (this->P.get_offset(player) == 2)
        this->P.send_to(player, this->to_send);
}

template<class T>
void ReplicatedPO<T>::receive()
{
    this->P.receive_relative(1, this->to_receive[0]);
}

#endif /* PROTOCOLS_REPLICATEDPO_HPP_ */