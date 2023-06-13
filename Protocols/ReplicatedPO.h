/*
 * ReplicatedPO.h
 *
 */

#ifndef PROTOCOLS_REPLICATEDPO_H_
#define PROTOCOLS_REPLICATEDPO_H_

#include "MaliciousRepPO.h"

template<class T>
class ReplicatedPO : public MaliciousRepPO<T>
{
public:
    ReplicatedPO(Player& P) :
            MaliciousRepPO<T>(P)
    {
    }

    void send(int player);
    void receive();
};

#endif /* PROTOCOLS_REPLICATEDPO_H_ */
