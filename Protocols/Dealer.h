/*
 * Dealer.h
 *
 */

#ifndef PROTOCOLS_DEALER_H_
#define PROTOCOLS_DEALER_H_

#include "Beaver.h"

template<class T>
class Dealer : public Beaver<T>
{
    SeededPRNG G;

public:
    Dealer(Player& P) :
            Beaver<T>(P)
    {
    }

    T get_random()
    {
        if (T::real_shares(this->P))
            return G.get<T>();
        else
            return {};
    }

    vector<int> get_relevant_players()
    {
        return vector<int>(1, this->P.num_players() - 1);
    }
};

#endif /* PROTOCOLS_DEALER_H_ */
