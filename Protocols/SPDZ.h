/*
 * Multiplication.h
 *
 */

#ifndef PROTOCOLS_SPDZ_H_
#define PROTOCOLS_SPDZ_H_

#include "Beaver.h"

#include <vector>
using namespace std;

template<class T> class SubProcessor;
template<class T> class Share;
class Player;

/**
 * SPDZ protocol
 */
template <class T>
class SPDZ : public Beaver<T>
{
public:
    SPDZ(Player& P) : Beaver<T>(P)
    {
    }

    static void assign(typename T::open_type& share,
            const typename T::open_type& clear, int my_num)
    {
        if (my_num == 0)
            share = clear;
        else
            share = 0;
    }

    int get_n_relevant_players()
    {
        return this->P.num_players();
    }
};

#endif /* PROTOCOLS_SPDZ_H_ */
