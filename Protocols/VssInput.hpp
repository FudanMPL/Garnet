/*
 * VssInput.cpp
 *
 */

#ifndef PROTOCOLS_VssInput_HPP_
#define PROTOCOLS_VssInput_HPP_

#include "VssInput.h"

#include "ShamirInput.hpp"

template <class T>
VssInput<T>::VssInput(SubProcessor<T> *proc, PlayerBase &P) : SemiInput<T>(proc, P), P(P)
{
    P.public_matrix.resize(4);
    P.inv.resize(3);
    for (int i = 0; i < 4; i++)
    {
        P.public_matrix[i].resize(3);
    }
    int array[4][3] = {{1, 0, 1},
                       {2, 2, -3},
                       {3, 3, -4},
                       {1, 1, -1}};
    int array_inv[3] = {1, 3, -2};
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            P.public_matrix[i][j] = array[i][j];
        }
    }
    for(int i = 0; i < 3; i++)
    {
        P.inv[i] = array_inv[i];
    }
    this->reset_all(P);
}

template <class T>
void VssInput<T>::reset(int player)
{
    if (player == P.my_num())
        this->shares.clear();
}

template <class T>
void VssInput<T>::add_mine(const typename T::clear &input, int)
{
    auto &P = this->P;
    typename T::open_type sum;
    std::vector<typename T::open_type> shares(P.num_players());
    for (int i = 0; i < P.num_players(); i++)
    {
        if (i != P.my_num())
        {
            sum += this->send_prngs[i].template get<typename T::open_type>() * P.inv[i];
            // sum += this->send_prngs[i].template get<typename T::open_type>();
        }
    }
    bigint temp = input - sum;
    stringstream ss;
    ss << temp;
    long value = stol(ss.str()) / P.inv[P.my_num()];
    this->shares.push_back(value);
}

template <class T>
void VssInput<T>::add_other(int, int)
{
}

template <class T>
void VssInput<T>::exchange()
{
}

template <class T>
void VssInput<T>::finalize_other(int player, T &target, octetStream &,
                                 int)
{
    target = this->recv_prngs[player].template get<T>();
}

template <class T>
T VssInput<T>::finalize_mine()
{
    return this->shares.next();
}

#endif
