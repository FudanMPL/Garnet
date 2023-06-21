/*
 * VssInput.cpp
 *
 */

#ifndef PROTOCOLS_VssInput_HPP_
#define PROTOCOLS_VssInput_HPP_

#include "VssInput.h"

#include "ShamirInput.hpp"

template <class T>
VssInput<T>::VssInput(SubProcessor<T> *proc, Player &P) : SemiInput<T>(proc, P), P(P)
{
    P.public_matrix.resize(4);
    P.inv.resize(3);
    for (int i = 0; i < 4; i++)
    {
        P.public_matrix[i].resize(3);
    }
    os.resize(2);
    os[0].resize(P.public_matrix[0].size());
    os[1].resize(P.public_matrix[0].size());
    expect.resize(P.public_matrix[0].size());
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
    for (int i = 0; i < 3; i++)
    {
        P.inv[i] = array_inv[i];
    }
    this->reset_all(P);
}

template <class T>
void VssInput<T>::reset(int player)
{
    if (player == P.my_num())
    {
        this->shares.clear();
        os.resize(2);
        for (int i = 0; i < 2; i++)
        {
            os[i].resize(P.public_matrix[0].size());
            for (auto &o : os[i])
                o.reset_write_head();
        }
    }
    expect[player] = false;
}

template <class T>
void VssInput<T>::add_mine(const typename T::clear &input, int)
{
    auto &P = this->P;
    vector<typename T::open_type> v(P.public_matrix[0].size());
    vector<T> secrets(P.public_matrix.size());
    PRNG G;
    v[0] = input;
    for (int i = 1; i < P.public_matrix[0].size(); i++)
    {
        v[i] = G.get<typename T::open_type>();
    }
    for (int i = 0; i < P.public_matrix.size(); i++)
    {
        typename T::open_type sum = 0;
        for (int j = 0; j < P.public_matrix[0].size(); j++)
        {
            sum += v[j] * P.public_matrix[i][j];
        }
        secrets[i] = sum;
    }
    this->shares.push_back(secrets[P.my_num()]);
    for (int i = 0; i < P.num_players(); i++)
    {
        if (i != P.my_num())
        {
            secrets[i].pack(os[0][i]);
        }
    }
    // typename T::open_type sum;
    // std::vector<typename T::open_type> shares(P.num_players());
    // for (int i = 0; i < P.num_players(); i++)
    // {
    //     if (i != P.my_num())
    //     {
    //         sum += this->send_prngs[i].template get<typename T::open_type>() * P.inv[i];
    //         // sum += this->send_prngs[i].template get<typename T::open_type>();
    //     }
    // }
    // cout << sum <<endl;
    // bigint temp = input - sum;
    // stringstream ss;
    // ss << temp;
    // long value = stol(ss.str()) / P.inv[P.my_num()];
    // cout << value << endl;
    // this->shares.push_back(value);
}

template <class T>
void VssInput<T>::add_other(int player, int)
{
    expect[player] = true;
}

template <class T>
void VssInput<T>::exchange()
{
    if (!os[0][(P.my_num() + 1) % P.num_players()].empty())
    {
        for (int i = 0; i < P.num_players(); i++)
        {
            if (i != P.my_num())
                P.send_to(i, os[0][i]);
        }
        for (int i = 0; i < P.num_players(); i++)
        {
            if (expect[i])
                P.receive_player(i, os[1][i]);
        }
    }
    else
    {
        for (int i = 0; i < P.num_players(); i++)
        {
            if (expect[i])
                P.receive_player(i, os[1][i]);
        }
    }
}

template <class T>
void VssInput<T>::finalize_other(int player, T &target, octetStream &,
                                 int)
{
    target = os[1][player].template get<T>();
}

template <class T>
T VssInput<T>::finalize_mine()
{
    return this->shares.next();
}

#endif
