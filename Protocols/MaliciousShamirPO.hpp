/*
 * MaliciousShamirPO.cpp
 *
 */

#include "MaliciousShamirPO.h"

template<class T>
MaliciousShamirPO<T>::MaliciousShamirPO(Player& P) :
        P(P), shares(P.num_players())
{
    MC.init_open(P);
}

template<class T>
void MaliciousShamirPO<T>::prepare_sending(const T& secret, int)
{
    secret.pack(to_send);
}

template<class T>
void MaliciousShamirPO<T>::send(int player)
{
    P.send_to(player, to_send);
}

template<class T>
void MaliciousShamirPO<T>::receive()
{
    to_receive.resize(P.num_players());
    for (int i = 0; i < P.num_players(); i++)
        if (i != P.my_num())
            P.receive_player(i, to_receive[i]);
}

template<class T>
typename T::clear MaliciousShamirPO<T>::finalize(const T& secret)
{
    for (int i = 0; i < P.num_players(); i++)
    {
        if (i == P.my_num())
            shares[0] = secret;
        else
            shares[P.get_offset(i)].unpack(to_receive[i]);
    }

    return MC.reconstruct(shares);
}
