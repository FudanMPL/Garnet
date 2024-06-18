/*
 * Beaver.cpp
 *
 */

#ifndef PROTOCOLS_BEAVER_HPP_
#define PROTOCOLS_BEAVER_HPP_

#include "Beaver.h"

#include "Replicated.hpp"

#include <array>

template <class T>
typename T::Protocol Beaver<T>::branch()
{
    typename T::Protocol res(P);
    res.prep = prep;
    res.MC = MC;
    res.init_mul();
    return res;
}

template <class T>
void Beaver<T>::init(Preprocessing<T> &prep, typename T::MAC_Check &MC)
{
    this->prep = &prep;
    this->MC = &MC;
}

template <class T>
void Beaver<T>::init_mul()
{
    assert(this->prep);
    assert(this->MC);
    shares.clear();
    opened.clear();
    triples.clear();
    lengths.clear();
}

template <class T>
void Beaver<T>::prepare_mul(const T &x, const T &y, int n)
{
    (void)n;
    triples.push_back({{}});
    auto &triple = triples.back();
    triple = prep->get_triple(n);
    shares.push_back(x - triple[0]); // shares的值在下面的exchange中被恢复出来了
    shares.push_back(y - triple[1]);
    lengths.push_back(n);
    // cout<<endl;
    // cout<<"Beaver的prepare_mul:"<<endl;
    // cout<<"x:"<<x<<endl; // x = 10
    // cout<<"y:"<<y<<endl; // y = 2
    // cout<<"n:"<<n<<endl; // n = -1
    // cout<<"triple[0]:"<<triple[0]<<endl;
    // cout<<"triple[1]:"<<triple[1]<<endl;
    // cout<<endl;
}

template <class T>
void Beaver<T>::exchange()
{
    // cout<<"Beaver的exchange"<<endl;
    assert(shares.size() == 2 * lengths.size());
    MC->init_open(P, shares.size()); // clear
    for (size_t i = 0; i < shares.size(); i++)
        MC->prepare_open(shares[i], lengths[i / 2]); // 把shares[i]加到values中
    MC->exchange(P);                                 // MAC_Check里的run(values, P),再进入start->rec
    for (size_t i = 0; i < shares.size(); i++)
        opened.push_back(MC->finalize_raw()); // finalize_raw()是values.next();
    it = opened.begin();
    triple = triples.begin();
}

template <class T>
void Beaver<T>::start_exchange()
{
    MC->POpen_Begin(opened, shares, P); // 广播opened中的每个数
}

template <class T>
void Beaver<T>::stop_exchange()
{
    MC->POpen_End(opened, shares, P); // 利用os(P)里的数据rec并存在opened中, 好像没用到shares
    it = opened.begin();
    triple = triples.begin();
}

template <class T>
T Beaver<T>::finalize_mul(int n)
{
    // cout<<endl;
    // cout<<"Beaver的finalize_mul:"<<endl;

    (void)n;
    typename T::open_type masked[2];
    T &tmp = (*triple)[2];
    // cout<<"*triple[2]:"<<(*triple)[2]<<endl;

    for (int k = 0; k < 2; k++)
    {
        masked[k] = *it++; // it是opened的指针，mask存的是opened的值
    }
    tmp += (masked[0] * (*triple)[1]);
    // cout<<"tmp:"<<tmp<<endl;
    tmp += ((*triple)[0] * masked[1]);
    // cout<<"tmp:"<<tmp<<endl;
    tmp += T::constant(masked[0] * masked[1], P.my_num(), MC->get_alphai()); // P0的是正确的，P1和P2的是错误的
    // tmp += (masked[0] * masked[1]);
    // cout<<"masked[0]:"<<masked[0]<<endl;
    // cout<<"masked[1]:"<<masked[1]<<endl;
    // cout<<"*triple[0]:"<<(*triple)[0]<<endl;
    // cout<<"*triple[1]:"<<(*triple)[1]<<endl;
    // cout<<"tmp:"<<tmp<<endl;
    // cout<<endl;

    triple++;
    return tmp;
}

template <class T>
void Beaver<T>::check()
{
    assert(MC);
    MC->Check(P);
}

#endif
