/*
 * Beaver.cpp
 *
 */

#ifndef PROTOCOLS_BEAVER_HPP_
#define PROTOCOLS_BEAVER_HPP_

#include "Beaver.h"

#include "Replicated.hpp"

#include <array>

template<class T>
typename T::Protocol Beaver<T>::branch()
{
    typename T::Protocol res(P);
    res.prep = prep;
    res.MC = MC;
    res.init_mul();
    return res;
}

template<class T>
void Beaver<T>::init(Preprocessing<T>& prep, typename T::MAC_Check& MC)
{
    this->prep = &prep;
    this->MC = &MC;
}

template<class T>
void Beaver<T>::init_mul()
{
    assert(this->prep);
    assert(this->MC);
    shares.clear();
    opened.clear();
    triples.clear();
    lengths.clear();
}

template<class T>
void Beaver<T>::prepare_mul(const T& x, const T& y, int n)
{
    (void) n;
    triples.push_back({{}});
    auto& triple = triples.back();
    triple = prep->get_triple(n);
    shares.push_back(x - triple[0]);
    shares.push_back(y - triple[1]);
    lengths.push_back(n);
}

template<class T>
void Beaver<T>::exchange()
{
    assert(shares.size() == 2 * lengths.size());
    MC->init_open(P, shares.size());
    for (size_t i = 0; i < shares.size(); i++)
        MC->prepare_open(shares[i], lengths[i / 2]);
    MC->exchange(P);
    for (size_t i = 0; i < shares.size(); i++)
        opened.push_back(MC->finalize_raw());
    it = opened.begin();
    triple = triples.begin();
}

template<class T>
void Beaver<T>::start_exchange()
{
    MC->POpen_Begin(opened, shares, P);
}

template<class T>
void Beaver<T>::stop_exchange()
{
    MC->POpen_End(opened, shares, P);
    it = opened.begin();
    triple = triples.begin();
}

template<class T>
T Beaver<T>::finalize_mul(int n)
{
    (void) n;
    typename T::open_type masked[2];
    T& tmp = (*triple)[2];
    for (int k = 0; k < 2; k++)
    {
        masked[k] = *it++;
    }
    tmp += (masked[0] * (*triple)[1]);
    tmp += ((*triple)[0] * masked[1]);
    tmp += T::constant(masked[0] * masked[1], P.my_num(), MC->get_alphai());
    triple++;
    return tmp;
}

template<class T>
void Beaver<T>::check()
{
    assert(MC);
    MC->Check(P);
}

#endif
