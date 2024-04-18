/*
 * LimitedPrep.cpp
 *
 */

#ifndef PROTOCOLS_LIMITEDPREP_HPP
#define PROTOCOLS_LIMITEDPREP_HPP

#include "LimitedPrep.h"

template<class T>
LimitedPrep<T>::LimitedPrep() :
        BufferPrep<T>(usage)
{
}

template<class T>
LimitedPrep<T>::~LimitedPrep()
{
#ifdef VERBOSE
    if (not this->triples.empty())
        cerr << "Triples left" << endl;
#endif
}

template<class T>
void LimitedPrep<T>::set_protocol(typename T::Protocol& protocol)
{
    usage.set_num_players(protocol.P.num_players());
}

#endif
