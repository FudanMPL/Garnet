#ifndef PROTOCOLS_SHARE_H_
#define PROTOCOLS_SHARE_H_

#include "Share.h"


template<class T, class V>
template<class U>
void Share_<T, V>::read_or_generate_mac_key(string directory, const Player& P,
        U& key)
{
    try
    {
        read_mac_key(directory, P.N, key);
    }
    catch (mac_key_error&)
    {
#ifdef VERBOSE
        cerr << "Generating fresh MAC key" << endl;
#endif
        SeededPRNG G;
        key.randomize(G);
    }
}

template<class T, class V>
void Share_<T, V>::randomize(PRNG& G)
{
  a.randomize(G);
  mac.randomize(G);
}

template<class T, class V>
inline void Share_<T, V>::pack(octetStream& os, bool full) const
{
  a.pack(os, full);
  if (full)
    mac.pack(os);
}

template<class T, class V>
inline void Share_<T, V>::unpack(octetStream& os, bool full)
{
  a.unpack(os, full);
  if (full)
    mac.unpack(os);
}

#endif
