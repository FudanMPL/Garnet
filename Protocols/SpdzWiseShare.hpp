/*
 * SpdzWiseShare.hpp
 *
 */

#ifndef PROTOCOLS_SPDZWISESHARE_HPP_
#define PROTOCOLS_SPDZWISESHARE_HPP_

#include "SpdzWiseShare.h"

#include "fake-stuff.hpp"

template<class T>
void SpdzWiseShare<T>::read_or_generate_mac_key(string directory, Player& P, T& mac_key)
{
    bool fresh = false;

    try
    {
        read_mac_key(directory, P.N, mac_key);
    }
    catch (mac_key_error&)
    {
        fresh = true;
    }

    try
    {
        // validate MAC key
        typename open_part_type::MAC_Check MC;
        auto masked = typename T::Honest::Protocol(P).get_random() + mac_key;
        MC.open(masked, P);
        MC.Check(P);
    }
    catch (mac_fail&)
    {
        fresh = true;
        cerr << "Invalid " << type_string() << " MAC key, generating fresh one" << endl;
    }

    if (fresh)
        mac_key = typename T::Honest::Protocol(P).get_random();
}

template<class T>
void SpdzWiseShare<T>::pack(octetStream& os, bool full) const
{
    super::pack(os, full);
}

template<class T>
void SpdzWiseShare<T>::pack(octetStream& os, open_type factor) const
{
    this->get_share().pack(os, factor);
}

#endif /* PROTOCOLS_SPDZWISESHARE_HPP_ */
