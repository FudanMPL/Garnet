/*
 * mac_key.hpp
 *
 */

#ifndef PROTOCOLS_MAC_KEY_HPP_
#define PROTOCOLS_MAC_KEY_HPP_

#include "Networking/Player.h"
#include "Math/Setup.h"

#include "fake-stuff.hpp"

template<class T>
typename T::mac_key_type read_or_generate_mac_key(const Player& P,
        string directory = "")
{
    if (directory == "")
        directory = get_prep_sub_dir<T>(P.num_players());
    typename T::mac_key_type res;
    T::read_or_generate_mac_key(directory, P, res);
    return res;
}

template<class T>
void check_field_size()
{
    if (T::length() < OnlineOptions::singleton.security_parameter)
        throw runtime_error("Field too small for chosen security. "
                "Increase size with -lgp or decrease security with -S");
}

#endif /* PROTOCOLS_MAC_KEY_HPP_ */
