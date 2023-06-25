/*
 * Bit.cpp
 *
 */

#include "Bit.h"
#include "gf2n.h"

template<class T>
Bit::Bit(const gf2n_<T>& other) :
        super(other.get_bit(0))
{
    assert(other.is_one() or other.is_zero());
}

template Bit::Bit(const gf2n_<octet>& other);
template Bit::Bit(const gf2n_<word>& other);
