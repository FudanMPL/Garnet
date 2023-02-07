/*
 * Square.cpp
 *
 */

#include "Square.h"
#include "BitVec.h"
#include "gf2n.h"
#include "gfp.h"

template<>
void Square<gf2n_short>::to(gf2n_short& result, false_type)
{
    int128 sum;
    for (int i = 0; i < gf2n_short::degree(); i++)
        sum ^= int128(rows[i].get()) << i;
    result = sum;
}

template<>
void Square<BitVec>::to(BitVec& result, false_type)
{
    result = 0;
    for (int i = 0; i < N_ROWS; i++)
        result ^= ((rows[i] >> i) & 1) << i;
}
