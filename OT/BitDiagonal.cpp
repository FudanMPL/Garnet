/*
 * Diagonal.cpp
 *
 */

#include <OT/BitDiagonal.h>

void BitDiagonal::pack(octetStream& os) const
{
    for (int i = 0; i < N_ROWS; i++)
        os.store_int(rows[i].get_bit(i), 1);
}

void BitDiagonal::unpack(octetStream& os)
{
    *this = {};
    for (int i = 0; i < N_ROWS; i++)
        rows[i] = os.get_int(1) << i;
}
