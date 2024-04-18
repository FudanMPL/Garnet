/*
 * square64.h
 *
 */

#ifndef GC_SQUARE64_H_
#define GC_SQUARE64_H_

#include <string.h>
#include <cstdint>
#include "Tools/int.h"
#include "Tools/intrinsics.h"

union square64
{
    __m256i quadrows[16];
    __m128i doublerows[32];
    int64_t rows[64];
    int32_t halfrows[128][2];
    octet bytes[64][8];

    square64()
    {
        memset(bytes, 0, sizeof(bytes));
    }

    bool get_bit(int x, int y)
    { return (bytes[x][y/8] >> (y % 8)) & 1; }

    void transpose(int n_rows, int n_cols);

    bool operator!=(const square64& other);

    void print();
};

#endif /* GC_SQUARE64_H_ */
