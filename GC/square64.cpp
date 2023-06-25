/*
 * square64.cpp
 *
 */

#include "square64.h"
#include "Tools/cpu_support.h"
#include "OT/BitMatrix.h"

#include <stdexcept>
#include <iostream>
#include <assert.h>
using namespace std;

union matrix32x8
{
    __m256i whole;
    octet rows[32];

    matrix32x8(const __m256i& x = _mm256_setzero_si256()) : whole(x) {}

    matrix32x8(square64& input, int x, int y)
    {
        for (int l = 0; l < 32; l++)
            rows[l] = input.bytes[32*x+l][y];
    }

    void transpose(square64& output, int x, int y)
    {
#if defined(__AVX2__) || !defined(__x86_64__)
        if (cpu_has_avx2())
        {
            for (int j = 0; j < 8; j++)
            {
                int row = _mm256_movemask_epi8(whole);
                whole = _mm256_slli_epi64(whole, 1);

                // _mm_movemask_epi8 uses most significant bit, hence +7-j
                output.halfrows[8*x+7-j][y] = row;
            }
        }
        else
#endif
        {
            (void) output, (void) x, (void) y;
            throw runtime_error("need AVX2 support");
        }
    }
};

#ifdef DEBUG_TRANS
ostream& operator<<(ostream& os, const __m256i& x)
{
    for (int i = 0; i < 4; i++)
        os << hex << " " << ((long*)&x)[i];
    os << dec;
    return os;
}
#endif


#define ZIP_CASE(I, LOWS, HIGHS, A, B) \
case I: \
    LOWS = _mm256_unpacklo_epi##I(A, B); \
    HIGHS = _mm256_unpackhi_epi##I(A, B); \
    break;

void zip(int chunk_size, __m256i& lows, __m256i& highs,
        const __m256i& a, const __m256i& b)
{
#if defined(__AVX2__) || !defined(__x86_64__)
    if (cpu_has_avx2())
    {
        switch (chunk_size)
        {
        ZIP_CASE(8, lows, highs, a, b);
        ZIP_CASE(16, lows, highs, a, b);
        ZIP_CASE(32, lows, highs, a, b);
        ZIP_CASE(64, lows, highs, a, b);
        case 128:
            lows = a;
            highs = b;
            swap(((__m128i*)&lows)[1], ((__m128i*)&highs)[0]);
            break;
        default:
            throw invalid_argument("not supported");
        }
    }
    else
#endif
    {
        (void) chunk_size, (void) lows, (void) highs, (void) a, (void) b;
        throw runtime_error("need AVX2 support");
    }
}

void square64::transpose(int n_rows, int n_cols)
{
#ifdef DEBUG_TRANS
    cout << "transpose" << endl;
    print();
#endif

    assert(n_rows <= 64);
    assert(n_cols <= 64);

#ifndef __AVX2__
    square128 tmp2;
    tmp2.set_zero();
    for (int i = 0; i < n_rows; i++)
        tmp2.rows[i] = _mm_cvtsi64_si128(rows[i]);
    tmp2.transpose();
    *this = {};
    for (int i = 0; i < n_cols; i++)
        rows[i] = _mm_cvtsi128_si64(tmp2.rows[i]);
    return;
#endif

    square64 tmp = *this;
    *this = {};

    for (int k = 0; k < DIV_CEIL(n_rows, 32); k++)
    {
        __m256i x[8], lows[4], highs[4];
        memcpy(x, &tmp.quadrows[8 * k], sizeof(x));
#ifdef DEBUG_TRANS
        for (int j = 0; j < 8; j++)
            if (not _mm256_testz_si256(x[j], x[j]))
            {
                cout << "transpose k " << k << " j " << j << ": ";
                for (int i = 0; i < 4; i++)
                    cout << hex << " " << ((long*)&x[j])[i];
                cout << dec << endl;
            }
#endif
        for (int chunk_size = 128; chunk_size >= 64; chunk_size /= 2)
        {
            for (int j = 0; j < 4; j ++)
            {
                int a, b;
                if (chunk_size > 64)
                {
                    a = j;
                    b = a + 4;
                }
                else if (chunk_size == 64)
                {
                    a = j / 2 * 2 + j;
                    b = a + 2;
                }
                else
                {
                    a = 2 * j;
                    b = a + 1;
                }
                zip(chunk_size, lows[j], highs[j], x[a], x[b]);
            }
            memcpy(x, lows, sizeof(lows));
            memcpy(&x[4], highs, sizeof(highs));
#ifdef DEBUG_TRANS
            for (int j = 0; j < 8; j++)
                if (not _mm256_testz_si256(x[j], x[j]))
                {
                    cout << "transpose k " << k << " chunk " << chunk_size
                            << " j " << j << ": ";
                    for (int i = 0; i < 4; i++)
                        cout << hex << " " << ((long*)&x[j])[i];
                    cout << dec << endl;
                }
#endif
        }
        for (int chunk_size = 8; chunk_size < 128; chunk_size *= 2)
        {
            for (int j = 0; j < 4; j ++)
            {
                int a = j / 2 * 2 + j;
                int b = a + 2;
                if (chunk_size == 8)
                {
                    a = j;
                    b = j + 4;
                }
                if (chunk_size == 64)
                {
                    a = 2 * j;
                    b = a + 1;
                }
                if (chunk_size == 32)
                {
                    a = 2 * j;
                    b = a + 1;
                }
                zip(chunk_size, lows[j], highs[j], x[a], x[b]);
            }

            memcpy(x, lows, sizeof(lows));
            memcpy(&x[4], highs, sizeof(highs));
#ifdef DEBUG_TRANS
            for (int j = 0; j < 8; j++)
                if (not _mm256_testz_si256(x[j], x[j]))
                {
                    cout << "transpose k " << k << " chunk " << chunk_size
                            << " j " << j << ": ";
                    for (int i = 0; i < 4; i++)
                        cout << hex << " " << ((long*)&x[j])[i];
                    cout << dec << endl;
                }
#endif
        }

        int perm[] = { 0, 4, 2, 6, 1, 5, 3, 7 };
        for (int i = 0; i < DIV_CEIL(n_cols, 8); i++)
        {
            matrix32x8(x[perm[i]]).transpose(*this, i, k);
        }
    }
#ifdef DEBUG_TRANS
    cout << "after transpose" << endl;
    print();
#endif
}

bool square64::operator !=(const square64& other)
{
    for (int i = 0; i < 64; i++)
        if (rows[i] != other.rows[i])
            return false;
    return true;
}


void square64::print()
{
    for (int i = 0; i < 64; i++)
    {
        for (int j = 0; j < 64; j++)
            cout << get_bit(i, j);
        cout << endl;
    }
    cout << flush;
}
