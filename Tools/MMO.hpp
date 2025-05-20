/*
 * MMO.cpp
 *
 *
 */

#include "MMO.h"
#include "Math/gfp.hpp"
#include <unistd.h>


inline
void MMO::zeroIV()
{
    if (N_KEYS > (1 << 8))
        throw not_implemented();
    for (int i = 0; i < N_KEYS; i++)
    {
        octet key[AES_BLK_SIZE];
        memset(key, 0, AES_BLK_SIZE * sizeof(octet));
        key[0] = i;
        setIV(i, key);
    }
}

inline
void MMO::setIV(int i, octet key[AES_BLK_SIZE])
{
    aes_schedule(IV[i],key);
}


template<int N>
void MMO::encrypt_and_xor(void* output, const void* input, const octet* key,
        const int* indices)
{
    __m128i in[N], out[N];
    for (int i = 0; i < N; i++)
        in[i] = _mm_loadu_si128(((__m128i*)input) + indices[i]);
    encrypt_and_xor<N>(out, in, key);
    for (int i = 0; i < N; i++)
        _mm_storeu_si128(((__m128i*)output) + indices[i], out[i]);
}

template <int N, int N_BYTES>
void MMO::hashBlocks(void* output, const void* input, size_t alloc_size)
{
    size_t used_size = N_BYTES;
    int n_blocks = DIV_CEIL(used_size, 16);
    if (n_blocks > N_KEYS)
        throw runtime_error("not enough MMO keys");
    __m128i tmp[N];
    size_t block_size = sizeof(tmp[0]);
    for (int i = 0; i < n_blocks; i++)
    {
        encrypt_and_xor<N>(tmp, input, IV[i]);
        for (int j = 0; j < N; j++)
            memcpy((char*)output + j * alloc_size + i * block_size, &tmp[j],
                    min(used_size - i * block_size, block_size));
    }
}

template <class T, int N>
void MMO::hashBlocks(void* output, const void* input)
{
    hashBlocks<N, T::N_BYTES>(output, input, sizeof(T));
    for (int j = 0; j < N; j++)
        ((T*)output + j)->normalize();
}

template <int X, int L>
void MMO::hashEightBlocks(gfp_<X, L>* output, const void* input)
{
    hashEightBlocks<gfp_<X, L>, gfp_<X, L>::N_BYTES>(output, input);
}

template <class T, int N_BYTES>
void MMO::hashEightBlocks(T* output, const void* input)
{
    T* out = output;
    const int block_size = sizeof(__m128i);
    const int n_blocks = (N_BYTES + block_size - 1) / block_size;
    __m128i tmp[8][n_blocks];
    hashBlocks<8, n_blocks * block_size>(tmp, input, n_blocks * block_size);
    int left = 8;
    int indices[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    while (left)
    {
        int now_left = 0;
        for (int j = 0; j < left; j++)
        {
            memcpy(out[indices[j]].get_ptr(), &tmp[indices[j]], N_BYTES);
            out[indices[j]].zero_overhang();
            if (mpn_cmp((mp_limb_t*) out[indices[j]].get_ptr(),
                    T::get_ZpD().get_prA(), T::get_ZpD().get_t()) >= 0)
            {
                indices[now_left] = indices[j];
                now_left++;
            }
        }
        left = now_left;

        for (int j = 0; j < left; j++)
        {
            __m128i in = tmp[indices[j]][0];
            for (int i = 0; i < n_blocks; i++)
            {
                tmp[indices[j]][i] = aes_128_encrypt(in, IV[i]);
            }
        }
    }
}

template <class T>
void MMO::hashEightBlocks(T* output, const void* input)
{
    hashBlocks<T, 8>(output, input);
}

template <>
inline
void MMO::hashEightBlocks(__m128i* output, const void* input)
{
    hashBlocks<8, 16>(output, input, 16);
}

template<int X, int L>
void MMO::hashEightBlocks(gfpvar_<X, L>* output, const void* input)
{
#define X(N_LIMBS) \
    case N_LIMBS: \
        hashEightBlocks<gfpvar_<X, L>, N_LIMBS * 8>(output, input); \
        break;
    switch(gfpvar_<X, L>::get_ZpD().get_t())
    {
    X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) X(10) X(11) X(12)
    default:
        throw runtime_error("MMO not implemented");
    }
#undef X
}
