/*
 * MMO.h
 *
 */

#ifndef TOOLS_MMO_H_
#define TOOLS_MMO_H_

#include "Tools/aes.h"
#include "BMR/Key.h"

// Matyas-Meyer-Oseas hashing
class MMO
{
#ifdef GFP_MOD_SZ
    static const int N_KEYS = GFP_MOD_SZ > 8 ? GFP_MOD_SZ : 8;
#else
    static const int N_KEYS = 8;
#endif

    octet IV[N_KEYS][176]  __attribute__((aligned (16)));

    template<int N>
    static void encrypt_and_xor(__m128i* output, const __m128i* input,
            const octet* key);
    template<int N>
    static void encrypt_and_xor(void* output, const void* input,
            const octet* key);
    template<int N>
    static void encrypt_and_xor(void* output, const void* input,
            const octet* key, const int* indices);

public:
    MMO() { zeroIV(); }
    void zeroIV();
    void setIV(int i, octet key[AES_BLK_SIZE]);
    template <class T>
    void hashOneBlock(void* output, const void* input) { hashBlocks<T, 1>((T*)output, input); }
    template <int N, int N_BYTES>
    void hashBlocks(void* output, const void* input, size_t alloc_size);
    template <class T, int N>
    void hashBlocks(void* output, const void* input);
    template <class T>
    void hashEightBlocks(T* output, const void* input);
    template <class T, int N_BYTES>
    void hashEightBlocks(T* output, const void* input);
    template <int X, int L>
    void hashEightBlocks(gfp_<X, L>* output, const void* input);
    template <int X, int L>
    void hashEightBlocks(gfpvar_<X, L>* output, const void* input);
    template <class T>
    void outputOneBlock(octet* output);
    Key hash(const Key& input);
    template <int N>
    void hash(Key* output, const Key* input);
};

template<int N>
inline void MMO::encrypt_and_xor(__m128i* out, const __m128i* in, const octet* key)
{
    ecb_aes_128_encrypt<N>(out, in, key);
    for (int i = 0; i < N; i++)
        out[i] = _mm_xor_si128(out[i], in[i]);
}

template<int N>
inline void MMO::encrypt_and_xor(void* output, const void* input, const octet* key)
{
    __m128i in[N], out[N];
    avx_memcpy(in, input, sizeof(in));
    encrypt_and_xor<N>(out, in, key);
    avx_memcpy(output, out, sizeof(out));
}

inline Key MMO::hash(const Key& input)
{
    Key res;
    encrypt_and_xor<1>(&res.r, &input.r, IV[0]);
    return res;
}

template <int N>
inline void MMO::hash(Key* output, const Key* input)
{
    encrypt_and_xor<N>(&output->r, &input->r, IV[0]);
}

#endif /* TOOLS_MMO_H_ */
