#ifndef __AES_H
#define __AES_H

#include "Networking/data.h"
#include "cpu_support.h"
#include "intrinsics.h"

typedef unsigned int  uint;

#define AES_BLK_SIZE 16

/************* C Version *************/
// Key Schedule
void aes_schedule( int nb, int nr, const octet* k, uint* RK );

inline void aes_schedule( uint* RK, octet* K )
{ aes_schedule(4,10,K,RK); }
inline void aes_128_schedule( uint* RK, const octet* K )
{ aes_schedule(4,10,K,RK); }
inline void aes_192_schedule( uint* RK, octet* K )
{ aes_schedule(6,12,K,RK); }
inline void aes_256_schedule( uint* RK, octet* K )
{ aes_schedule(8,14,K,RK); }

// Encryption Function 
void aes_128_encrypt( octet* C, octet* M, uint* RK );
void aes_192_encrypt( octet* C, octet* M, uint* RK );
void aes_256_encrypt( octet* C, octet* M, uint* RK );

inline void aes_encrypt( octet* C, octet* M, uint* RK )
{ aes_128_encrypt(C,M,RK ); }


/*********** M-Code Version ***********/
// Check can support this
inline int Check_CPU_support_AES() { return cpu_has_aes(); }
// Key Schedule 
void aes_128_schedule( octet* key, const octet* userkey );
void aes_192_schedule( octet* key, const octet* userkey );
void aes_256_schedule( octet* key, const octet* userkey );

inline void aes_schedule( octet* key, const octet* userkey )
{ aes_128_schedule(key,userkey); }


// Encryption Function 
void aes_128_encrypt( octet* C, const octet* M,const octet* RK );
void aes_192_encrypt( octet* C, const octet* M,const octet* RK );
void aes_256_encrypt( octet* C, const octet* M,const octet* RK );

#ifndef __clang__
__attribute__((optimize("unroll-loops")))
#endif
inline __m128i aes_128_encrypt(__m128i in, const octet* key)
{
#if defined(__AES__) || !defined(__x86_64__)
    if (cpu_has_aes())
    {
        __m128i& tmp = in;
        tmp = _mm_xor_si128 (tmp,((__m128i*)key)[0]);
        int j;
        for(j=1; j <10; j++)
            tmp = _mm_aesenc_si128 (tmp,((__m128i*)key)[j]);
        tmp = _mm_aesenclast_si128 (tmp,((__m128i*)key)[j]);
        return tmp;
    }
    else
#endif
    {
        __m128i tmp;
        aes_128_encrypt((octet*) &tmp, (octet*) &in, (uint*) key);
        return tmp;
    }
}

template <int N>
inline void software_ecb_aes_128_encrypt(__m128i* out, const __m128i* in, uint* key)
{
    for (int i = 0; i < N; i++)
        aes_128_encrypt((octet*)&out[i], (octet*)&in[i], key);
}

template <int N>
#ifndef __clang__
__attribute__((optimize("unroll-loops")))
#endif
inline void ecb_aes_128_encrypt(__m128i* out, const __m128i* in, const octet* key)
{
#if defined(__AES__) || !defined(__x86_64__)
    if (cpu_has_aes())
    {
        __m128i tmp[N];
        for (int i = 0; i < N; i++)
            tmp[i] = _mm_xor_si128 (in[i],((__m128i*)key)[0]);
        int j;
        for(j=1; j <10; j++)
            for (int i = 0; i < N; i++)
                tmp[i] = _mm_aesenc_si128 (tmp[i],((__m128i*)key)[j]);
        for (int i = 0; i < N; i++)
            out[i] = _mm_aesenclast_si128 (tmp[i],((__m128i*)key)[j]);
    }
    else
#endif
        software_ecb_aes_128_encrypt<N>(out, in, (uint*) key);
}

template <int N>
inline void ecb_aes_128_encrypt(__m128i* out, const __m128i* in, const octet* key, const int* indices)
{
    __m128i tmp[N];
    for (int i = 0; i < N; i++)
        tmp[i] = in[indices[i]];
    ecb_aes_128_encrypt<N>(tmp, tmp, key);
    for (int i = 0; i < N; i++)
        out[indices[i]] = tmp[i];
}

inline void aes_encrypt( octet* C, const octet* M,const octet* RK )
{ aes_128_encrypt(C,M,RK); }

inline __m128i aes_encrypt( __m128i M,const octet* RK )
{ return aes_128_encrypt(M,RK); }


#endif

