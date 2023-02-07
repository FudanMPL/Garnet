/*
 * memcpy.h
 *
 */

#ifndef TOOLS_AVX_MEMCPY_H_
#define TOOLS_AVX_MEMCPY_H_

#include <string.h>
#include <cstdint>

#include "intrinsics.h"

inline void avx_memcpy(void* dest, const void* source, size_t length)
{
	memcpy(dest, source, length);
}

template<size_t L>
inline void avx_memcpy(void* dest, const void* source)
{
	size_t length = L;
#ifdef __SSE2__
	__m256i* d = (__m256i*)dest, *s = (__m256i*)source;
#ifdef __AVX__
	while (length >= 32)
	{
		_mm256_storeu_si256(d++, _mm256_loadu_si256(s++));
		length -= 32;
	}
#endif
	__m128i* d2 = (__m128i*)d;
	__m128i* s2 = (__m128i*)s;
	while (length >= 16)
	{
		_mm_storeu_si128(d2++, _mm_loadu_si128(s2++));
		length -= 16;
	}
#else
	void* d2 = dest;
	const void* s2 = source;
#endif
	switch (length)
	{
	case 0:
		return;
	case 1:
		*(char*)d2 = *(char*)s2;
		return;
	case 8:
	    *(int64_t*)d2 = *(int64_t*)s2;
	    return;
	default:
		memcpy(d2, s2, length);
		return;
	}
}

inline void avx_memzero(void* dest, size_t length)
{
#if defined(__AVX__) and defined(__clang__)
	__m256i* d = (__m256i*)dest;
	__m256i s = _mm256_setzero_si256();
	while (length >= 32)
	{
		_mm256_storeu_si256(d++, s);
		length -= 32;
	}
#else
	void* d = dest;
#endif
	switch (length)
	{
	case 8:
		*(int64_t*)d = 0;
	    return;
	default:
		memset((void*)d, 0, length);
	}
}

#endif /* TOOLS_AVX_MEMCPY_H_ */
