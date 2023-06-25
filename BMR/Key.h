/*
 * Key.h
 *
 */

#ifndef COMMON_INC_KEY_H_
#define COMMON_INC_KEY_H_

#include <iostream>
#include <string.h>

#include "Tools/FlexBuffer.h"
#include "Tools/intrinsics.h"
#include "Math/gf2nlong.h"

using namespace std;

class Key {
public:
	__m128i r;

	Key() {}
	Key(long long a) : r(_mm_cvtsi64_si128(a)) {}
	Key(long long a, long long b) : r(_mm_set_epi64x(a, b)) {}
	Key(__m128i r) : r(r) {}

	bool operator==(const Key& other);
	bool operator!=(const Key& other) { return !(*this == other); }

	Key& operator-=(const Key& other);
	Key& operator+=(const Key& other);

	Key operator^(const Key& other) const { return r ^ other.r; }
	Key operator^=(const Key& other) { r ^= other.r; return *this; }

	void serialize(SendBuffer& output) const { output.serialize(r); }
	void serialize_no_allocate(SendBuffer& output) const { output.serialize_no_allocate(r); }

	bool get_signal() const { return _mm_cvtsi128_si32(r) & 1; }
	void set_signal(bool signal);

	Key doubling(int i) const;

	template <class T>
	T get() const;
};

ostream& operator<<(ostream& o, const Key& key);
ostream& operator<<(ostream& o, const __m128i& x);


inline bool Key::operator==(const Key& other) {
    return int128(r) == other.r;
}

inline Key& Key::operator-=(const Key& other) {
    r ^= other.r;
    return *this;
}

inline Key& Key::operator+=(const Key& other) {
    r ^= other.r;
    return *this;
}

template <>
inline unsigned long Key::get() const
{
	return _mm_cvtsi128_si64(r);
}

template <>
inline __m128i Key::get() const
{
	return r;
}

inline void Key::set_signal(bool signal)
{
	r &= ~_mm_cvtsi64_si128(1);
	r ^= _mm_cvtsi64_si128(signal);
}

inline Key Key::doubling(int i) const
{
#ifdef __AVX2__
    if (cpu_has_avx2())
        return _mm_sllv_epi64(r, _mm_set_epi64x(i, i));
    else
#endif
    {
        uint64_t halfs[2];
        halfs[1] = _mm_cvtsi128_si64(_mm_unpackhi_epi64(r, r)) << i;
        halfs[0] = _mm_cvtsi128_si64(r) << i;
        return _mm_loadu_si128((__m128i*)halfs);
    }
}

#endif /* COMMON_INC_KEY_H_ */
