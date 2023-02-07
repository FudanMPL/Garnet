/*
 * Rectangle.h
 *
 */

#ifndef OT_RECTANGLE_H_
#define OT_RECTANGLE_H_

#include "Tools/random.h"
#include "Tools/BitVector.h"
#include "Math/Z2k.h"

#define TAU(K, S) 2 * K + 4 * S

template <class U, class V>
class Rectangle
{
	static const int N_ROWS = U::N_BITS;

	// make sure number of allocated rows matches the number of bytes
	static const int N_ROWS_ALLOCATED = 8 * U::N_BYTES;

public:
	typedef V RowType;

	static int n_rows() { return U::N_BITS; }
	static int n_columns() { return V::N_BITS; }
	static int n_row_bytes() { return V::N_BYTES; }
	static int n_rows_allocated() { return N_ROWS_ALLOCATED; }

	V rows[N_ROWS_ALLOCATED];

	static size_t size() { return N_ROWS * RowType::size(); }

	bool operator==(const Rectangle<U,V>& other) const;
	bool operator!=(const Rectangle<U,V>& other) const
			{ return not (*this == other); }

	bool get_bit(int i, int j) { return rows[i].get_bit(j); }

	Rectangle<U, V>& operator+=(const Rectangle<U, V>& other);
	Rectangle<U, V> operator-(const Rectangle<U, V> & other);

	Rectangle<U, V>& sub(Rectangle<U, V>& other) { return other.rsub_(*this); }
	Rectangle<U, V>& rsub(Rectangle<U, V>& other) { return rsub_(other); }
	Rectangle<U, V>& rsub_(Rectangle<U, V>& other);
	Rectangle<U, V>& sub(const void* other) { return sub_(other); }
	Rectangle<U, V>& sub_(const void* other);

	void bit_sub(const BitVector& bits, int start);

	void mul(const BitVector& a, const V& b);

	void randomize(PRNG& G);
	void randomize(int row, PRNG& G) { rows[row].randomize(G); }
	void conditional_add(BitVector& conditions, Rectangle<U, V>& other,
			int offset) { conditional_add_(conditions, other, offset); }
	void conditional_add_(BitVector& conditions, Rectangle<U, V>& other,
			int offset);
	template <class T>
	void to(T& result);
	void to(Rectangle<U, V>& result) { result = *this; }

	void pack(octetStream& o) const;
	void unpack(octetStream& o);

	void print(int i, int j);
};

template <int K, int L>
using Z2kRectangle = Rectangle<Z2<K>, Z2<L> >;

template <int K>
using Z2kSquare = Rectangle<Z2<K>, Z2<K>>;

template <class U, class V>
ostream& operator<<(ostream& o, const Rectangle<U, V>&)
{
	throw not_implemented();
	return o;
}

#endif /* OT_RECTANGLE_H_ */
