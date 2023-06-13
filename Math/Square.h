/*
 * gf2nSquare.h
 *
 */

#ifndef MATH_SQUARE_H_
#define MATH_SQUARE_H_

#include "Tools/BitVector.h"

template<class U>
class Square
{
protected:
    static const int N_ROWS = U::MAX_N_BITS;

public:
    typedef U RowType;

    static int n_rows() { return U::size_in_bits(); }
    static int n_rows_allocated() { return n_rows(); }
    static int n_columns() { return n_rows(); }
    static int n_row_bytes() { return U::size(); }

    static size_t size() { return U::length() * U::size(); }

    U rows[N_ROWS];

    Square& sub(const Square& other);
    Square& rsub(const Square& other);
    Square& sub(const void* other);

    void bit_sub(const BitVector& bits, int start);

    void randomize(int row, PRNG& G) { rows[row].randomize(G); }
    void conditional_add(BitVector& conditions, Square& other,
            int offset);
    void to(U& result);
    void to(U& result, false_type);
    void to(U& result, true_type);
    template<int X, int L>
    void to(gfp_<X, L>& result, true_type);

    void pack(octetStream& os) const;
    void unpack(octetStream& os);
};

#endif /* MATH_SQUARE_H_ */
