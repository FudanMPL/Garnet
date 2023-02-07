/*
 * MamaRectangle.h
 *
 */

#ifndef OT_MAMARECTANGLE_H_
#define OT_MAMARECTANGLE_H_

#include "Math/FixedVec.h"
#include "Math/gfp.h"
#include "Tools/BitVector.h"

template<class T, int N>
class MamaRectangle
{
    typedef MamaRectangle This;

    typename T::Square squares[N];

public:
    typedef GC::NoValue RowType;

    static int n_rows() { return T::Square::n_rows(); }
    static int n_columns() { return T::Square::n_columns(); }
    static int n_row_bytes() { return T::Square::n_row_bytes(); }

    static int size()
    {
        return N * T::Square::size();
    }

    void conditional_add(BitVector& conditions, This& other,
            int offset)
    {
        for (int i = 0; i < N; i++)
            squares[i].conditional_add(conditions, other.squares[i],
                    offset * N + i);
    }

    This& sub(const This& other)
    {
        for (int i = 0; i < N; i++)
            squares[i].sub(other.squares[i]);
        return *this;
    }

    This& rsub(const This& other)
    {
        for (int i = 0; i < N; i++)
            squares[i].rsub(other.squares[i]);
        return *this;
    }

    This& sub(const void* other)
    {
        for (int i = 0; i < N; i++)
            squares[i].sub(other);
        return *this;
    }

    void bit_sub(const BitVector&, int)
    {
        throw not_implemented();
    }

    void randomize(int row, PRNG& G)
    {
        squares[row / T::Square::n_rows()].randomize(
                row % T::Square::n_rows(), G);
    }

    void pack(octetStream& os) const
    {
        for (int i = 0; i < N; i++)
            squares[i].pack(os);
    }

    void unpack(octetStream& os)
    {
        for (int i = 0; i < N; i++)
            squares[i].unpack(os);
    }

    template<class U>
    void to(FixedVec<U, N>& result)
    {
        for (int i = 0; i < N; i++)
            squares[i].to(result[i]);
    }
};

#endif /* OT_MAMARECTANGLE_H_ */
