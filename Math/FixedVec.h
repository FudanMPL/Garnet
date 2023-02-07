/*
 * FixedVec.h
 *
 */

#ifndef MATH_FIXEDVEC_H_
#define MATH_FIXEDVEC_H_

#include <string>
#include <array>
using namespace std;

#include "Tools/octetStream.h"
#include "Tools/random.h"
#include "field_types.h"

template <class T, int L>
class FixedVec
{
    typedef FixedVec This;
    array<T, L> v;

public:
    typedef T value_type;
    typedef FixedVec Scalar;

    static const int vector_length = L;

    static int length()
    {
        return L * T::length();
    }

    static int size()
    {
        return L * T::size();
    }
    static int size_in_bits()
    {
        return L * T::size_in_bits();
    }

    static string type_string()
    {
        return T::type_string() + "^" + to_string(L);
    }
    static string type_short()
    {
        return string(1, T::type_char());
    }

    static void specification(octetStream& os)
    {
        T::specification(os);
    }

    template<class U, class V>
    static FixedVec Mul(const FixedVec<U, L>& a, const V& b)
    {
        FixedVec res;
        for (int i = 0; i < L; i++)
            res[i] = T::Mul(a[i], b);
        return res;
    }

    FixedVec<T, L>(const T& other = {})
    {
        for (auto& x : v)
            x = other;
    }

    FixedVec<T, L>(long other) :
            FixedVec<T, L>(T(other))
    {
    }

    template<class U>
    FixedVec<T, L>(const FixedVec<U, L>& other)
    {
        for (int i = 0; i < L; i++)
            v[i] = other[i];
    }

    FixedVec<T, L>(const array<T, L>& other)
    {
        v = other;
    }

    const array<T, L>& get() const
    {
        return v;
    }

    T& operator[](int i)
    {
        return v[i];
    }
    const T& operator[](int i) const
    {
        return v[i];
    }

    void assign(const T& other)
    {
        for (auto& x : v)
            x = other;
    }
    void assign(const char* buffer)
    {
        for (int i = 0; i < L; i++)
            v[i].assign(buffer + i * T::size());
    }

    void assign_zero()
    {
        for (auto& x : v)
            x = 0;
    }
    void assign_one()
    {
        assign(1);
    }

    void add(const FixedVec<T, L>& x, const FixedVec<T, L>& y)
    {
        for (int i = 0; i < L; i++)
            v[i] = x.v[i] + y.v[i];
    }
    void sub(const FixedVec<T, L>& x, const FixedVec<T, L>& y)
    {
        for (int i = 0; i < L; i++)
            v[i] = x.v[i] - y.v[i];
    }

    void mul(const FixedVec<T, L>& x, const FixedVec<T, L>& y)
    {
        for (int i = 0; i < L; i++)
            v[i] = (x.v[i] * y.v[i]);
    }

    void add(octetStream& os)
    {
        for (int i = 0; i < L; i++)
            v[i].add(os);
    }

    void negate()
    {
        for (auto& x : v)
            x = -x;
    }

    bool equal(const FixedVec<T, L>& x) const
    {
        for (int i = 0; i < L; i++)
            if (v[i] != x[i])
                return false;
        return true;
    }

    bool is_zero()
    {
        return equal(0);
    }
    bool is_one()
    {
        return equal(1);
    }

    bool operator!=(const FixedVec<T, L>& other) const
    {
        return not equal(other);
    }

    FixedVec<T, L>operator+(const FixedVec<T, L>& other) const
    {
        FixedVec<T, L> res;
        res.add(*this, other);
        return res;
    }

    FixedVec<T, L>operator-(const FixedVec<T, L>& other) const
    {
        FixedVec<T, L> res;
        res.sub(*this, other);
        return res;
    }

    FixedVec<T, L>operator*(const FixedVec<T, L>& other) const
    {
        FixedVec<T, L> res;
        res.mul(*this, other);
        return res;
    }

    FixedVec<T, L>operator/(const FixedVec<T, L>& other) const
    {
        FixedVec<T, L> res;
        for (int i = 0; i < L; i++)
            res[i] = v[i] / other[i];
        return res;
    }

    FixedVec<T, L>operator^(const FixedVec<T, L>& other) const
    {
        FixedVec<T, L> res;
        for (int i = 0; i < L; i++)
            res[i] = v[i] ^ other[i];
        return res;
    }

    FixedVec<T, L>operator&(const FixedVec<T, L>& other) const
    {
        FixedVec<T, L> res;
        for (int i = 0; i < L; i++)
            res[i] = v[i] & other[i];
        return res;
    }

    FixedVec<T, L>operator~() const
    {
        FixedVec<T, L> res;
        for (int i = 0; i < L; i++)
            res[i] = ~v[i];
        return res;
    }

    FixedVec<T, L>& operator+=(const FixedVec<T, L>& other)
    {
        add(*this, other);
        return *this;
    }

    FixedVec<T, L>& operator-=(const FixedVec<T, L>& other)
    {
        sub(*this, other);
        return *this;
    }

    FixedVec<T, L>& operator*=(const FixedVec<T, L>& other)
    {
        *this = *this * other;
        return *this;
    }

    FixedVec<T, L>& operator/=(const FixedVec<T, L>& other)
    {
        *this = *this / other;
        return *this;
    }

    FixedVec<T, L>& operator^=(const FixedVec<T, L>& other)
    {
        for (int i = 0; i < L; i++)
            v[i] ^= other[i];
        return *this;
    }

    FixedVec<T, L>& operator&=(const FixedVec<T, L>& other)
    {
        for (int i = 0; i < L; i++)
            v[i] &= other[i];
        return *this;
    }

    FixedVec<T, L> operator<<(int i) const
    {
        FixedVec<T ,L> res;
        for (int j = 0; j < L; j++)
            res[j] = v[j] << i;
        return res;
    }

    FixedVec<T, L> operator>>(int i) const
    {
        FixedVec<T ,L> res;
        for (int j = 0; j < L; j++)
            res[j] = v[j] >> i;
        return res;
    }

    FixedVec<T, L>& operator<<=(int i)
    {
        *this = *this << i;
        return *this;
    }

    FixedVec<T, L>& operator>>=(int i)
    {
        *this = *this >> i;
        return *this;
    }

    T sum() const
    {
        T res = v[0];
        for (int i = 1; i < L; i++)
            res += v[i];
        return res;
    }

    T lazy_sum() const
    {
        assert(L > 1);
        T res = v[0].lazy_add(v[1]);
        for (int i = 2; i < L; i++)
            res = res.lazy_add(v[i]);
        return res;
    }

    void extend_bit(This& res, int n_bits) const
    {
        for (int i = 0; i < L; i++)
            v[i].extend_bit(res[i], n_bits);
    }

    void mask(This& res, int n_bits) const
    {
        for (int i = 0; i < L; i++)
            v[i].mask(res[i], n_bits);
    }

    This extend_bit() const
    {
        This res;
        extend_bit(res, T::N_BITS);
        return res;
    }

    This mask(int n_bits) const
    {
        This res;
        mask(res, n_bits);
        return res;
    }

    void randomize(PRNG& G)
    {
        for (auto& x : v)
            x.randomize(G);
    }

    void almost_randomize(PRNG& G)
    {
        for (auto& x : v)
            x.almost_randomize(G);
    }

    void randomize_to_sum(const T& sum, PRNG& G)
    {
        T s = 0;
        for (int i = 1; i < L; i++)
        {
            v[i].randomize(G);
            s += v[i];
        }
        v[0] = sum - s;
    }

    void force_to_bit()
    {
        for (auto& x : v)
            x.force_to_bit();
    }

    FixedVec get_bit(int i)
    {
        return (*this >> i) & 1;
    }

    void output(ostream& s, bool human) const
    {
        if (human)
            s << *this;
        else
            for (auto& x : v)
                x.output(s, human);
    }
    void input(istream& s, bool human)
    {
        for (int i = 0; i < L; i++)
        {
            if (human and i != 0)
                if (s.get() != ',')
                    throw runtime_error("cannot read vector");
            (*this)[i].input(s, human);
        }
    }

    void pack(octetStream& os) const
    {
        for (auto& x : v)
            x.pack(os);
    }
    void unpack(octetStream& os)
    {
        for (auto& x : v)
            x.unpack(os);
    }
};

template <class U, class T, int L>
FixedVec<T, L> operator*(const U& a, const FixedVec<T, L>& b)
{
    return b * T(a);
}

template <class T, int L>
ostream& operator<<(ostream& os, const FixedVec<T, L>& v)
{
    for (int i = 0; i < L; i++)
    {
        os << v[i];
        if (i < L - 1)
            os << ",";
    }
    return os;
}

#endif /* MATH_FIXEDVEC_H_ */
