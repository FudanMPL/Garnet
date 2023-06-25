/*
 * bigint.hpp
 *
 */

#ifndef MATH_BIGINT_HPP_
#define MATH_BIGINT_HPP_

#include "bigint.h"
#include "Integer.h"

template<int X, int L>
bigint& bigint::from_signed(const gfp_<X, L>& other)
{
    to_signed_bigint(*this, other);
    return *this;
}

template<class T>
bigint& bigint::from_signed(const T& other)
{
    *this = other;
    return *this;
}

template<class T>
mpf_class bigint::get_float(T v, T p, T z, T s)
{
    // MPIR can't handle more precision in exponent
    Integer exp = Integer(p, 31).get();
    bigint tmp;
    tmp.from_signed(v);
    mpf_class res = tmp;
    if (exp > 0)
        mpf_mul_2exp(res.get_mpf_t(), res.get_mpf_t(), exp.get());
    else
        mpf_div_2exp(res.get_mpf_t(), res.get_mpf_t(), -exp.get());
    if (z.is_one())
        res = 0;
    if (s.is_one())
    {
        res *= -1;
    }
    if (not z.is_bit() or not s.is_bit())
      {
        cerr << "z=" << z << " s=" << s << endl;
        throw Processor_Error("invalid floating point number");
      }
    return res;
}

template<class U, class T>
void bigint::output_float(U& o, const mpf_class& x, T nan)
{
    assert(nan.is_bit());
    if (nan.is_zero())
        o << x;
    else
        o << "NaN";
}

#endif /* MATH_BIGINT_HPP_ */
