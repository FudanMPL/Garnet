/*
 * Secret_inline.h
 *
 */

#ifndef GC_SECRET_INLINE_H_
#define GC_SECRET_INLINE_H_

#include "BMR/config.h"
#include "GC/Secret.h"

#ifdef MAX_INLINE
#define MAYBE_INLINE inline
#else
#define MAYBE_INLINE
#endif

namespace GC {

template <class T>
inline T XOR(const T& left, const T& right)
{
	T res;
	XOR<T>(res, left, right);
	return res;
}

template <class T>
inline void AND(T& res, const T& left, const T& right)
{
#ifdef DEBUG_REGS
    cout << "*" << res.get_id() << " = AND(*" << left.get_id() << ",*" << right.get_id() << ")" << endl;
#endif
    res.op(left, right, 0x0001);
}

template <class T>
inline T AND(const T& left, const T& right)
{
    T res;
	AND<T>(res, left, right);
	return res;
}

template<class T>
inline Secret<T> GC::Secret<T>::operator+(const Secret<T>& x) const
{
    Secret<T> res;
    int min_n = min(registers.size(), x.registers.size());
    int n = max(registers.size(), x.registers.size());
    res.xor_(min_n, *this, x);
    if (min_n < n)
    {
        const vector<T>* more_regs;
        if (registers.size() < x.registers.size())
            more_regs = &x.registers;
        else
            more_regs = &registers;
        res.registers.insert(res.registers.end(), more_regs->begin() + min_n,
                more_regs->begin() + min((size_t)n, more_regs->size()));
    }
    return res;
}

template <class T>
void Secret<T>::invert(int n, const Secret<T>& x)
{
    resize_regs(n);
    T one;
    one.public_input(1);
    for (int i = 0; i < n; i++)
    {
        XOR(get_reg(i), x.get_reg(i), one);
    }
}

template <class T>
MAYBE_INLINE void Secret<T>::and_(int n, const Secret<T>& x, const Secret<T>& y, bool repeat)
{
    assert(n <= default_length);
    resize_regs(n);
    for (int i = 0; i < n; i++)
        AND<T>(registers[i], x.get_reg(i), y.get_reg(repeat ? 0 : i));
}

template <class T>
inline void Secret<T>::resize_regs(size_t n)
{
    if (registers.size() != n)
        registers.resize(n);
}

} /* namespace GC */

#endif /* GC_SECRET_INLINE_H_ */
