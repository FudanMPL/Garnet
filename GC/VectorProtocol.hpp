/*
 * VectorProtocol.cpp
 *
 */

#ifndef GC_VECTORPROTOCOL_HPP_
#define GC_VECTORPROTOCOL_HPP_

#include "VectorProtocol.h"

namespace GC
{

template<class T>
VectorProtocol<T>::VectorProtocol(Player& P) :
        part_protocol(P), P(P)
{
}

template<class T>
void VectorProtocol<T>::init(Preprocessing<T>& prep,
        typename T::MAC_Check& MC)
{
    part_protocol.init(prep.get_part(), MC.get_part_MC());
}

template<class T>
void VectorProtocol<T>::init_mul()
{
    part_protocol.init_mul();
}

template<class T>
void VectorProtocol<T>::prepare_mul(const T& x,
        const T& y, int n)
{
    if (n == -1)
        n = T::default_length;
    for (int i = 0; i < n; i++)
        part_protocol.prepare_mul(x.get_reg(i), y.get_reg(i), 1);
}

template<class T>
void VectorProtocol<T>::exchange()
{
    part_protocol.exchange();
}

template<class T>
T VectorProtocol<T>::finalize_mul(int n)
{
    T res;
    finalize_mult(res, n);
    return res;
}

template<class T>
void VectorProtocol<T>::finalize_mult(T& res, int n)
{
    if (n == -1)
        n = T::default_length;
    res.resize_regs(n);
    for (int i = 0; i < n; i++)
        res.get_reg(i) = part_protocol.finalize_mul(1);
}

} /* namespace GC */

#endif
