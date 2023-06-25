/*
 * FixInput.cpp
 *
 */

#include "FixInput.h"

#include <math.h>

template<>
void FixInput_<Integer>::read(std::istream& in, const int* params)
{
    double x;
    in >> x;
    items[0] = x * exp2(*params);
}

template<>
void FixInput_<bigint>::read(std::istream& in, const int* params)
{
#ifdef HIGH_PREC_INPUT
    mpf_class x;
    in >> x;
    items[0] = x << *params;
#else
    double x;
    in >> x;
    items[0] = x * exp2(*params);
#endif
}
