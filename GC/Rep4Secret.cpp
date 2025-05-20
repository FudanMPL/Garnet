/*
 * Rep4Secret.cpp
 *
 */

#ifndef GC_REP4SECRET_CPP_
#define GC_REP4SECRET_CPP_

#include "Rep4Secret.h"

#include "ShareSecret.hpp"
#include "ShareThread.hpp"
#include "Protocols/Rep4MC.hpp"

namespace GC
{

void Rep4Secret::load_clear(int n, const Integer& x)
{
    this->check_length(n, x);
    *this = constant(x, ShareThread<This>::s().P->my_num());
}

}

#endif /* GC_REP4SECRET_CPP_ */
