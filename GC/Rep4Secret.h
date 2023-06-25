/*
 * Rep4Secret.h
 *
 */

#ifndef GC_REP4SECRET_H_
#define GC_REP4SECRET_H_

#include "ShareSecret.h"
#include "Protocols/Rep4MC.h"
#include "Protocols/Rep4Share.h"

namespace GC
{

class Rep4Secret : public RepSecretBase<Rep4Secret, 3>
{
    typedef RepSecretBase<Rep4Secret, 3> super;
    typedef Rep4Secret This;

public:
    typedef DummyLivePrep<This> LivePrep;
    typedef Rep4<This> Protocol;
    typedef Rep4MC<This> MC;
    typedef MC MAC_Check;
    typedef Rep4Input<This> Input;

    static const bool expensive_triples = false;

    static MC* new_mc(typename super::mac_key_type) { return new MC; }

    static This constant(const typename super::clear& constant, int my_num,
            typename super::mac_key_type = {}, int = -1)
    {
        return Rep4Share<typename super::clear>::constant(constant, my_num);
    }

    Rep4Secret()
    {
    }
    template <class T>
    Rep4Secret(const T& other) :
            super(other)
    {
    }

    void load_clear(int n, const Integer& x);
};

}

#endif /* GC_REP4SECRET_H_ */
