/*
 * Semi.h
 *
 */

#ifndef GC_SEMI_H_
#define GC_SEMI_H_

#include "Protocols/Beaver.h"
#include "SemiSecret.h"

namespace GC
{

class Semi : public Beaver<SemiSecret>
{
    typedef Beaver<SemiSecret> super;

public:
    Semi(Player& P) :
            super(P)
    {
    }

    void prepare_mult(const SemiSecret& x, const SemiSecret& y, int n,
            bool repeat);
};

} /* namespace GC */

#endif /* GC_SEMI_H_ */
