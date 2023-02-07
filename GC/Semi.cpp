/*
 * Semi.cpp
 *
 */

#include "Semi.h"
#include "SemiPrep.h"

#include "Protocols/MAC_Check_Base.hpp"
#include "Protocols/Replicated.hpp"
#include "Protocols/SemiInput.hpp"
#include "Protocols/Beaver.hpp"

namespace GC
{

void Semi::prepare_mult(const SemiSecret& x, const SemiSecret& y, int n,
		bool repeat)
{
    if (repeat and OnlineOptions::singleton.live_prep)
    {
        this->triples.push_back({{}});
        auto& triple = this->triples.back();
        triple = dynamic_cast<SemiPrep*>(prep)->get_mixed_triple(n);
        for (int i = 0; i < 2; i++)
            triple[1 + i] = triple[1 + i].mask(n);
        triple[0] = triple[0].extend_bit().mask(n);
        shares.push_back(y - triple[0]);
        shares.push_back(x - triple[1]);
        lengths.push_back(n);
    }
    else
        prepare_mul(x, y, n);
}

} /* namespace GC */
