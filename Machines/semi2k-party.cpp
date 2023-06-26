/*
 * semi2k-party.cpp
 *
 */

#include "Protocols/Semi2kShare.h"
#include "Protocols/SemiPrep2k.h"
#include "Math/gf2n.h"
#include "Processor/RingOptions.h"
#include "GC/SemiPrep.h"

#include "Semi.hpp"
#include "GC/ShareSecret.hpp"
#include "Protocols/RepRingOnlyEdabitPrep.hpp"
#include "Processor/RingMachine.hpp"

int main(int argc, const char** argv)
{
    ez::ezOptionParser opt;
    DishonestMajorityRingMachine<Semi2kShare, SemiShare>(argc, argv, opt);
}
