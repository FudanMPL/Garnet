/*
 * brain-party.cpp
 *
 */

#include "Protocols/BrainShare.h"
#include "Protocols/MaliciousRep3Share.h"
#include "Processor/RingOptions.h"

#include "Processor/RingMachine.hpp"
#include "Protocols/BrainPrep.hpp"
#include "Machines/RepRing.hpp"
#include "Machines/MalRep.hpp"
#include "Math/gfp.hpp"

int main(int argc, const char** argv)
{
    ez::ezOptionParser opt;
    HonestMajorityRingMachineWithSecurity<BrainShare, MaliciousRep3Share>(argc,
            argv, opt);
}
