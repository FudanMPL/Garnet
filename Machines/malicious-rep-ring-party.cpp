/*
 * mal-rep-ring-party.cpp
 *
 */

#include "Protocols/MalRepRingShare.h"
#include "Protocols/MalRepRingOptions.h"
#include "Processor/RingMachine.hpp"
#include "Processor/RingOptions.h"
#include "Machines/RepRing.hpp"
#include "Machines/MalRep.hpp"

int main(int argc, const char** argv)
{
    ez::ezOptionParser opt;
    MalRepRingOptions::singleton = MalRepRingOptions(opt, argc, argv);
    HonestMajorityRingMachineWithSecurity<MalRepRingShare, MaliciousRep3Share>(argc, argv, opt);
}
