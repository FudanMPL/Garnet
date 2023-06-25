/*
 * mal-rep-ring-party.cpp
 *
 */

#include "Protocols/PostSacriRepRingShare.h"
#include "Protocols/PostSacriRepFieldShare.h"
#include "Processor/RingMachine.hpp"
#include "Processor/RingOptions.h"
#include "Machines/RepRing.hpp"
#include "Machines/MalRep.hpp"
#include "Protocols/PostSacrifice.hpp"

int main(int argc, const char** argv)
{
    ez::ezOptionParser opt;
    HonestMajorityRingMachineWithSecurity<PostSacriRepRingShare,
            PostSacriRepFieldShare>(argc, argv, opt);
}
