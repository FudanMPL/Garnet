/*
 * replicated-ring-party.cpp
 *
 */
#define BIG_DOMAIN_FOR_RSS

#include "Protocols/Rep3Share2k.h"
#include "Protocols/Rep3Share128.h"
#include "Processor/RingOptions.h"
#include "Math/Integer.h"
#include "Machines/RepRing.hpp"
#include "Processor/RingMachine.hpp"


int main(int argc, const char** argv)
{
    HonestMajorityRingMachine<Rep3Share2, Rep3Share>(argc, argv);
}
