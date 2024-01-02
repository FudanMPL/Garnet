/*
 * replicated-ring-party.cpp
 *
 */
#define BIG_DOMAIN_FOR_RING
#define BIG_DOMAIN_USE_RSS
#define BigDomainShare Rep3Share128

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
