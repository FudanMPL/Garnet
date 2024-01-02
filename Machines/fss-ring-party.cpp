/*
 * replicated-ring-party.cpp
 *
 */



#include "Protocols/Rep3Share2k.h"
#include "Protocols/Fss3Prep.h"
#include "Protocols/Fss3Prep.hpp"
#include "Protocols/Fss.h"
#include "Protocols/Fss.hpp"
#include "Protocols/Fss3Share.h"
#include "Protocols/Fss3Share2k.h"
#include "Processor/RingOptions.h"
#include "Math/Integer.h"
#include "Machines/RepRing.hpp"
#include "Processor/RingMachine.hpp"

int main(int argc, const char** argv)
{
    HonestMajorityRingMachine<Fss3Share2, Fss3Share>(argc, argv);
}
