/*
 * sy-rep-ring-party.cpp
 *
 */

#include "Protocols/SpdzWiseRingShare.h"
#include "Protocols/MaliciousRep3Share.h"
#include "Protocols/SpdzWiseMC.h"
#include "Protocols/SpdzWiseRingPrep.h"
#include "Protocols/SpdzWiseInput.h"
#include "Protocols/MalRepRingPrep.h"
#include "Processor/RingOptions.h"
#include "GC/MaliciousCcdSecret.h"
#include "GC/SemiHonestRepPrep.h"

#include "Processor/RingMachine.hpp"
#include "Protocols/Replicated.hpp"
#include "Protocols/Share.hpp"
#include "Protocols/fake-stuff.hpp"
#include "Protocols/SpdzWise.hpp"
#include "Protocols/SpdzWiseRing.hpp"
#include "Protocols/SpdzWisePrep.hpp"
#include "Protocols/SpdzWiseInput.hpp"
#include "Protocols/SpdzWiseShare.hpp"
#include "Protocols/PostSacrifice.hpp"
#include "Protocols/MalRepRingPrep.hpp"
#include "Protocols/MaliciousRepPrep.hpp"
#include "Protocols/RepRingOnlyEdabitPrep.hpp"
#include "Processor/Data_Files.hpp"
#include "Processor/Instruction.hpp"
#include "Processor/Machine.hpp"
#include "GC/ShareSecret.hpp"
#include "GC/RepPrep.hpp"
#include "GC/ThreadMaster.hpp"
#include "MalRep.hpp"

int main(int argc, const char** argv)
{
    ez::ezOptionParser opt;
    HonestMajorityRingMachineWithSecurity<SpdzWiseRingShare, SpdzWiseRepFieldShare>(
            argc, argv, opt);
}
