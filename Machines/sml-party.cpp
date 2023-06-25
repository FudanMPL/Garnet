/*
 * secureML-party.cpp
 *
 */

#include "Protocols/SmlShare.h"
#include "Protocols/SemiPrep2k.h"
#include "Math/gf2n.h"
#include "Processor/RingOptions.h"
#include "GC/SemiPrep.h"

#include "Protocols/SemiShare.h"
#include "Protocols/SemiMC.h"
#include "Protocols/SemiPrep.h"

#include "Processor/Data_Files.hpp"
#include "Processor/Instruction.hpp"
#include "Processor/Machine.hpp"
#include "Protocols/MascotPrep.hpp"
#include "Protocols/SemiPrep.hpp"
#include "Protocols/SemiInput.hpp"
#include "Protocols/MAC_Check_Base.hpp"
#include "Protocols/MAC_Check.hpp"
#include "Protocols/SemiMC.hpp"
#include "Protocols/Beaver.hpp"
#include "Protocols/MalRepRingPrep.hpp"
#include "GC/SemiSecret.hpp"
#include "GC/ShareSecret.hpp"
#include "Protocols/RepRingOnlyEdabitPrep.hpp"
#include "Processor/RingMachine.hpp"

int main(int argc, const char** argv)
{
    ez::ezOptionParser opt;
    DishonestMajorityRingMachine<SmlShare, SemiShare>(argc, argv, opt);
}
