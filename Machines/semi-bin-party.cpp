/*
 * semi-bin-party.cpp
 *
 */

#include "GC/ShareParty.h"
#include "GC/SemiSecret.h"

#include "GC/ShareParty.hpp"
#include "GC/ShareSecret.hpp"

#include "GC/Machine.hpp"
#include "GC/Program.hpp"
#include "GC/Thread.hpp"
#include "GC/ThreadMaster.hpp"
#include "GC/Processor.hpp"
#include "GC/SemiSecret.hpp"
#include "Protocols/MAC_Check_Base.hpp"
#include "Protocols/SemiMC.hpp"
#include "Protocols/SemiInput.hpp"
#include "Protocols/ReplicatedInput.hpp"
#include "Processor/Instruction.hpp"
#include "Processor/Input.hpp"

int main(int argc, const char** argv)
{
    GC::simple_binary_main<GC::SemiSecret>(argc, argv);
}
