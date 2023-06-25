/*
 * tiny-party.cpp
 *
 */

#include "GC/TinySecret.h"
#include "GC/TinierSecret.h"
#include "GC/ShareParty.h"
#include "GC/TinyMC.h"
#include "GC/VectorInput.h"

#include "GC/ShareParty.hpp"
#include "GC/ShareSecret.hpp"
#include "GC/Machine.hpp"
#include "GC/Processor.hpp"
#include "GC/Program.hpp"
#include "GC/Thread.hpp"
#include "GC/ThreadMaster.hpp"
#include "GC/Secret.hpp"
#include "GC/TinyPrep.hpp"
#include "GC/CcdPrep.hpp"
#include "GC/TinierSharePrep.hpp"

#include "Processor/Instruction.hpp"
#include "Protocols/MAC_Check.hpp"
#include "Protocols/MAC_Check_Base.hpp"
#include "Protocols/Beaver.hpp"
#include "Protocols/MascotPrep.hpp"

int main(int argc, const char** argv)
{
    GC::simple_binary_main<GC::TinySecret<DEFAULT_SECURITY>>(argc, argv, 1000);
}
