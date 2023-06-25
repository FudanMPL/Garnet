/*
 * cowgear-offline.cpp
 *
 */

#include "SPDZ.hpp"
#include "Math/gfp.hpp"
#include "Protocols/CowGearShare.h"
#include "Protocols/CowGearOptions.h"
#include "Protocols/CowGearPrep.hpp"
#include "Processor/FieldMachine.hpp"
#include "Processor/OfflineMachine.hpp"

int main(int argc, const char** argv)
{
    ez::ezOptionParser opt;
    CowGearOptions::singleton = CowGearOptions(opt, argc, argv);
    DishonestMajorityFieldMachine<CowGearShare, CowGearShare, gf2n_short,
            OfflineMachine<DishonestMajorityMachine>>(argc, argv, opt);
}
