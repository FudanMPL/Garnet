/*
 * cowgear-party.cpp
 *
 */

#include "Protocols/CowGearShare.h"
#include "Protocols/CowGearPrep.h"
#include "Protocols/CowGearOptions.h"

#include "FHE/FHE_Params.h"
#include "FHE/FFT_Data.h"
#include "FHE/NTL-Subs.h"

#include "GC/TinierSecret.h"
#include "GC/TinyMC.h"

#include "SPDZ.hpp"

#include "Protocols/CowGearPrep.hpp"
#include "Processor/FieldMachine.hpp"
#include "Math/gfp.hpp"

int main(int argc, const char** argv)
{
    ez::ezOptionParser opt;
    CowGearOptions::singleton = CowGearOptions(opt, argc, argv);
    DishonestMajorityFieldMachine<CowGearShare, CowGearShare, gf2n_short>(argc, argv, opt);
}
