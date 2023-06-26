/*
 * lowgear-party.cpp
 *
 */

#include "Protocols/LowGearShare.h"

#include "SPDZ.hpp"
#include "Protocols/CowGearPrep.hpp"

int main(int argc, const char** argv)
{
    ez::ezOptionParser opt;
    CowGearOptions::singleton = CowGearOptions(opt, argc, argv, false);
    DishonestMajorityFieldMachine<LowGearShare, LowGearShare, gf2n_short>(argc, argv, opt);
}
