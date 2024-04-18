/*
 * highgear-party.cpp
 *
 */

#include "Protocols/HighGearShare.h"

#include "SPDZ.hpp"
#include "Protocols/ChaiGearPrep.hpp"

int main(int argc, const char** argv)
{
    ez::ezOptionParser opt;
    CowGearOptions::singleton = CowGearOptions(opt, argc, argv, false);
    DishonestMajorityFieldMachine<HighGearShare, HighGearShare, gf2n_short>(argc, argv, opt);
}
