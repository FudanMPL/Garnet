/*
 * chaigear-party.cpp
 *
 */

#include "Protocols/ChaiGearShare.h"

#include "SPDZ.hpp"
#include "Protocols/ChaiGearPrep.hpp"
#include "Processor/FieldMachine.hpp"
#include "Math/gfp.hpp"

int main(int argc, const char** argv)
{
    ez::ezOptionParser opt;
    CowGearOptions::singleton = CowGearOptions(opt, argc, argv);
    DishonestMajorityFieldMachine<ChaiGearShare, ChaiGearShare, gf2n_short>(
            argc, argv, opt);
}
