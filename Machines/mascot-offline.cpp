/*
 * mascot-offline.cpp
 *
 */

#include "GC/TinierSecret.h"

#include "SPDZ.hpp"
#include "Math/gfp.hpp"
#include "Processor/FieldMachine.hpp"
#include "Processor/OfflineMachine.hpp"
#include "Protocols/MascotPrep.hpp"

int main(int argc, const char** argv)
{
    ez::ezOptionParser opt;
    DishonestMajorityFieldMachine<Share, Share, gf2n,
            OfflineMachine<DishonestMajorityMachine>>(argc, argv, opt);
}
