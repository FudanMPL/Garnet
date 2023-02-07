/*
 * semi-party.cpp
 *
 */

#include "Math/gfp.h"
#include "Protocols/SemiShare.h"
#include "Tools/SwitchableOutput.h"
#include "GC/SemiPrep.h"

#include "Processor/FieldMachine.hpp"
#include "Semi.hpp"
#include "GC/ShareSecret.hpp"
#include "Math/gfp.hpp"

int main(int argc, const char** argv)
{
    ez::ezOptionParser opt;
    DishonestMajorityFieldMachine<SemiShare>(argc, argv, opt);
}
