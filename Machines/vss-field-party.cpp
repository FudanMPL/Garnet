/*
 * vss-field-party.cpp
 *
 */

#include "Math/gfp.hpp"
#include "Processor/FieldMachine.hpp"
#include "Machines/Semi.hpp"
#include "Machines/vss-field-party.h"
#include "Machines/vss-field-party.hpp"
#include "Protocols/VssField.h"
#include "Protocols/VssFieldShare.h"

int main(int argc, const char** argv)
{
    // ez::ezOptionParser opt;
    VssFieldMachineSpec<VssFieldShare>(argc, argv);
}