/*
 * malicious-rep-field-party.cpp
 *
 */

#include "Protocols/MaliciousRep3Share.h"
#include "Processor/FieldMachine.hpp"
#include "Machines/Rep.hpp"
#include "Machines/MalRep.hpp"
#include "Math/gfp.hpp"

int main(int argc, const char** argv)
{
    HonestMajorityFieldMachine<MaliciousRep3Share>(argc, argv);
}
