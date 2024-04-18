/*
 * ps-rep-field-party.cpp
 *
 */

#include "Protocols/PostSacriRepFieldShare.h"
#include "Processor/FieldMachine.hpp"
#include "Machines/Rep.hpp"
#include "Machines/MalRep.hpp"
#include "Protocols/PostSacrifice.hpp"
#include "Math/gfp.hpp"

int main(int argc, const char** argv)
{
    HonestMajorityFieldMachine<PostSacriRepFieldShare>(argc, argv);
}
