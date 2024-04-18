/*
 * replicated-field-party.cpp
 *
 */

#include "Math/gfp.hpp"
#include "Processor/FieldMachine.hpp"
#include "Machines/Rep.hpp"

int main(int argc, const char** argv)
{
    HonestMajorityFieldMachine<Rep3Share>(argc, argv);
}
