/*
 * rep-bmr-party.cpp
 *
 */

#include "BMR/RealProgramParty.hpp"
#include "Machines/Rep.hpp"

int main(int argc, const char** argv)
{
    RealProgramParty<Rep3Share<gf2n_long>>(argc, argv);
}
