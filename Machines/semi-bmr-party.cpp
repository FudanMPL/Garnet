/*
 * semi-bmr-party.cpp
 *
 */

#include "BMR/RealProgramParty.hpp"
#include "Machines/Semi.hpp"

int main(int argc, const char** argv)
{
    RealProgramParty<SemiShare<gf2n_long>>(argc, argv);
}
