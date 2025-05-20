/*
 * mal-shamir-bmr-party.cpp
 *
 */

#include "BMR/RealProgramParty.hpp"
#include "Machines/ShamirMachine.hpp"
#include "Math/Z2k.hpp"
#include "Machines/MalRep.hpp"

int main(int argc, const char** argv)
{
	ez::ezOptionParser opt;
	ShamirOptions::singleton = {opt, argc, argv};
	RealProgramParty<MaliciousShamirShare<gf2n_long>>(argc, argv);
}
