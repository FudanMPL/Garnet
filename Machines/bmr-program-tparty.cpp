/*
 * bmr-program-tparty.cpp
 *
 */

#include "BMR/TrustedParty.h"

int main(int argc, char** argv)
{
	TrustedProgramParty party(argc, argv);
	party.Start();
}
