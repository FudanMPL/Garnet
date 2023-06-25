/*
 * bmr-program-party.cpp
 *
 */

#include "BMR/Party.h"

int main(int argc, const char** argv)
{
	FakeProgramParty party(argc, argv);
	party.Start();
}
