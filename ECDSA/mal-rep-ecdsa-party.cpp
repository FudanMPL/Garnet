/*
 * mal-rep-ecdsa-party.cpp
 *
 */

#include "Protocols/MaliciousRep3Share.h"

#include "hm-ecdsa-party.hpp"

int main(int argc, const char** argv)
{
    run<MaliciousRep3Share>(argc, argv);
}
