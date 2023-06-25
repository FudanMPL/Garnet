/*
 * mal-shamir-ecdsa-party.cpp
 *
 */

#include "Protocols/ShamirShare.h"

#include "Protocols/Shamir.hpp"
#include "Protocols/ShamirInput.hpp"

#include "hm-ecdsa-party.hpp"

int main(int argc, const char** argv)
{
    ez::ezOptionParser opt;
    ShamirOptions(opt, argc, argv);
    run<ShamirShare>(argc, argv);
}
