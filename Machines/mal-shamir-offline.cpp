/*
 * mal-shamir-offline.cpp
 *
 */

#include "ShamirMachine.hpp"
#include "MalRep.hpp"
#include "Processor/OfflineMachine.hpp"

int main(int argc, const char** argv)
{
    auto& opts = ShamirOptions::singleton;
    ez::ezOptionParser opt;
    opts = {opt, argc, argv};
    HonestMajorityFieldMachine<MaliciousShamirShare,
            OfflineMachine<HonestMajorityMachine>>(argc, argv, opt,
            opts.nparties);
}
