/*
 * no-party.cpp
 *
 */

#include "Protocols/NoShare.h"

#include "Processor/OnlineMachine.hpp"
#include "Processor/Machine.hpp"
#include "Protocols/Replicated.hpp"
#include "Protocols/MalRepRingPrep.hpp"
#include "Protocols/ReplicatedPrep.hpp"
#include "Protocols/MAC_Check_Base.hpp"
#include "Math/gfp.hpp"
#include "Math/Z2k.hpp"

int main(int argc, const char** argv)
{
    ez::ezOptionParser opt;
    OnlineOptions::singleton = {opt, argc, argv};
    OnlineMachine machine(argc, argv, opt, OnlineOptions::singleton);
    OnlineOptions::singleton.finalize(opt, argc, argv);
    machine.start_networking();
    // use primes of length 65 to 128 for arithmetic computation
    machine.run<NoShare<gfp_<0, 2>>, NoShare<gf2n>>();
}
