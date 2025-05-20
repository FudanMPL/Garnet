/*
 * ccd-party.cpp
 *
 */

#include "GC/MaliciousCcdSecret.h"
#include "GC/TinyMC.h"
#include "GC/CcdPrep.h"
#include "GC/VectorInput.h"

#include "GC/ShareParty.hpp"
#include "GC/ShareSecret.hpp"
#include "GC/ThreadMaster.hpp"
#include "GC/Secret.hpp"
#include "GC/CcdPrep.hpp"
#include "Machines/ShamirMachine.hpp"
#include "Machines/MalRep.hpp"

int main(int argc, const char** argv)
{
    ez::ezOptionParser opt;
    ShamirOptions::singleton = {opt, argc, argv};
    OnlineOptions opts(opt, argc, argv);
    gf2n_short::init_minimum(opts.security_parameter);
    GC::ShareParty<GC::MaliciousCcdSecret<gf2n_short>>(argc, argv, opt);
}
