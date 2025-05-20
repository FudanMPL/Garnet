/*
 * ccd-party.cpp
 *
 */

#include "GC/CcdSecret.h"
#include "GC/TinyMC.h"
#include "GC/CcdPrep.h"
#include "GC/VectorInput.h"

#include "GC/ShareParty.hpp"
#include "GC/ShareSecret.hpp"
#include "GC/ThreadMaster.hpp"
#include "GC/Secret.hpp"
#include "GC/CcdPrep.hpp"
#include "Machines/ShamirMachine.hpp"

int main(int argc, const char** argv)
{
    gf2n_<octet>::init_field(8);
    ez::ezOptionParser opt;
    ShamirOptions::singleton = {opt, argc, argv};
    assert(ShamirOptions::singleton.nparties < (1 << gf2n_<char>::length()));
    GC::ShareParty<GC::CcdSecret<gf2n_<octet>>>(argc, argv, opt);
}
