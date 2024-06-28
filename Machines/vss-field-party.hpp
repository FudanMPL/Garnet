 /*
 * vss-field-machine.hpp
 *
 */

#include <Machines/vss-field-party.h>
#include "Processor/FieldMachine.hpp"
#include "Machines/Semi.hpp"
#include "Protocols/Vss.h"
#include "Protocols/VssFieldShare.h"

#include "Math/gfp.h"
#include "Math/gf2n.h"
#include "Math/gfp.hpp"

#include "Processor/Data_Files.hpp"
#include "Processor/Instruction.hpp"
#include "Processor/Machine.hpp"


VssFieldOptions VssFieldOptions::singleton;

VssFieldOptions& VssFieldOptions::s()
{
    return singleton;
}

VssFieldOptions::VssFieldOptions(int nparties, int npparties, int naparties, int ndparties) :
        nparties(nparties), npparties(npparties), naparties(naparties), ndparties(ndparties)
{
}

VssFieldOptions::VssFieldOptions(ez::ezOptionParser& opt, int argc, const char** argv)
{
    opt.add(
            "3", // Default.
            1, // Required?
            1, // Number of args expected.
            0, // Delimiter if expecting multiple args.
            "Number of parties", // Help description.
            "-N", // Flag token.
            "--nparties" // Flag token.
    );
    opt.add(
            "1", // Default.
            1, // Required?
            1, // Number of args expected.
            0, // Delimiter if expecting multiple args.
            "Number of privileged parties (default: one)", // Help description.
            "-NP", // Flag token.
            "--npparties" // Flag token.
    );
    opt.add(
            "2", // Default.
            1, // Required?
            1, // Number of args expected.
            0, // Delimiter if expecting multiple args.
            "Number of assistant parties (default: two)", // Help description.
            "-NA", // Flag token.
            "--naparties" // Flag token. >=p
    );
    opt.add(
            "1", // Default.
            1, // Required?
            1, // Number of args expected.
            0, // Delimiter if expecting multiple args.
            "Number of assistant parties allowed to drop out (default: one)", // Help description.
            "-ND", // Flag token.
            "--ndparties" // Flag token. <a
    );
    opt.parse(argc, argv);
    opt.get("-N")->getInt(nparties);
    set_npparties(opt);
    // set_nparties(opt);
    set_naparties(opt);
    set_ndparties(opt);
    opt.resetArgs();
}

void VssFieldOptions::set_npparties(ez::ezOptionParser& opt)
{
    if (opt.isSet("-NP"))
        opt.get("-NP")->getInt(npparties);
    else
        naparties = 1;
}

void VssFieldOptions::set_naparties(ez::ezOptionParser& opt)
{
    if (opt.isSet("-NA"))
        opt.get("-NA")->getInt(naparties);
    else
        naparties = 2;
#ifdef VERBOSE
    cerr << "Using naparties " << naparties << " out of " << nparties << endl;
#endif
    if (naparties < npparties)
        throw runtime_error("naparties not enough");
}

void VssFieldOptions::set_ndparties(ez::ezOptionParser& opt)
{
    if (opt.isSet("-ND"))
        opt.get("-ND")->getInt(ndparties);
    else
        ndparties = 1;
#ifdef VERBOSE
    cerr << "Using ndparties " << ndparties << " out of " << nparties << endl;
#endif
    if (ndparties >= naparties)
        throw runtime_error("ndparties too many");
    if (ndparties < 0)
    {
        cerr << "ndparties has to be positive" << endl;
        exit(1);
    }
}

template<template<class U> class T>
VssFieldMachineSpec<T>::VssFieldMachineSpec(int argc, const char** argv)
{
    auto& opts = VssFieldOptions::singleton;
    ez::ezOptionParser opt;
    opts = {opt, argc, argv};
    HonestMajorityFieldMachine<T>(argc, argv, opt, opts.nparties);
}