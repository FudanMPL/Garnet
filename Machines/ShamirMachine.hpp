/*
 * ShamirMachine.cpp
 *
 */

#include <Machines/ShamirMachine.h>
#include "Protocols/ShamirShare.h"
#include "Protocols/MaliciousShamirShare.h"
#include "Math/gfp.h"
#include "Math/gf2n.h"
#include "GC/VectorProtocol.h"
#include "GC/CcdPrep.h"
#include "GC/TinyMC.h"
#include "GC/MaliciousCcdSecret.h"
#include "GC/VectorInput.h"

#include "Processor/FieldMachine.hpp"

#include "Processor/Data_Files.hpp"
#include "Processor/Instruction.hpp"
#include "Processor/Machine.hpp"
#include "Protocols/ShamirInput.hpp"
#include "Protocols/Shamir.hpp"
#include "Protocols/ShamirMC.hpp"
#include "Protocols/MaliciousShamirMC.hpp"
#include "Protocols/MaliciousShamirPO.hpp"
#include "Protocols/MAC_Check_Base.hpp"
#include "Protocols/Beaver.hpp"
#include "Protocols/Spdz2kPrep.hpp"
#include "Protocols/ReplicatedPrep.hpp"
#include "Protocols/MalRepRingPrep.hpp"
#include "GC/ShareSecret.hpp"
#include "GC/VectorProtocol.hpp"
#include "GC/Secret.hpp"
#include "GC/CcdPrep.hpp"
#include "Math/gfp.hpp"

ShamirOptions ShamirOptions::singleton;

ShamirOptions& ShamirOptions::s()
{
    return singleton;
}

ShamirOptions::ShamirOptions(int nparties, int threshold) :
        nparties(nparties), threshold(threshold)
{
}

ShamirOptions::ShamirOptions(ez::ezOptionParser& opt, int argc, const char** argv)
{
    opt.add(
            "3", // Default.
            0, // Required?
            1, // Number of args expected.
            0, // Delimiter if expecting multiple args.
            "Number of players", // Help description.
            "-N", // Flag token.
            "--nparties" // Flag token.
    );
    opt.add(
            "", // Default.
            0, // Required?
            1, // Number of args expected.
            0, // Delimiter if expecting multiple args.
            "Number of corrupted parties (default: just below half)", // Help description.
            "-T", // Flag token.
            "--threshold" // Flag token.
    );
    opt.parse(argc, argv);
    opt.get("-N")->getInt(nparties);
    set_threshold(opt);
    opt.resetArgs();
}

void ShamirOptions::set_threshold(ez::ezOptionParser& opt)
{
    if (opt.isSet("-T"))
        opt.get("-T")->getInt(threshold);
    else
        threshold = (nparties - 1) / 2;
#ifdef VERBOSE
    cerr << "Using threshold " << threshold << " out of " << nparties << endl;
#endif
    if (2 * threshold >= nparties)
        throw runtime_error("threshold too high");
    if (threshold < 1)
    {
        cerr << "Threshold has to be positive" << endl;
        exit(1);
    }
}

template<template<class U> class T>
ShamirMachineSpec<T>::ShamirMachineSpec(int argc, const char** argv)
{
    auto& opts = ShamirOptions::singleton;
    ez::ezOptionParser opt;
    opts = {opt, argc, argv};
    HonestMajorityFieldMachine<T>(argc, argv, opt, opts.nparties);
}
