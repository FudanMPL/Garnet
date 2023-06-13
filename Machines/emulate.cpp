/*
 * emulate.cpp
 *
 */

#include "Protocols/FakeShare.h"
#include "Processor/Machine.h"
#include "Math/Z2k.h"
#include "Math/gf2n.h"
#include "Processor/RingOptions.h"

#include "Processor/Machine.hpp"
#include "Processor/OnlineOptions.hpp"
#include "Math/Z2k.hpp"
#include "Protocols/Replicated.hpp"
#include "Protocols/ShuffleSacrifice.hpp"
#include "Protocols/ReplicatedPrep.hpp"
#include "Protocols/FakeShare.hpp"
#include "Protocols/MalRepRingPrep.hpp"
#include "Protocols/MAC_Check_Base.hpp"

int main(int argc, const char** argv)
{
    OnlineOptions& online_opts = OnlineOptions::singleton;
    Names N;
    ez::ezOptionParser opt;
    RingOptions ring_opts(opt, argc, argv);
    online_opts = {opt, argc, argv, FakeShare<SignedZ2<64>>()};
    opt.parse(argc, argv);
    opt.syntax = string(argv[0]) + " <progname>";

    string progname;
    if (opt.firstArgs.size() > 1)
        progname = *opt.firstArgs.at(1);
    else if (not opt.lastArgs.empty())
        progname = *opt.lastArgs.at(0);
    else if (not opt.unknownArgs.empty())
        progname = *opt.unknownArgs.at(0);
    else
    {
        string usage;
        opt.getUsage(usage);
        cerr << usage << endl;
        exit(1);
    }

#ifdef ROUND_NEAREST_IN_EMULATION
    cerr << "Using nearest rounding instead of probabilistic truncation" << endl;
#else
    online_opts.set_trunc_error(opt);
#endif

    int R = ring_opts.ring_size_from_opts_or_schedule(progname);
    switch (R)
    {
#define X(L) \
    case L: \
        Machine<FakeShare<SignedZ2<L>>, FakeShare<gf2n>>(N, false, online_opts, \
                gf2n::default_degree()).run(progname); \
        break;
    X(64) X(128) X(256) X(192) X(384) X(512)
#ifdef RING_SIZE
    X(RING_SIZE)
#endif
#undef X
    default:
        cerr << "Not compiled for " << R << "-bit rings" << endl;
    }
}
