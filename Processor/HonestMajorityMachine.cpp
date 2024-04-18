/*
 * ReplicatedMachine.cpp
 *
 */

#include "HonestMajorityMachine.h"

#include "Tools/ezOptionParser.h"
#include "Tools/benchmarking.h"
#include "Tools/NetworkOptions.h"
#include "Networking/Server.h"
#include "Protocols/Rep3Share.h"
#include "Processor/Machine.h"

#include "OnlineMachine.hpp"

HonestMajorityMachine::HonestMajorityMachine(int argc, const char** argv,
        ez::ezOptionParser& opt, OnlineOptions& online_opts, int nplayers) :
        OnlineMachine(argc, argv, opt, online_opts, nplayers)
{
    opt.add(
            "", // Default.
            0, // Required?
            0, // Number of args expected.
            0, // Delimiter if expecting multiple args.
            "Unencrypted communication.", // Help description.
            "-u", // Flag token.
            "--unencrypted" // Flag token.
    );
    online_opts.finalize(opt, argc, argv);
    OnlineOptions::singleton = online_opts;

    use_encryption = not opt.get("-u")->isSet;

    if (not use_encryption)
        insecure("unencrypted communication");

    start_networking();
}
