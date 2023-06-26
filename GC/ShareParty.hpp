/*
 * ReplicatedParty.cpp
 *
 */

#include "ShareParty.h"

#include "Thread.h"
#include "ShareThread.h"
#include "SemiPrep.h"
#include "Networking/Server.h"
#include "Networking/CryptoPlayer.h"
#include "Tools/ezOptionParser.h"
#include "Tools/benchmarking.h"
#include "Tools/NetworkOptions.h"
#include "Protocols/fake-stuff.h"

#include "ShareThread.hpp"
#include "ThreadMaster.hpp"
#include "Thread.hpp"
#include "ShareSecret.hpp"

#include "Protocols/Replicated.hpp"
#include "Protocols/ReplicatedPrep.hpp"
#include "Protocols/fake-stuff.hpp"

namespace GC
{

template<class T>
ShareParty<T>* ShareParty<T>::singleton = 0;

template<class T>
void simple_binary_main(int argc, const char** argv, int default_batch_size = 0)
{
    ez::ezOptionParser opt;
    ShareParty<T>(argc, argv, opt, default_batch_size);
}

template<class T>
ShareParty<T>::ShareParty(int argc, const char** argv, ez::ezOptionParser& opt,
        int default_batch_size) :
        ThreadMaster<T>(online_opts), opt(opt),
        online_opts(this->opt, argc, argv,
                default_batch_size)
{
    if (singleton)
        throw runtime_error("there can only be one");
    singleton = this;

    int nplayers = 0;
    opt.parse(argc, argv);
    if (opt.get("-N"))
        opt.get("-N")->getInt(nplayers);
    opt.resetArgs();
    NetworkOptionsWithNumber network_opts(opt, argc, argv,
            nplayers > 0 ? nplayers : (T::dishonest_majority ? 2 : 3),
            T::variable_players and nplayers == 0);
    if (T::dishonest_majority)
        opt.add(
                "", // Default.
                0, // Required?
                0, // Number of args expected.
                0, // Delimiter if expecting multiple args.
                "Use encrypted channels.", // Help description.
                "-e", // Flag token.
                "--encrypted" // Flag token.
        );
    else
        opt.add(
                "", // Default.
                0, // Required?
                0, // Number of args expected.
                0, // Delimiter if expecting multiple args.
                "Unencrypted communication.", // Help description.
                "-u", // Flag token.
                "--unencrypted" // Flag token.
        );
    opt.add(
            "", // Default.
            0, // Required?
            0, // Number of args expected.
            0, // Delimiter if expecting multiple args.
            "Check opening by communication instead of hashing.", // Help description.
            "-c", // Flag token.
            "--communication" // Flag token.
    );
    online_opts.finalize(opt, argc, argv);
    OnlineOptions::singleton = online_opts;
    this->progname = online_opts.progname;
    int my_num = online_opts.playerno;

    if (T::dishonest_majority)
        this->machine.use_encryption = opt.get("-e")->isSet;
    else
        this->machine.use_encryption = not opt.get("-u")->isSet;

    this->machine.more_comm_less_comp = opt.get("-c")->isSet;

    if (not this->machine.use_encryption and not T::dishonest_majority)
        insecure("unencrypted communication");

    network_opts.start_networking(this->N, my_num);

    Player* P;
    if (this->machine.use_encryption)
        P = new CryptoPlayer(this->N, "shareparty");
    else
        P = new PlainPlayer(this->N, "shareparty");

    try
    {
        read_mac_key(
                get_prep_sub_dir<typename T::part_type>(PREP_DIR, network_opts.nplayers),
                this->N,
                this->mac_key);
    }
    catch (exception& e)
    {
        SeededPRNG G;
        this->mac_key.randomize(G);
    }

    T::MC::setup(*P);

    this->run();

    T::MC::teardown();
    delete P;

    this->machine.write_memory(this->N.my_num());
}

template<class T>
Thread<T>* ShareParty<T>::new_thread(int i)
{
    return new StandaloneShareThread<T>(i, *this);
}

template<class T>
void ShareParty<T>::post_run()
{
    DataPositions usage;
    for (auto thread : this->threads)
        usage.increase(dynamic_cast<StandaloneShareThread<T>*>(thread)->usage);
    usage.print_cost();
}

}
