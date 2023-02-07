/*
 * binary-example.cpp
 *
 */

#include "GC/TinierSecret.h"
#include "GC/PostSacriSecret.h"
#include "GC/CcdSecret.h"
#include "GC/MaliciousCcdSecret.h"
#include "GC/AtlasSecret.h"
#include "GC/TinyMC.h"
#include "GC/VectorInput.h"
#include "GC/PostSacriBin.h"
#include "Protocols/ProtocolSet.h"

#include "GC/ShareSecret.hpp"
#include "GC/CcdPrep.hpp"
#include "GC/TinierSharePrep.hpp"
#include "GC/RepPrep.hpp"
#include "GC/Secret.hpp"
#include "GC/TinyPrep.hpp"
#include "GC/ThreadMaster.hpp"
#include "GC/SemiSecret.hpp"
#include "Protocols/Atlas.hpp"
#include "Protocols/MaliciousRepPrep.hpp"
#include "Protocols/Share.hpp"
#include "Protocols/MaliciousRepMC.hpp"
#include "Protocols/Shamir.hpp"
#include "Protocols/fake-stuff.hpp"
#include "Machines/ShamirMachine.hpp"
#include "Machines/Rep4.hpp"
#include "Machines/Rep.hpp"

template<class T>
void run(int argc, char** argv);

int main(int argc, char** argv)
{
    // need player number and number of players
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0]
                << "<my number: 0/1/...> <total number of players> [protocol [bit length [threshold]]]"
                << endl;
        exit(1);
    }

    string protocol = "Tinier";
    if (argc > 3)
        protocol = argv[3];

    if (protocol == "Tinier")
        run<GC::TinierSecret<gf2n_mac_key>>(argc, argv);
    else if (protocol == "Rep3")
        run<GC::SemiHonestRepSecret>(argc, argv);
    else if (protocol == "Rep4")
        run<GC::Rep4Secret>(argc, argv);
    else if (protocol == "PS")
        run<GC::PostSacriSecret>(argc, argv);
    else if (protocol == "Semi")
        run<GC::SemiSecret>(argc, argv);
    else if (protocol == "CCD" or protocol == "MalCCD" or protocol == "Atlas")
    {
        int nparties = (atoi(argv[2]));
        int threshold = (nparties - 1) / 2;
        if (argc > 5)
            threshold = atoi(argv[5]);
        assert(2 * threshold < nparties);
        ShamirOptions::s().threshold = threshold;
        ShamirOptions::s().nparties = nparties;

        if (protocol == "CCD")
            run<GC::CcdSecret<gf2n_<octet>>>(argc, argv);
        else if (protocol == "MalCCD")
            run<GC::MaliciousCcdSecret<gf2n_short>>(argc, argv);
        else
            run<GC::AtlasSecret>(argc, argv);
    }
    else
    {
        cerr << "Unknown protocol: " << protocol << endl;
        exit(1);
    }
}

template<class T>
void run(int argc, char** argv)
{
    // run 16-bit computation by default
    int n_bits = 16;
    if (argc > 4)
        n_bits = atoi(argv[4]);

    // set up networking on localhost
    int my_number = atoi(argv[1]);
    int n_parties = atoi(argv[2]);
    int port_base = 9999;
    Names N(my_number, n_parties, "localhost", port_base);
    CryptoPlayer P(N);

    // protocol setup (domain, MAC key if needed etc)
    BinaryProtocolSetup<T> setup(P);

    // set of protocols (input, multiplication, output)
    BinaryProtocolSet<T> set(P, setup);
    auto& input = set.input;
    auto& protocol = set.protocol;
    auto& output = set.output;

    int n = 10;
    vector<T> a(n), b(n);

    input.reset_all(P);
    for (int i = 0; i < n; i++)
        input.add_from_all(i + P.my_num(), n_bits);
    input.exchange();
    for (int i = 0; i < n; i++)
    {
        a[i] = input.finalize(0, n_bits);
        b[i] = input.finalize(1, n_bits);
    }

    protocol.init_mul();
    for (int i = 0; i < n; i++)
        protocol.prepare_mul(a[i], b[i], n_bits);
    protocol.exchange();
    output.init_open(P, n);
    for (int i = 0; i < n; i++)
    {
        auto c = protocol.finalize_mul(n_bits);
        output.prepare_open(c);
    }
    output.exchange(P);
    set.check();

    cout << "result: ";
    for (int i = 0; i < n; i++)
        cout << output.finalize_open() << " ";
    cout << endl;

    set.check();
}
