/*
 * mixed-example.cpp
 *
 */

#include "Protocols/ProtocolSet.h"

#include "Machines/SPDZ.hpp"
#include "Machines/SPDZ2k.hpp"
#include "Machines/Semi2k.hpp"
#include "Machines/Rep.hpp"
#include "Machines/Rep4.hpp"
#include "Machines/Atlas.hpp"

template<class T>
void run(char** argv);

int main(int argc, char** argv)
{
    // need player number and number of players
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0]
                << "<my number: 0/1/...> <total number of players> [protocol]"
                << endl;
        exit(1);
    }

    string protocol = "SPDZ2k";
    if (argc > 3)
        protocol = argv[3];

    if (protocol == "SPDZ2k")
        run<Spdz2kShare<64, 64>>(argv);
    else if (protocol == "Semi2k")
        run<Semi2kShare<64>>(argv);
    else if (protocol == "Rep3")
        run<Rep3Share2<64>>(argv);
    else if (protocol == "Rep4")
        run<Rep4Share2<64>>(argv);
    else if (protocol == "Atlas")
        run<AtlasShare<gfp_<0, 2>>>(argv);
    else
    {
        cerr << "Unknown protocol: " << protocol << endl;
        exit(1);
    }
}

template<class T>
void run(char** argv)
{
    // reduce batch size
    OnlineOptions::singleton.bucket_size = 5;
    OnlineOptions::singleton.batch_size = 100;

    // set up networking on localhost
    int my_number = atoi(argv[1]);
    int n_parties = atoi(argv[2]);
    int port_base = 9999;
    Names N(my_number, n_parties, "localhost", port_base);
    CryptoPlayer P(N);

    // protocol setup (domain, MAC key if needed etc)
    MixedProtocolSetup<T> setup(P);

    // set of protocols (bit_input, multiplication, output)
    MixedProtocolSet<T> set(P, setup);
    auto& output = set.output;
    auto& bit_input = set.binary.input;
    auto& bit_protocol = set.binary.protocol;
    auto& bit_output = set.binary.output;
    auto& prep = set.preprocessing;

    int n = 10;
    int n_bits = 16;
    vector<typename T::bit_type> a(n), b(n);

    // inputs in binary domain
    bit_input.reset_all(P);
    for (int i = 0; i < n; i++)
        bit_input.add_from_all(i + P.my_num(), n_bits);
    bit_input.exchange();
    for (int i = 0; i < n; i++)
    {
        a[i] = bit_input.finalize(0, n_bits);
        b[i] = bit_input.finalize(1, n_bits);
    }

    // compute AND in binary domain
    bit_protocol.init_mul();
    for (int i = 0; i < n; i++)
        bit_protocol.prepare_mul(a[i], b[i], n_bits);
    bit_protocol.exchange();
    bit_protocol.check();
    bit_output.init_open(P, n * n_bits);
    PointerVector<pair<T, typename T::bit_type>> dabits;
    for (int i = 0; i < n; i++)
    {
        auto c = bit_protocol.finalize_mul(n_bits);

        // mask result with dabits and open
        for (int j = 0; j < n_bits; j++)
        {
            dabits.push_back({});
            auto& dabit = dabits.back();
            prep.get_dabit(dabit.first, dabit.second);
            bit_output.prepare_open(
                    typename T::bit_type::part_type(
                            dabit.second.get_bit(0) + c.get_bit(j)));
        }
    }
    bit_output.exchange(P);
    output.init_open(P, n);
    for (int i = 0; i < n; i++)
    {
        T res;
        // unmask via XOR and recombine
        for (int j = 0; j < n_bits; j++)
        {
            typename T::clear masked = bit_output.finalize_open().get_bit(0);
            auto mask = dabits.next().first;
            res += (mask - mask * masked * 2
                    + T::constant(masked, P.my_num(), setup.get_mac_key()))
                    << j;
        }
        output.prepare_open(res);
    }
    output.exchange(P);
    set.check();

    cout << "result: ";
    for (int i = 0; i < n; i++)
        cout << output.finalize_open() << " ";
    cout << endl;

    set.check();
}
