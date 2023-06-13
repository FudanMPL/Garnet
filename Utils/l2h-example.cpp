/*
 * l2h-example.cpp
 *
 */

#include "Protocols/ProtocolSet.h"

#include "Math/gfp.hpp"
#include "Machines/SPDZ.hpp"
#include "Protocols/MascotPrep.hpp"

int main(int argc, char** argv)
{
    // need player number and number of players
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << "<my number: 0/1/...> <total number of players>" << endl;
        exit(1);
    }

    // set up networking on localhost
    int my_number = atoi(argv[1]);
    int n_parties = atoi(argv[2]);
    int port_base = 9999;
    Names N(my_number, n_parties, "localhost", port_base);

    // template parameters are share types for integer and GF(2^n) computation
    Machine<Share<gfp0>, Share<gf2n>> machine(N);

    // protocols to be used directly
    ProtocolSet<Share<gfp0>> set(machine.get_player(), machine.get_sint_mac_key());

    // data to be used in steps
    set.input.reset_all(machine.get_player());
    set.input.add_from_all(2 + my_number);
    set.input.exchange();
    machine.Mp.MS.resize(n_parties);
    for (int i = 0; i < n_parties; i++)
        machine.Mp.MS[i] = set.input.finalize(i);

    machine.run_step("l2h_multiplication");
    machine.run_step("l2h_comparison");

    // check results
    // multiplication
    assert(set.output.open(machine.Mp.MS[2], machine.get_player()) == 6);
    // comparison
    assert(set.output.open(machine.Mp.MS[3], machine.get_player()) == 1);

    set.check();

    // print usage
    auto res = machine.stop_threads();
    res.first.print_cost();
}
