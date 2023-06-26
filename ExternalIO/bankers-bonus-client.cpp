/*
 * Demonstrate external client inputing and receiving outputs from a SPDZ process, 
 * following the protocol described in https://eprint.iacr.org/2015/1006.pdf.
 *
 * Provides a client to bankers_bonus.mpc program to calculate which banker pays for lunch based on
 * the private value annual bonus. Up to 8 clients can connect to the SPDZ engines running 
 * the bankers_bonus.mpc program.
 *
 * Each connecting client:
 * - sends an increasing id to identify the client, starting with 0
 * - sends an integer (0 meaining more players will join this round or 1 meaning stop the round and calc the result).
 * - sends an integer input (bonus value to compare)
 *
 * The result is returned authenticated with a share of a random value:
 * - share of winning unique id [y]
 * - share of random value [r]
 * - share of winning unique id * random value [w]
 *   winning unique id is valid if ∑ [y] * ∑ [r] = ∑ [w]
 *
 * To run with 2 parties / SPDZ engines:
 *   ./Scripts/setup-online.sh to create triple shares for each party (spdz engine).
 *   ./Scripts/setup-clients.sh to create SSL keys and certificates for clients
 *   ./compile.py bankers_bonus
 *   ./Scripts/run-online.sh bankers_bonus to run the engines.
 *
 *   ./bankers-bonus-client.x 0 2 100 0
 *   ./bankers-bonus-client.x 1 2 200 0
 *   ./bankers-bonus-client.x 2 2 50 1
 *
 *   Expect winner to be second client with id 1.
 */

#include "Math/gfp.h"
#include "Math/gf2n.h"
#include "Networking/sockets.h"
#include "Networking/ssl_sockets.h"
#include "Tools/int.h"
#include "Math/Setup.h"
#include "Protocols/fake-stuff.h"

#include "Math/gfp.hpp"
#include "Client.hpp"

#include <sodium.h>
#include <iostream>
#include <sstream>
#include <fstream>

template<class T, class U>
void one_run(T salary_value, Client& client)
{
    // Run the computation
    client.send_private_inputs<T>({salary_value});
    cout << "Sent private inputs to each SPDZ engine, waiting for result..." << endl;

    // Get the result back (client_id of winning client)
    U result = client.receive_outputs<T>(1)[0];

    cout << "Winning client id is : " << result << endl;
}

template<class T, class U>
void run(double salary_value, Client& client)
{
    // sint
    one_run<T, U>(long(round(salary_value)), client);
    // sfix with f = 16
    one_run<T, U>(long(round(salary_value * exp2(16))), client);
}

int main(int argc, char** argv)
{
    int my_client_id;
    int nparties;
    double salary_value;
    int finish;
    int port_base = 14000;

    if (argc < 5) {
        cout << "Usage is bankers-bonus-client <client identifier> <number of spdz parties> "
           << "<salary to compare> <finish (0 false, 1 true)> <optional host names..., default localhost> "
           << "<optional spdz party port base number, default 14000>" << endl;
        exit(0);
    }

    my_client_id = atoi(argv[1]);
    nparties = atoi(argv[2]);
    salary_value = atof(argv[3]);
    finish = atoi(argv[4]);
    vector<string> hostnames(nparties, "localhost");

    if (argc > 5)
    {
        if (argc < 5 + nparties)
        {
            cerr << "Not enough hostnames specified";
            exit(1);
        }

        for (int i = 0; i < nparties; i++)
            hostnames[i] = argv[5 + i];
    }

    if (argc > 5 + nparties)
        port_base = atoi(argv[5 + nparties]);

    bigint::init_thread();

    // Setup connections from this client to each party socket
    Client client(hostnames, port_base, my_client_id);
    auto& specification = client.specification;
    auto& sockets = client.sockets;
    for (int i = 0; i < nparties; i++)
    {
        octetStream os;
        os.store(finish);
        os.Send(sockets[i]);
    }
    cout << "Finish setup socket connections to SPDZ engines." << endl;

    int type = specification.get<int>();
    switch (type)
    {
    case 'p':
    {
        gfp::init_field(specification.get<bigint>());
        cerr << "using prime " << gfp::pr() << endl;
        run<gfp, gfp>(salary_value, client);
        break;
    }
    case 'R':
    {
        int R = specification.get<int>();
        switch (R)
        {
        case 64:
            run<Z2<64>, Z2<64>>(salary_value, client);
            break;
        case 104:
            run<Z2<104>, Z2<64>>(salary_value, client);
            break;
        case 128:
            run<Z2<128>, Z2<64>>(salary_value, client);
            break;
        default:
            cerr << R << "-bit ring not implemented";
            exit(1);
        }
        break;
    }
    default:
        cerr << "Type " << type << " not implemented";
        exit(1);
    }

    return 0;
}
