/*
 * NetworkOptions.cpp
 *
 */

#include "NetworkOptions.h"
#include "Networking/Server.h"

using namespace std;

NetworkOptions::NetworkOptions(ez::ezOptionParser& opt, int argc,
        const char** argv)
{
    opt.add(
            "localhost", // Default.
            0, // Required?
            1, // Number of args expected.
            0, // Delimiter if expecting multiple args.
            "Host where party 0 is running (default: localhost)", // Help description.
            "-h", // Flag token.
            "--hostname" // Flag token.
    );
    opt.add(
            "5000", // Default.
            0, // Required?
            1, // Number of args expected.
            0, // Delimiter if expecting multiple args.
            "Base port number (default: 5000).", // Help description.
            "-pn", // Flag token.
            "--portnum" // Flag token.
    );
    opt.parse(argc, argv);
    opt.get("-pn")->getInt(portnum_base);
    opt.get("-h")->getString(hostname);
    opt.resetArgs();
}

NetworkOptionsWithNumber::NetworkOptionsWithNumber(ez::ezOptionParser& opt,
        int argc, const char** argv, int default_nplayers, bool variable_nplayers) :
        NetworkOptions(opt, argc, argv)
{
    if (variable_nplayers)
        opt.add(
                to_string(default_nplayers).c_str(), // Default.
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
            "Filename containing list of party ip addresses. Alternative to --hostname for startup coordination.", // Help description.
            "-ip", // Flag token.
            "--ip-file-name" // Flag token.
    );

    opt.parse(argc, argv);

    if (variable_nplayers)
        opt.get("-N")->getInt(nplayers);
    else
        nplayers = default_nplayers;

    opt.get("-ip")->getString(ip_filename);

    opt.resetArgs();
}

Server* NetworkOptionsWithNumber::start_networking(Names& N, int my_num)
{
    if (ip_filename.length() > 0)
    {
        N.init(my_num, portnum_base, ip_filename, nplayers);
        return 0;
    }
    else
        return Server::start_networking(N, my_num, nplayers, hostname, portnum_base);
}
