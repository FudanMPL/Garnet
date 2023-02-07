/*
 * CowGearOptions.cpp
 *
 */

#include "CowGearOptions.h"
#include "Tools/benchmarking.h"

#include <math.h>
#include <string>
using namespace std;

CowGearOptions CowGearOptions::singleton;

CowGearOptions::CowGearOptions(bool covert)
{
    if (covert)
    {
        covert_security = 20;
    }
    else
    {
        covert_security = -1;
    }

    use_top_gear = false;
}

CowGearOptions::CowGearOptions(ez::ezOptionParser& opt, int argc,
        const char** argv, bool covert) : CowGearOptions(covert)
{
    if (covert)
    {
        opt.add(
                "", // Default.
                0, // Required?
                1, // Number of args expected.
                0, // Delimiter if expecting multiple args.
                ("Covert security parameter c. "
                        "Cheating will be detected with probability 1/c (default: "
                        + to_string(covert_security) + ")").c_str(), // Help description.
                        "-c", // Flag token.
                        "--covert-security" // Flag token.
        );
    }
    opt.add(
            "", // Default.
            0, // Required?
            1, // Number of args expected.
            0, // Delimiter if expecting multiple args.
            "DEPRECATED: use -S/--security", // Help description.
            "-l", // Flag token.
            "--lowgear-security" // Flag token.
    );
    opt.add(
            "", // Default.
            0, // Required?
            0, // Number of args expected.
            0, // Delimiter if expecting multiple args.
            "Obsolete", // Help description.
            "-T", // Flag token.
            "--top-gear" // Flag token.
    );
    opt.add(
            "", // Default.
            0, // Required?
            0, // Number of args expected.
            0, // Delimiter if expecting multiple args.
            "Don't use TopGear", // Help description.
            "-J", // Flag token.
            "--no-top-gear" // Flag token.
    );
    opt.parse(argc, argv);
    if (opt.isSet("-c"))
        opt.get("-c")->getInt(covert_security);
    if (opt.isSet("-l"))
    {
        cerr << "Deprecated parameter, use -S/--security" << endl;
        exit(1);
    }
    use_top_gear = not opt.isSet("-J");
    if (opt.isSet("-T"))
        cerr << "WARNING: Option -T/--top-gear is obsolete "
            "because it is the default now. Use -J to deactivate it." << endl;
    opt.resetArgs();
}
