/*
 * MalRepRingOptions.cpp
 *
 */

#include "MalRepRingOptions.h"

MalRepRingOptions MalRepRingOptions::singleton;

MalRepRingOptions::MalRepRingOptions()
{
    shuffle = false;
}

MalRepRingOptions::MalRepRingOptions(ez::ezOptionParser& opt, int argc,
        const char** argv) : MalRepRingOptions()
{
    opt.add(
          "", // Default.
          0, // Required?
          0, // Number of args expected.
          0, // Delimiter if expecting multiple args.
          "Shuffle sacrifice (default: disabled)", // Help description.
          "-SH", // Flag token.
          "--shuffle" // Flag token.
    );
    opt.parse(argc, argv);
    shuffle = opt.isSet("-SH");
    opt.resetArgs();
}
