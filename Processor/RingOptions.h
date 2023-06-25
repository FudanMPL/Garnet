/*
 * RingOptions.h
 *
 */

#ifndef PROCESSOR_RINGOPTIONS_H_
#define PROCESSOR_RINGOPTIONS_H_

#include "Tools/ezOptionParser.h"
#include <string>
using namespace std;

class RingOptions
{
    bool R_is_set;

public:
    int R;

    RingOptions(ez::ezOptionParser& opt, int argc, const char** argv);

    int ring_size_from_opts_or_schedule(string progname);
};

#endif /* PROCESSOR_RINGOPTIONS_H_ */
