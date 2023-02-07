/*
 * CowGearOptions.h
 *
 */

#ifndef PROTOCOLS_COWGEAROPTIONS_H_
#define PROTOCOLS_COWGEAROPTIONS_H_

#include "Tools/ezOptionParser.h"

class CowGearOptions
{
    bool use_top_gear;

public:
    static CowGearOptions singleton;

    int covert_security;

    CowGearOptions(bool covert = true);
    CowGearOptions(ez::ezOptionParser& opt, int argc, const char** argv,
            bool covert = true);

    bool top_gear()
    {
        return use_top_gear;
    }

    void set_top_gear(bool use)
    {
        use_top_gear = use;
    }
};

#endif /* PROTOCOLS_COWGEAROPTIONS_H_ */
