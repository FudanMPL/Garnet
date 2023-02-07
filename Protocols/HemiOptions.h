/*
 * HemiOptions.h
 *
 */

#ifndef PROTOCOLS_HEMIOPTIONS_H_
#define PROTOCOLS_HEMIOPTIONS_H_

#include "Tools/ezOptionParser.h"

class HemiOptions
{
public:
    static HemiOptions singleton;

    bool plain_matmul;


    HemiOptions() :
            plain_matmul(false)
    {
    }

    HemiOptions(ez::ezOptionParser& opt, int argc,
            const char** argv)
    {
        opt.add("", // Default.
                0, // Required?
                0, // Number of args expected.
                0, // Delimiter if expecting multiple args.
                "Don't use homomorphic matrix multplication", // Help description.
                "-M", // Flag token.
                "--plain-matmul" // Flag token.
                );

        opt.parse(argc, argv);
        plain_matmul = opt.isSet("-M");
        opt.resetArgs();
    }
};

#endif /* PROTOCOLS_HEMIOPTIONS_H_ */
