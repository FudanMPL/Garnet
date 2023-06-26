/*
 * EcdsaOptions.h
 *
 */

#ifndef ECDSA_ECDSAOPTIONS_H_
#define ECDSA_ECDSAOPTIONS_H_

#include "Tools/ezOptionParser.h"

class EcdsaOptions
{
public:
    bool prep_mul;
    bool fewer_rounds;
    bool check_open;
    bool check_beaver_open;
    bool R_after_msg;

    EcdsaOptions(ez::ezOptionParser& opt, int argc, const char** argv)
    {
        opt.add(
                "", // Default.
                0, // Required?
                0, // Number of args expected.
                0, // Delimiter if expecting multiple args.
                "Delay multiplication until signing", // Help description.
                "-D", // Flag token.
                "--delay-multiplication" // Flag token.
        );
        opt.add(
                "", // Default.
                0, // Required?
                0, // Number of args expected.
                0, // Delimiter if expecting multiple args.
                "Fewer rounds, more EC", // Help description.
                "-P", // Flag token.
                "--parallel-open" // Flag token.
        );
        opt.add(
                "", // Default.
                0, // Required?
                0, // Number of args expected.
                0, // Delimiter if expecting multiple args.
                "Skip checking final openings (but not necessarily openings for Beaver; only relevant with active protocols)", // Help description.
                "-C", // Flag token.
                "--no-open-check" // Flag token.
        );
        opt.add(
                "", // Default.
                0, // Required?
                0, // Number of args expected.
                0, // Delimiter if expecting multiple args.
                "Skip checking Beaver openings (only relevant with active protocols)", // Help description.
                "-B", // Flag token.
                "--no-beaver-open-check" // Flag token.
        );
        opt.add(
                "", // Default.
                0, // Required?
                0, // Number of args expected.
                0, // Delimiter if expecting multiple args.
                "Only open R after message is known", // Help description.
                "-R", // Flag token.
                "--R-after-msg" // Flag token.
        );
        opt.parse(argc, argv);
        prep_mul = not opt.isSet("-D");
        fewer_rounds = opt.isSet("-P");
        check_open = not opt.isSet("-C");
        check_beaver_open = not opt.isSet("-B");
        R_after_msg = opt.isSet("-R");
        opt.resetArgs();
    }
};

#endif /* ECDSA_ECDSAOPTIONS_H_ */
