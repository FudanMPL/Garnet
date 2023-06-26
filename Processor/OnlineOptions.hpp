/*
 * OnlineOptions.hpp
 *
 */

#ifndef PROCESSOR_ONLINEOPTIONS_HPP_
#define PROCESSOR_ONLINEOPTIONS_HPP_

#include "OnlineOptions.h"

template<class T>
OnlineOptions::OnlineOptions(ez::ezOptionParser& opt, int argc,
        const char** argv, T, bool default_live_prep) :
        OnlineOptions(opt, argc, argv, T::dishonest_majority ? 1000 : 0,
                default_live_prep, T::clear::prime_field)
{
    if (T::has_trunc_pr)
        opt.add(
                to_string(trunc_error).c_str(), // Default.
                0, // Required?
                1, // Number of args expected.
                0, // Delimiter if expecting multiple args.
                ("Probabilistic truncation error (2^-x, default: "
                        + to_string(trunc_error) + ")").c_str(), // Help description.
                "-E", // Flag token.
                "--trunc-error" // Flag token.
        );

    if (T::dishonest_majority)
    {
        opt.add(
              "0", // Default.
              0, // Required?
              1, // Number of args expected.
              0, // Delimiter if expecting multiple args.
              "Sum at most n shares at once when using indirect communication", // Help description.
              "-s", // Flag token.
              "--opening-sum" // Flag token.
        );
        opt.add(
              "", // Default.
              0, // Required?
              0, // Number of args expected.
              0, // Delimiter if expecting multiple args.
              "Use player-specific threads for communication", // Help description.
              "-t", // Flag token.
              "--threads" // Flag token.
        );
        opt.add(
              "0", // Default.
              0, // Required?
              1, // Number of args expected.
              0, // Delimiter if expecting multiple args.
              "Maximum number of parties to send to at once", // Help description.
              "-mb", // Flag token.
              "--max-broadcast" // Flag token.
        );
    }
}

template<class T>
OnlineOptions::OnlineOptions(T) : OnlineOptions()
{
    if (T::dishonest_majority)
        batch_size = 1000;
}

#endif /* PROCESSOR_ONLINEOPTIONS_HPP_ */
