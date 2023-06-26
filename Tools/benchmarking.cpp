/*
 * benchmarking.cpp
 *
 */

#include "benchmarking.h"

void insecure(string message, bool warning)
{
#ifdef INSECURE
    if (warning)
        cerr << "WARNING: insecure " << message << endl;
#else
    (void)warning;
    string msg = "You are trying to use insecure benchmarking functionality for "
            + message + ".\nYou can activate this at compile time "
                    "by adding -DINSECURE to the compiler options.\n"
                    "Make sure to run 'make clean' as well before compiling.";
    cerr << msg << endl;
#ifdef INSECURE_EXCEPTION
    throw exception();
#endif
    exit(1);
#endif
}

void insecure_fake(bool warning)
{
#if defined(INSECURE) or defined(INSECURE_FAKE)
    if (warning)
        cerr << "WARNING: insecure preprocessing" << endl;
#else
    (void) warning;
    insecure("preprocessing");
#endif
}
