/*
 * prime.cpp
 *
 */

#include "Math/gfp.h"
#include "Math/Setup.h"

int main(int argc, char** argv)
{
    int lgp = gfp0::size_in_bits();
    if (argc > 1)
        lgp = atoi(argv[1]);
    if (argc > 2)
        cout << generate_prime(lgp, 1 << atoi(argv[2])) << endl;
    else
        cout << SPDZ_Data_Setup_Primes(lgp) << endl;
}
