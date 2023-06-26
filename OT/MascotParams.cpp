/*
 * TripleMachine.cpp
 *
 */

#include "MascotParams.h"

MascotParams::MascotParams()
{
    generateMACs = true;
    amplify = true;
    check = true;
    correlation_check = true;
    generateBits = false;
    use_extension = true;
    fewer_rounds = false;
    fiat_shamir = false;
    timerclear(&start);
}

void MascotParams::set_passive()
{
    generateMACs = amplify = check = correlation_check = false;
}
