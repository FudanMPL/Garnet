/*
 * LowGearKeyGen.cpp
 *
 */

#include "FHEOffline/DataSetup.h"
#include "Processor/OnlineOptions.h"

#include "Protocols/LowGearKeyGen.hpp"

template<>
void PairwiseSetup<FFT_Data>::key_and_mac_generation(Player& P,
        PairwiseMachine& machine, int, false_type)
{
    LowGearKeyGen<0>(P, machine, params).run(*this);
}

template<>
void PairwiseSetup<P2Data>::key_and_mac_generation(Player& P,
        PairwiseMachine& machine, int, false_type)
{
    LowGearKeyGen<0>(P, machine, params).run(*this);
}
