/*
 * KeyGen.cpp
 *
 */

#include "FHEOffline/DataSetup.h"
#include "Processor/OnlineOptions.h"

#include "Protocols/HighGearKeyGen.hpp"

template<>
void PartSetup<FFT_Data>::key_and_mac_generation(Player& P,
        MachineBase& machine, int, false_type)
{
    HighGearKeyGen<0, 0>(P, params).run(*this, machine);
}

template<>
void PartSetup<P2Data>::key_and_mac_generation(Player& P, MachineBase& machine,
        int, false_type)
{
    HighGearKeyGen<0, 0>(P, params).run(*this, machine);
}
