/*
 * sy-shamir-party.cpp
 *
 */

#include "ShamirMachine.h"
#include "Protocols/SpdzWiseShare.h"
#include "Protocols/MaliciousShamirShare.h"
#include "Protocols/SpdzWiseMC.h"
#include "Protocols/SpdzWiseInput.h"
#include "Math/gfp.h"
#include "Math/gf2n.h"
#include "GC/CcdSecret.h"
#include "GC/MaliciousCcdSecret.h"
#include "GC/SemiHonestRepPrep.h"

#include "Protocols/Share.hpp"
#include "Protocols/SpdzWise.hpp"
#include "Protocols/SpdzWisePrep.hpp"
#include "Protocols/SpdzWiseInput.hpp"
#include "Protocols/SpdzWiseShare.hpp"
#include "Machines/ShamirMachine.hpp"
#include "Machines/MalRep.hpp"

template<class T>
using SpdzWiseShamirShare = SpdzWiseShare<MaliciousShamirShare<T>>;

int main(int argc, const char** argv)
{
    ShamirMachineSpec<SpdzWiseShamirShare>(argc, argv);
}
