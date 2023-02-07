/*
 * abfllnow-party.cpp
 *
 */

#include "GC/PostSacriBin.h"
#include "GC/PostSacriSecret.h"
#include "GC/RepPrep.h"

#include "GC/ShareParty.hpp"
#include "GC/RepPrep.hpp"
#include "Protocols/MaliciousRepMC.hpp"

int main(int argc, const char** argv)
{
    GC::simple_binary_main<GC::PostSacriSecret>(argc, argv);
}
