/*
 * Tinier.cpp
 *
 */

#include "GC/TinyMC.h"
#include "GC/TinierSecret.h"
#include "GC/VectorInput.h"

#include "GC/ShareParty.hpp"
#include "GC/Secret.hpp"
#include "GC/TinyPrep.hpp"
#include "GC/ShareSecret.hpp"
#include "GC/TinierSharePrep.hpp"
#include "GC/CcdPrep.hpp"
#include "GC/PersonalPrep.hpp"

//template class GC::ShareParty<GC::TinierSecret<gf2n_mac_key>>;
template class GC::CcdPrep<GC::TinierSecret<gf2n_mac_key>>;
template class Preprocessing<GC::TinierSecret<gf2n_mac_key>>;
template class GC::TinierSharePrep<GC::TinierShare<gf2n_mac_key>>;
template class GC::ShareSecret<GC::TinierSecret<gf2n_mac_key>>;
template class TripleShuffleSacrifice<GC::TinierSecret<gf2n_mac_key>>;
