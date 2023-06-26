/*
 * SPDZ2k.cpp
 *
 */

#include "SPDZ2k.hpp"

#ifdef RING_SIZE
template class Machine<Spdz2kShare<RING_SIZE, SPDZ2K_DEFAULT_SECURITY>, Share<gf2n>>;
#endif
