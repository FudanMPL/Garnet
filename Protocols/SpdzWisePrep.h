/*
 * SpdzWisePrep.h
 *
 */

#ifndef PROTOCOLS_SPDZWISEPREP_H_
#define PROTOCOLS_SPDZWISEPREP_H_

#include "ReplicatedPrep.h"

template<class T> class MaliciousShamirShare;

/**
 * Preprocessing for honest-majority protocol with MAC
 */
template<class T>
class SpdzWisePrep : public MaliciousRingPrep<T>
{
    typedef MaliciousRingPrep<T> super;

    void buffer_triples();
    void buffer_bits();

    void buffer_inputs(int player);

    template<int X, int L>
    void buffer_bits(MaliciousRep3Share<gfp_<X, L>>);
    template<int X, int L>
    void buffer_bits(MaliciousShamirShare<gfp_<X, L>>);
    template<class U>
    void buffer_bits(U);

public:
    SpdzWisePrep(SubProcessor<T>* proc, DataPositions& usage) :
        BufferPrep<T>(usage),
        BitPrep<T>(proc, usage), RingPrep<T>(proc, usage),
        MaliciousDabitOnlyPrep<T>(proc, usage),
        MaliciousRingPrep<T>(proc, usage)
    {
    }
};

#endif /* PROTOCOLS_SPDZWISEPREP_H_ */
