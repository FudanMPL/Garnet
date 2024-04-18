/*
 * Spdz2kPrep.h
 *
 */

#ifndef PROTOCOLS_SPDZ2KPREP_H_
#define PROTOCOLS_SPDZ2KPREP_H_

#include "MascotPrep.h"
#include "RingOnlyPrep.h"

template<int, int> class Spdz2kShare;

template<class T, class U>
void bits_from_square_in_ring(vector<T>& bits, int buffer_size, U* bit_prep);

/**
 * SPDZ2k preprocessing
 */
template<class T>
class Spdz2kPrep : public virtual MaliciousRingPrep<T>,
        public virtual MascotTriplePrep<T>,
        public virtual RingOnlyPrep<T>
{
    typedef typename T::bit_prep_type BitShare;
    DataPositions bit_pos;
    MascotTriplePrep<BitShare>* bit_prep;
    SubProcessor<BitShare>* bit_proc;
    typename BitShare::MAC_Check* bit_MC;

public:
    Spdz2kPrep(SubProcessor<T>* proc, DataPositions& usage);
    ~Spdz2kPrep();

    void set_protocol(typename T::Protocol& protocol);

    void buffer_inverses() { throw division_by_zero(); }
    void buffer_bits();
    void buffer_dabits(ThreadQueues* queues);

#ifdef SPDZ2K_BIT
    void get_dabit(T& a, GC::TinySecret<T::s>& b);
#endif
};

#endif /* PROTOCOLS_SPDZ2KPREP_H_ */
