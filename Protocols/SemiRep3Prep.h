/*
 * Rep2kPrep.h
 *
 */

#ifndef PROTOCOLS_SEMIREP3PREP_H_
#define PROTOCOLS_SEMIREP3PREP_H_

#include "ReplicatedPrep.h"

/**
 * Preprocessing for three-party replicated secret sharing modulo a power of two
 */
template<class T>
class SemiRep3Prep : public virtual SemiHonestRingPrep<T>,
        public virtual ReplicatedRingPrep<T>
{
    void buffer_dabits(ThreadQueues*);

public:
    SemiRep3Prep(SubProcessor<T>* proc, DataPositions& usage) :
            BufferPrep<T>(usage), BitPrep<T>(proc, usage),
			RingPrep<T>(proc, usage),
            SemiHonestRingPrep<T>(proc, usage), ReplicatedRingPrep<T>(proc, usage)
    {
    }

    void buffer_bits() { this->buffer_bits_without_check(); }

    void get_one_no_count(Dtype dtype, T& a)
    {
        if (dtype != DATA_BIT)
            throw not_implemented();

        typename T::bit_type b;
        this->get_dabit_no_count(a, b);
    }
};

#endif /* PROTOCOLS_SEMIREP3PREP_H_ */
