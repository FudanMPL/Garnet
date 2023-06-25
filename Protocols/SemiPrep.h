/*
 * SemiPrep.h
 *
 */

#ifndef PROTOCOLS_SEMIPREP_H_
#define PROTOCOLS_SEMIPREP_H_

#include "MascotPrep.h"

template<class T> class HemiPrep;

/**
 * Semi-honest triple generation based on oblivious transfer
 */
template<class T>
class SemiPrep : public virtual OTPrep<T>, public virtual SemiHonestRingPrep<T>
{
    friend class HemiPrep<T>;

public:
    SemiPrep(SubProcessor<T>* proc, DataPositions& usage);

    void buffer_triples();

    void buffer_dabits(ThreadQueues* queues);

    void get_one_no_count(Dtype dtype, T& a);

    bool bits_from_dabits();
};

#endif /* PROTOCOLS_SEMIPREP_H_ */
