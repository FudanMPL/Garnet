/*
 * SpdzWiseRingPrep.h
 *
 */

#ifndef PROTOCOLS_SPDZWISERINGPREP_H_
#define PROTOCOLS_SPDZWISERINGPREP_H_

#include "SpdzWisePrep.h"
#include "RepRingOnlyEdabitPrep.h"

/**
 * Preprocessing for three-party replicated secret sharing protocol with MAC
 * modulo a power of two
 */
template<class T>
class SpdzWiseRingPrep : public virtual SpdzWisePrep<T>,
        public virtual RepRingOnlyEdabitPrep<T>
{
    void buffer_bits();

    void buffer_edabits(int n_bits, ThreadQueues* queues)
    {
        RepRingOnlyEdabitPrep<T>::buffer_edabits(n_bits, queues);
    }

    void buffer_edabits(bool strict, int n_bits, ThreadQueues* queues)
    {
        BufferPrep<T>::buffer_edabits(strict, n_bits, queues);
    }

    void buffer_sedabits(int n_bits, ThreadQueues*)
    {
        this->buffer_sedabits_from_edabits(n_bits);
    }

public:
    static void edabit_sacrifice_buckets(vector<edabit<T>>&, size_t, bool, int,
            SubProcessor<T>&, int, int, const void* = 0)
    {
        throw runtime_error("no need for sacrifice");
    }

    SpdzWiseRingPrep(SubProcessor<T>* proc, DataPositions& usage) :
        BufferPrep<T>(usage),
        BitPrep<T>(proc, usage), RingPrep<T>(proc, usage),
        MaliciousDabitOnlyPrep<T>(proc, usage),
        SpdzWisePrep<T>(proc, usage), RepRingOnlyEdabitPrep<T>(proc, usage)
    {
    }

    void get_dabit_no_count(T& a, typename T::bit_type& b)
    {
        this->get_one_no_count(DATA_BIT, a);
        b = a.get_share() & 1;
    }
};

#endif /* PROTOCOLS_SPDZWISERINGPREP_H_ */
