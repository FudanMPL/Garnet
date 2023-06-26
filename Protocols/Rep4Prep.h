/*
 * Rep4Prep.h
 *
 */

#ifndef PROTOCOLS_REP4PREP_H_
#define PROTOCOLS_REP4PREP_H_

#include "MaliciousRingPrep.hpp"
#include "MalRepRingPrep.h"
#include "RepRingOnlyEdabitPrep.h"

template<class T>
class Rep4RingPrep : public MaliciousRingPrep<T>
{
    void buffer_triples();
    void buffer_squares();
    void buffer_bits();
    void buffer_inputs(int player);

public:
    Rep4RingPrep(SubProcessor<T>* proc, DataPositions& usage);
};

template<class T>
class Rep4RingOnlyPrep : public virtual Rep4RingPrep<T>,
        public virtual RepRingOnlyEdabitPrep<T>
{
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

    Rep4RingOnlyPrep(SubProcessor<T>* proc, DataPositions& usage);

    void get_dabit_no_count(T& a, typename T::bit_type& b)
    {
        this->get_one_no_count(DATA_BIT, a);
        b = a & 1;
    }
};

#endif /* PROTOCOLS_REP4PREP_H_ */
