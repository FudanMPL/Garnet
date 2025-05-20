/*
 * RepRingOnlyEdabitPrep.h
 *
 */

#ifndef PROTOCOLS_REPRINGONLYEDABITPREP_H_
#define PROTOCOLS_REPRINGONLYEDABITPREP_H_

#include "ReplicatedPrep.h"

/**
 * edaBit generation for replicated secret sharing modulo a power of two
 */
template<class T>
class RepRingOnlyEdabitPrep : public virtual BufferPrep<T>
{
protected:
    void buffer_edabits(int n_bits, ThreadQueues*);

public:
    static void edabit_sacrifice_buckets(vector<edabit<T>>&, size_t, bool, int,
            SubProcessor<T>&, int, int, const void* = 0)
    {
        throw runtime_error("no need for sacrifice");
    }

    RepRingOnlyEdabitPrep(SubProcessor<T>*, DataPositions& usage) :
            BufferPrep<T>(usage)
    {
    }
};

#endif /* PROTOCOLS_REPRINGONLYEDABITPREP_H_ */
