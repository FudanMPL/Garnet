/*
 * LimitedPrep.h
 *
 */

#ifndef PROTOCOLS_LIMITEDPREP_H_
#define PROTOCOLS_LIMITEDPREP_H_

#include "ReplicatedPrep.h"

template<class T>
class LimitedPrep : public BufferPrep<T>
{
    DataPositions usage;

    void buffer_triples() { throw ran_out(); }
    void buffer_squares() { throw ran_out(); }
    void buffer_inverses() { throw ran_out(); }
    void buffer_bits() { throw ran_out(); }

public:
    LimitedPrep();
    ~LimitedPrep();

    void set_protocol(typename T::Protocol& protocol);
};

#endif /* PROTOCOLS_LIMITEDPREP_H_ */
