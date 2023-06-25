/*
 * MalRepRingPrep.h
 *
 */

#ifndef PROTOCOLS_MALREPRINGPREP_H_
#define PROTOCOLS_MALREPRINGPREP_H_

#include "Protocols/ReplicatedPrep.h"

/**
 * Generate random triples with malicious security modulo a power two,
 * either via larger modulo or shuffling
 */
template<class T>
class MalRepRingPrep : public virtual BufferPrep<T>
{
public:
    MalRepRingPrep(SubProcessor<T>* proc, DataPositions& usage);

    void set_protocol(typename T::Protocol&)
    {
    }

    void buffer_triples();
    void simple_buffer_triples();
    void shuffle_buffer_triples();

    void buffer_squares();

    void buffer_inputs(int player);
};

/**
 * Generate random bits from squares modulo a power of two
 */
template<class T>
class RingOnlyBitsFromSquaresPrep : public virtual BufferPrep<T>
{
public:
    RingOnlyBitsFromSquaresPrep(SubProcessor<T>* proc, DataPositions& usage);

    void buffer_bits();
};

template<class T>
class SimplerMalRepRingPrep : public virtual MalRepRingPrep<T>,
        public virtual RingOnlyBitsFromSquaresPrep<T>
{
public:
    SimplerMalRepRingPrep(SubProcessor<T>* proc, DataPositions& usage);

    void buffer_triples()
    {
        MalRepRingPrep<T>::buffer_triples();
    }

    void buffer_squares()
    {
        MalRepRingPrep<T>::buffer_squares();
    }

    void buffer_bits()
    {
        RingOnlyBitsFromSquaresPrep<T>::buffer_bits();
    }

    void get_dabit_no_count(T& a, typename T::bit_type& b)
    {
        this->get_one_no_count(DATA_BIT, a);
        b = a & 1;
    }
};

template<class T>
class MalRepRingPrepWithBits: public virtual MaliciousRingPrep<T>,
        public virtual SimplerMalRepRingPrep<T>
{
public:
    MalRepRingPrepWithBits(SubProcessor<T>* proc, DataPositions& usage);

    void set_protocol(typename T::Protocol& protocol)
    {
        MaliciousRingPrep<T>::set_protocol(protocol);
    }

    void buffer_squares()
    {
        MalRepRingPrep<T>::buffer_squares();
    }

    void buffer_bits()
    {
        RingOnlyBitsFromSquaresPrep<T>::buffer_bits();
    };
};

#endif /* PROTOCOLS_MALREPRINGPREP_H_ */
