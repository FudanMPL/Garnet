/*
 * NoLivePrep.h
 *
 */

#ifndef PROCESSOR_NOLIVEPREP_H_
#define PROCESSOR_NOLIVEPREP_H_

#include "Tools/Exceptions.h"
#include "Protocols/ReplicatedPrep.h"

template<class T> class SubProcessor;
class DataPositions;

// preprocessing facility
template<class T>
class NoLivePrep : public BufferPrep<T>
{
public:
    // global setup for encryption keys if needed
    static void basic_setup(Player&)
    {
    }

    // destruct global setup
    static void teardown()
    {
    }

    NoLivePrep(SubProcessor<T>*, DataPositions& usage) :
            BufferPrep<T>(usage)
    {
    }

    NoLivePrep(DataPositions& usage, int = -1) :
            BufferPrep<T>(usage)
    {
    }

    // access to protocol instance if needed
    void set_protocol(typename T::Protocol&)
    {
    }

    // buffer batch of multiplication triples in this->triples
    void buffer_triples()
    {
        throw runtime_error("no triples");
    }

    // buffer batch of random bit shares in this->bits
    void buffer_bits()
    {
        throw runtime_error("no bits");
    }
};

#endif /* PROCESSOR_NOLIVEPREP_H_ */
