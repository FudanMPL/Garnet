/*
 * MaliciousRepPrep.h
 *
 */

#ifndef PROTOCOLS_MALICIOUSREPPREP_H_
#define PROTOCOLS_MALICIOUSREPPREP_H_

#include "Processor/Data_Files.h"
#include "ReplicatedPrep.h"
#include "MaliciousRepMC.h"

#include <array>

template<class T> class MalRepRingPrep;
template<int K, int S> class MalRepRingShare;
template<int K, int S> class PostSacriRepRingShare;

template<class T, class U>
void sacrifice(const vector<array<T, 5>>& check_triples, Player& P);

/**
 * Random bit generation from semi-honest protocol with sacrifice against square
 */
template<class T>
class MaliciousBitOnlyRepPrep : public virtual BufferPrep<T>
{
protected:
    DataPositions honest_usage;
    ReplicatedRingPrep<typename T::Honest> honest_prep;
    typename T::Honest::MAC_Check honest_mc;
    SubProcessor<typename T::Honest>* honest_proc;
    typename T::MAC_Check MC;

    void buffer_bits();

public:
    MaliciousBitOnlyRepPrep(SubProcessor<T>*, DataPositions& usage);
    ~MaliciousBitOnlyRepPrep();

    void set_protocol(typename T::Protocol& protocol);
    void init_honest(Player& P);
};

/**
 * Random triple/square from semi-honest protocol with sacrifice
 */
template<class T>
class MaliciousRepPrep : public MaliciousBitOnlyRepPrep<T>
{
    template<class U> friend class MalRepRingPrep;

protected:
    void buffer_triples();
    void buffer_squares();
    void buffer_inputs(int player);

public:
    MaliciousRepPrep(SubProcessor<T>* proc, DataPositions& usage);
    MaliciousRepPrep(DataPositions& usage, int = 0);
    template<class V>
    MaliciousRepPrep(DataPositions& usage, GC::ShareThread<V>&, int = 0);
};

template<class T>
class MaliciousRepPrepWithBits: public virtual MaliciousRepPrep<T>,
        public virtual MaliciousRingPrep<T>
{
    void buffer_squares()
    {
        MaliciousRepPrep<T>::buffer_squares();
    }

    void buffer_bits()
    {
        MaliciousRepPrep<T>::buffer_bits();
    }

public:
    MaliciousRepPrepWithBits(SubProcessor<T>* proc, DataPositions& usage);

    void set_protocol(typename T::Protocol& protocol)
    {
        MaliciousRingPrep<T>::set_protocol(protocol);
        MaliciousRepPrep<T>::set_protocol(protocol);
    }

};

#endif /* PROTOCOLS_MALICIOUSREPPREP_H_ */
