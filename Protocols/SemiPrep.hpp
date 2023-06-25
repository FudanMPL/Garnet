/*
 * SemiPrep.cpp
 *
 */

#include "SemiPrep.h"

#include "ReplicatedPrep.hpp"
#include "MascotPrep.hpp"
#include "OT/NPartyTripleGenerator.hpp"

template<class T>
SemiPrep<T>::SemiPrep(SubProcessor<T>* proc, DataPositions& usage) :
        BufferPrep<T>(usage),
        BitPrep<T>(proc, usage),
        OTPrep<T>(proc, usage),
        RingPrep<T>(proc, usage),
        SemiHonestRingPrep<T>(proc, usage)
{
    this->params.set_passive();
}

template<class T>
void SemiPrep<T>::buffer_triples()
{
    assert(this->triple_generator);
    this->triple_generator->generatePlainTriples();
    for (auto& x : this->triple_generator->plainTriples)
    {
        this->triples.push_back({{x[0], x[1], x[2]}});
    }
    this->triple_generator->unlock();
}

template<class T>
bool SemiPrep<T>::bits_from_dabits()
{
    assert(this->proc);
    return this->proc->P.num_players() == 2 and not T::clear::characteristic_two;
}

template<class T>
void SemiPrep<T>::buffer_dabits(ThreadQueues* queues)
{
    if (bits_from_dabits())
    {
        assert(this->triple_generator);
        this->triple_generator->generatePlainBits();
        for (auto& x : this->triple_generator->plainBits)
            this->dabits.push_back({x.first, x.second});
    }
    else
        SemiHonestRingPrep<T>::buffer_dabits(queues);
}

template<class T>
void SemiPrep<T>::get_one_no_count(Dtype dtype, T& a)
{
    if (bits_from_dabits())
    {
        if (dtype != DATA_BIT)
            throw not_implemented();

        typename T::bit_type b;
        this->get_dabit_no_count(a, b);
    }
    else
        SemiHonestRingPrep<T>::get_one_no_count(dtype, a);
}
