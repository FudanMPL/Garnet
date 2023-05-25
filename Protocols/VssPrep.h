/*
 * VssPrep.h
 *
 */

#ifndef PROTOCOLS_VSSPREP_H_
#define PROTOCOLS_VSSPREP_H_

#include "MascotPrep.h"

template <class T>
class VssPrep : public virtual OTPrep<T>, public virtual SemiHonestRingPrep<T>
{

public:
    VssPrep(SubProcessor<T> *proc, DataPositions &usage) : BufferPrep<T>(usage),
                                                            BitPrep<T>(proc, usage),
                                                            OTPrep<T>(proc, usage),
                                                            RingPrep<T>(proc, usage),
                                                            SemiHonestRingPrep<T>(proc, usage)
    {
        this->params.set_passive();
    }
    void buffer_triples()
    {
        assert(this->triple_generator);
        this->triple_generator->generatePlainTriples(true);
        for (auto &x : this->triple_generator->plainTriples)
        {
            this->triples.push_back({{x[0], x[1], x[2]}});
        }
        this->triple_generator->unlock();
    }
};

#endif /* PROTOCOLS_VSSPREP_H_ */