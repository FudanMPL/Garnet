/*
 * FakePrep.h
 *
 */

#ifndef PROTOCOLS_FAKEPREP_H_
#define PROTOCOLS_FAKEPREP_H_

#include "ReplicatedPrep.h"

template<class T>
class FakePrep : public BufferPrep<T>
{
    SeededPRNG G;

public:
    FakePrep(SubProcessor<T>*, DataPositions& usage) :
            BufferPrep<T>(usage)
    {
    }

    FakePrep(DataPositions& usage, GC::ShareThread<T>&) :
            BufferPrep<T>(usage)
    {
    }

    FakePrep(DataPositions& usage, int = 0) :
            BufferPrep<T>(usage)
    {
    }

    void set_protocol(typename T::Protocol&)
    {
    }

    void buffer_triples()
    {
        for (int i = 0; i < 1000; i++)
        {
            auto a = G.get<T>();
            auto b = G.get<T>();
            this->triples.push_back({{a, b, a * b}});
        }
    }

    void buffer_squares()
    {
        for (int i = 0; i < 1000; i++)
        {
            auto a = G.get<T>();
            this->squares.push_back({{a, a * a}});
        }
    }

    void buffer_inverses()
    {
        for (int i = 0; i < 1000; i++)
        {
            auto a = G.get<T>();
            this->inverses.push_back({{a, a.invert()}});
        }
    }

    void buffer_bits()
    {
        for (int i = 0; i < 1000; i++)
        {
            this->bits.push_back(G.get_bit());
        }
    }

    void buffer_inputs(int)
    {
        this->inputs.resize(1);
        for (int i = 0; i < 1000; i++)
        {
            auto r = G.get<T>();
            this->inputs[0].push_back({r, r});
        }
    }

    void get_dabit_no_count(T& a, typename T::bit_type& b)
    {
        auto bit = G.get_bit();
        a = bit;
        b = bit;
    }

    void get_one_no_count(Dtype dtype, T& a)
    {
        assert(dtype == DATA_BIT);
        a = G.get_uchar() & 1;
    }
};

#endif /* PROTOCOLS_FAKEPREP_H_ */
