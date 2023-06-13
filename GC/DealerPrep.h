/*
 * DealerPrep.h
 *
 */

#ifndef GC_DEALERPREP_H_
#define GC_DEALERPREP_H_

#include "Protocols/DealerPrep.h"
#include "Protocols/ProtocolSet.h"
#include "ShiftableTripleBuffer.h"
#include "SemiSecret.h"

namespace GC
{
class DealerPrep : public BufferPrep<DealerSecret>, ShiftableTripleBuffer<DealerSecret>
{
    Player* P;

public:
    DealerPrep(DataPositions& usage, int = -1) :
            BufferPrep<DealerSecret>(usage), P(0)
    {
    }

    void set_protocol(DealerSecret::Protocol& protocol)
    {
        P = &protocol.P;
    }

    void buffer_triples()
    {
        ProtocolSetup<DealerShare<BitVec>> setup(*P);
        ProtocolSet<DealerShare<BitVec>> set(*P, setup);
        for (int i = 0; i < OnlineOptions::singleton.batch_size; i++)
        {
            auto triple = set.preprocessing.get_triple(
                    DealerSecret::default_length);
            this->triples.push_back({{triple[0], triple[1], triple[2]}});
        }
    }

    void buffer_bits()
    {
        SeededPRNG G;
        if (P->my_num() != 0)
            for (int i = 0; i < OnlineOptions::singleton.batch_size; i++)
                this->bits.push_back(G.get_bit());
        else
            this->bits.resize(
                    this->bits.size() + OnlineOptions::singleton.batch_size);
    }

    void get(Dtype type, DealerSecret* data)
    {
        BufferPrep<DealerSecret>::get(type, data);
    }

    array<DealerSecret, 3> get_triple_no_count(int n_bits)
    {
        if (n_bits == -1)
            n_bits = DealerSecret::default_length;
        return ShiftableTripleBuffer<DealerSecret>::get_triple_no_count(n_bits);
    }
};

}

#endif /* GC_DEALERPREP_H_ */
