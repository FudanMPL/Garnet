/*
 * Rep4Share.h
 *
 */

#ifndef PROTOCOLS_REP4SHARE2K_H_
#define PROTOCOLS_REP4SHARE2K_H_

#include "Rep4Share.h"
#include "Processor/DummyProtocol.h"
#include "GC/square64.h"

template<class T> class Rep4MC;
template<class T> class Rep4Input;
template<class T> class Rep4RingOnlyPrep;

template<int K>
class Rep4Share2 : public Rep4Share<Z2<K>>
{
    typedef Rep4Share2 This;
    typedef Rep4Share<Z2<K>> super;

public:
    typedef SignedZ2<K> clear;
    typedef Rep4Share<Z2<K + 2>> SquareToBitShare;

    typedef Rep4<This> Protocol;
    typedef Rep4MC<This> MAC_Check;
    typedef MAC_Check Direct_MC;
    typedef Rep4Input<This> Input;
    typedef ::PrivateOutput<This> PrivateOutput;
    typedef Rep4RingOnlyPrep<This> LivePrep;

    static const bool has_trunc_pr = true;
    static const bool has_split = true;

    Rep4Share2()
    {
    }
    template<class T>
    Rep4Share2(const FixedVec<T, 3>& other) : super(other)
    {
    }

    template<class U>
    static void split(vector<U>& dest, const vector<int>& regs, int n_bits,
            const Rep4Share2* source, int n_inputs, Rep4<U>& protocol)
    {
        int n_split = regs.size() / n_bits;
        if (n_split == 2)
            protocol.split(dest, regs, n_bits, source, n_inputs);
        else
            split(dest, regs, n_bits, source, n_inputs, protocol.P);
    }

    template<class U>
    static void split(vector<U>& dest, const vector<int>& regs, int n_bits,
            const Rep4Share2* source, int n_inputs, Player& P)
    {
        int my_num = P.my_num();
        assert(n_bits <= 64);
        assert(regs.size() / n_bits == 4);
        int unit = GC::Clear::N_BITS;
        for (int k = 0; k < DIV_CEIL(n_inputs, unit); k++)
        {
            int start = k * unit;
            int m = min(unit, n_inputs - start);

            for (int i = 0; i < n_bits; i++)
                dest.at(regs.at(4 * i + my_num) + k) = {};

            for (int i = 0; i < 3; i++)
            {
                square64 square;

                for (int j = 0; j < m; j++)
                    square.rows[j] = Integer(source[j + start][i]).get();

                square.transpose(m, n_bits);

                for (int j = 0; j < n_bits; j++)
                {
                    auto &dest_reg = dest.at(
                            regs.at(4 * j + ((my_num + i + 1) % 4)) + k);
                    dest_reg = {};
                    dest_reg[i] = square.rows[j];
                }
            }
        }
    }
};

#endif /* PROTOCOLS_REP4SHARE2K_H_ */
