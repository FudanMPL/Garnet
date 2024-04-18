/*
 * Rep3Share2k.h
 *
 */

#ifndef PROTOCOLS_REP3SHARE2K_H_
#define PROTOCOLS_REP3SHARE2K_H_

#include "Rep3Share.h"
#include "ReplicatedInput.h"
#include "Math/Z2k.h"
#include "GC/square64.h"

template<class T> class SemiRep3Prep;

template<int K>
class Rep3Share2 : public Rep3Share<Z2<K>>
{
    typedef Z2<K> T;
    typedef Rep3Share2 This;

public:
    typedef Replicated<Rep3Share2> Protocol;
    typedef ReplicatedMC<Rep3Share2> MAC_Check;
    typedef MAC_Check Direct_MC;
    typedef ReplicatedInput<Rep3Share2> Input;
    typedef ReplicatedPO<This> PO;
    typedef SpecificPrivateOutput<This> PrivateOutput;
    typedef SemiRep3Prep<Rep3Share2> LivePrep;
    typedef Rep3Share2 Honest;
    typedef SignedZ2<K> clear;

    typedef GC::SemiHonestRepSecret bit_type;

    static const bool has_split = true;

    Rep3Share2()
    {
    }
    template<class U>
    Rep3Share2(const FixedVec<U, 2>& other)
    {
        FixedVec<T, 2>::operator=(other);
    }

    template<class U>
    static void split(vector<U>& dest, const vector<int>& regs, int n_bits,
            const Rep3Share2* source, int n_inputs,
            typename U::Protocol& protocol)
    {
        auto& P = protocol.P;
        int my_num = P.my_num();
        int unit = GC::Clear::N_BITS;
        for (int k = 0; k < DIV_CEIL(n_inputs, unit); k++)
        {
            int start = k * unit;
            int m = min(unit, n_inputs - start);

            switch (regs.size() / n_bits)
            {
            case 3:
                for (int l = 0; l < n_bits; l += unit)
                {
                    int base = l;
                    int n_left = min(n_bits - base, unit);
                    for (int i = base; i < base + n_left; i++)
                        dest.at(regs.at(3 * i + my_num) + k) = {};

                    for (int i = 0; i < 2; i++)
                    {
                        square64 square;

                        for (int j = 0; j < m; j++)
                            square.rows[j] = source[j + start][i].get_limb(
                                    l / unit);

                        square.transpose(m, n_left);

                        for (int j = 0; j < n_left; j++)
                        {
                            auto& dest_reg = dest.at(
                                    regs.at(3 * (base + j) + ((my_num + 2 - i) % 3))
                                            + k);
                            dest_reg[1 - i] = 0;
                            dest_reg[i] = square.rows[j];
                        }
                    }
                }
                break;
            case 2:
            {
                assert(n_bits <= 64);
                ReplicatedInput<U> input(P);
                input.reset_all(P);
                if (P.my_num() == 0)
                {
                    square64 square;
                    for (int j = 0; j < m; j++)
                        square.rows[j] = Integer(source[j + start].sum()).get();
                    square.transpose(m, n_bits);
                    for (int j = 0; j < n_bits; j++)
                        input.add_mine(square.rows[j], m);
                }
                else
                    for (int j = 0; j < n_bits; j++)
                        input.add_other(0);

                input.exchange();
                for (int j = 0; j < n_bits; j++)
                    dest.at(regs.at(2 * j) + k) = input.finalize(0, m);

                if (P.my_num() == 0)
                    for (int j = 0; j < n_bits; j++)
                        dest.at(regs.at(2 * j + 1) + k) = {};
                else
                {
                    square64 square;
                    for (int j = 0; j < m; j++)
                        square.rows[j] = Integer(source[j + start][P.my_num() - 1]).get();
                    square.transpose(m, n_bits);
                    for (int j = 0; j < n_bits; j++)
                    {
                        auto& dest_reg = dest.at(regs.at(2 * j + 1) + k);
                        dest_reg[P.my_num() - 1] = square.rows[j];
                        dest_reg[2 - P.my_num()] = {};
                    }
                }
                break;
            }
            default:
                throw runtime_error("number of split summands not implemented");
            }
        }
    }
};

#endif /* PROTOCOLS_REP3SHARE2K_H_ */
