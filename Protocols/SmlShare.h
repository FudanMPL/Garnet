/*
 * SmlShare.h
 *
 */

#ifndef PROTOCOLS_SmlShare_H_
#define PROTOCOLS_SmlShare_H_

#include "SemiShare.h"
#include "SecureML.h"
#include "OT/Rectangle.h"
#include "GC/SemiSecret.h"
#include "GC/square64.h"
#include "Processor/Instruction.h"

template<class T> class SemiPrep2k;
template<class T> class SemiPrep;
template<class T> class Sml;
template<class T> class SmlMatrixPrep;

template <int K>
class SmlShare : public SemiShare<SignedZ2<K>>
{
    typedef SmlShare This;
    typedef SemiShare<SignedZ2<K>> super;
public:
    typedef SignedZ2<K> Dtype;
    // opening facility
    typedef SemiMC<SmlShare> MAC_Check;
    typedef DirectSemiMC<SmlShare> Direct_MC;
    
    // private input facility
    typedef SemiInput<SmlShare> Input;
    
    // default private output facility (using input tuples)
    typedef ::PrivateOutput<SmlShare> PrivateOutput;
    
    // opening facility
    typedef SecureML<SmlShare> Protocol;
    
    // preprocessing facility
    // typedef SemiPrep<SmlShare> LivePrep;
    typedef SemiPrep2k<SmlShare> LivePrep;

    typedef SmlShare prep_type;
    typedef SemiMultiplier<SmlShare> Multiplier;
    typedef OTTripleGenerator<prep_type> TripleGenerator;
    typedef Z2kSquare<K> Rectangle;

    typedef SmlMatrixPrep<SmlShare> MatrixPrep;
    // static const bool has_split = true;

    typedef Semi<This> BasicProtocol;

    // static const bool needs_ot = false;
    static const bool local_mul = true;
    // static true_type triple_matmul;
    
    SmlShare()
    {
    }
    template<class U>
    SmlShare(const U& other) : SemiShare<Dtype>(other)
    {
    }
    // SmlShare(const T& other, int my_num, const T& alphai = {})
    // {
    //     (void) alphai;
    //     assign(other, my_num);
    // }

    template<class U>
    static void split(vector<U>& dest, const vector<int>& regs, int n_bits,
            const SmlShare* source, int n_inputs,
            typename U::Protocol& protocol)
    {
        auto& P = protocol.P;
        int my_num = P.my_num();
        int unit = GC::Clear::N_BITS;
        for (int k = 0; k < DIV_CEIL(n_inputs, unit); k++)
        {
            int start = k * unit;
            int m = min(unit, n_inputs - start);
            int n = regs.size() / n_bits;
            if (P.num_players() != n)
                throw runtime_error(
                        to_string(n) + "-way split not working with "
                                + to_string(P.num_players()) + " parties");

            for (int l = 0; l < n_bits; l += unit)
            {
                int base = l;
                int n_left = min(n_bits - base, unit);
                for (int i = base; i < base + n_left; i++)
                    for (int j = 0; j < n; j++)
                        dest.at(regs.at(n * i + j) + k) = {};

                square64 square;

                for (int j = 0; j < m; j++)
                    square.rows[j] = source[j + start].get_limb(l / unit);

                square.transpose(m, n_left);

                for (int j = 0; j < n_left; j++)
                {
                    auto& dest_reg = dest.at(
                            regs.at(n * (base + j) + my_num) + k);
                    dest_reg = square.rows[j];
                }
            }
        }
    }
};

#endif /* PROTOCOLS_SmlShare_H_ */