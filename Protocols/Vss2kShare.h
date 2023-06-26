/*
 * Vss2kShare.h
 *
 */

#ifndef PROTOCOLS_Vss2KSHARE_H_
#define PROTOCOLS_Vss2KSHARE_H_

#include "Vss.h"
#include "SecureML.h"
#include "VssMC.h"
#include "SemiShare.h"
#include "OT/Rectangle.h"
#include "GC/SemiSecret.h"
#include "VssInput.h"
#include "VssPrep.h"
#include "GC/square64.h"
#include "Processor/Instruction.h"


template <int K>
class Vss2kShare : public SemiShare<SignedZ2<K>>
{
    typedef Vss2kShare This;
    typedef SemiShare<SignedZ2<K>> super;
public:
    typedef SignedZ2<K> Dtype;
    typedef VssMC<Vss2kShare> MAC_Check;
    typedef DirectVssMC<Vss2kShare> Direct_MC;
    typedef VssInput<Vss2kShare> Input;
    typedef ::PrivateOutput<Vss2kShare> PrivateOutput;
    typedef Vss<Vss2kShare> Protocol;

    typedef VssPrep<Vss2kShare> LivePrep;

    typedef Vss2kShare prep_type;
    typedef SemiMultiplier<Vss2kShare> Multiplier;
    typedef OTTripleGenerator<prep_type> TripleGenerator;
    typedef Z2kSquare<K> Rectangle;

    typedef VssMatrixPrep<Vss2kShare> MatrixPrep;

    typedef Semi<This> BasicProtocol;

    // static const bool has_split = true;
    static const bool local_mul = true;

    Vss2kShare()
    {
    }
    template <class U>
    Vss2kShare(const U &other) : SemiShare<SignedZ2<K>>(other)
    {
    }
    Vss2kShare(const Dtype &other, int my_num, const Dtype &alphai = {})
    {
        (void)alphai;
        assign(other, my_num);
    }

    template <class U>
    static void split(vector<U> &dest, const vector<int> &regs, int n_bits,
                      const Vss2kShare *source, int n_inputs,
                      typename U::Protocol &protocol)
    {
        auto &P = protocol.P;
        int my_num = P.my_num();
        int unit = GC::Clear::N_BITS;
        for (int k = 0; k < DIV_CEIL(n_inputs, unit); k++)
        {
            int start = k * unit;
            int m = min(unit, n_inputs - start);
            int n = regs.size() / n_bits;
            if (P.num_players() != n)
                throw runtime_error(
                    to_string(n) + "-way split not working with " + to_string(P.num_players()) + " parties");

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
                    auto &dest_reg = dest.at(
                        regs.at(n * (base + j) + my_num) + k);
                    dest_reg = square.rows[j];
                }
            }
        }
    }
};

#endif /* PROTOCOLS_Vss2KSHARE_H_ */
