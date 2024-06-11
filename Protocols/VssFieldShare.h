/*
 * VssFieldShare.h
 *
 */

#ifndef PROTOCOLS_VSSFIELDSHARE_H_
#define PROTOCOLS_VSSFIELDSHARE_H_

#include "Vss.h"
#include "SecureML.h"
#include "VssFieldMC.h"
#include "SemiShare.h"
#include "GC/square64.h"
#include "OT/Rectangle.h"
#include "GC/SemiSecret.h"
#include "VssFieldInput.h"
#include "VssFieldPrep.h"
#include "Processor/Instruction.h"

template <class T>
class VssFieldShare : public SemiShare<T>
{
    typedef VssFieldShare This;
    typedef SemiShare<T> super;

public:
    typedef typename super::clear Dtype;
    typedef VssFieldInput<VssFieldShare> Input;
    typedef VssFieldMC<VssFieldShare> MAC_Check;
    typedef DirectVssFieldMC<VssFieldShare> Direct_MC;
    typedef ::PrivateOutput<VssFieldShare> PrivateOutput;
    typedef Vss<VssFieldShare> Protocol;

    typedef VssFieldPrep<This> LivePrep;

    typedef VssFieldShare<typename T::next> prep_type;
    typedef SemiMultiplier<VssFieldShare> Multiplier;
    typedef OTTripleGenerator<prep_type> TripleGenerator;
    typedef typename T::Square Rectangle;
    // typedef Z2kSquare<K> Rectangle;

    typedef VssMatrixPrep<VssFieldShare> MatrixPrep;

    typedef Semi<This> BasicProtocol;

    // static const bool has_split = true;
    static const bool local_mul = true;

    VssFieldShare()
    {
    }
    template <class U>
    VssFieldShare(const U &other) : SemiShare<T>(other)
    {
    }
    VssFieldShare(const Dtype &other, int my_num, const Dtype &alphai = {})
    {
        (void)alphai;
        assign(other, my_num);
    }

    template <class U>
    static void split(vector<U> &dest, const vector<int> &regs, int n_bits,
                      const VssFieldShare *source, int n_inputs,
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
                    square.rows[j] = source[j + start].get().get_limb(l / unit);

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

#endif /* PROTOCOLS_VSSFIELDSHARE_H_ */
