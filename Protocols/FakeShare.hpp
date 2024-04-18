/*
 * FakeShare.cpp
 *
 */

#include "FakeShare.h"
#include "Math/Z2k.h"
#include "GC/square64.h"

template<class T>
void FakeShare<T>::split(vector<bit_type>& dest,
        const vector<int>& regs, int n_bits, const This* source, int n_inputs,
        GC::FakeSecret::Protocol&)
{
    assert(n_bits <= 64);
    int unit = GC::Clear::N_BITS;
    for (int k = 0; k < DIV_CEIL(n_inputs, unit); k++)
    {
        int start = k * unit;
        int m = min(unit, n_inputs - start);

        switch (regs.size() / n_bits)
        {
        case 3:
        {
            for (int i = 0; i < n_bits; i++)
                for (int j = 1; j < 3; j++)
                    dest.at(regs.at(3 * i + j) + k) = {};

            square64 square;

            for (int j = 0; j < m; j++)
            {
                square.rows[j] = (source[j + start]).get_limb(0);
            }

            square.transpose(m, n_bits);

            for (int j = 0; j < n_bits; j++)
            {
                auto& dest_reg = dest.at(regs.at(3 * j) + k);
                dest_reg = square.rows[j];
            }
            break;
        }
        default:
            not_implemented();
        }
    }
}
