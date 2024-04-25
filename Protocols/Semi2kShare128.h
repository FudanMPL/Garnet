//
// Created by 林国鹏 on 2023/11/1.
//

#ifndef GARNET_SEMI2128SHARE128_H
#define GARNET_SEMI2128SHARE128_H

#include "SemiShare.h"
#include "Semi.h"
#include "OT/Rectangle.h"
#include "GC/SemiSecret.h"
#include "GC/square64.h"
#include "Processor/Instruction.h"

template<class T> class SemiPrep2k;

class Semi2kShare128 : public SemiShare<SignedZ2<128>>
{
  typedef SignedZ2<128> T;

public:
  typedef SemiMC<Semi2kShare128> MAC_Check;
  typedef DirectSemiMC<Semi2kShare128> Direct_MC;
  typedef SemiInput<Semi2kShare128> Input;
  typedef ::PrivateOutput<Semi2kShare128> PrivateOutput;
  typedef Semi<Semi2kShare128> Protocol;
  typedef SemiPrep2k<Semi2kShare128> LivePrep;

  typedef Semi2kShare128 prep_type;
  typedef SemiMultiplier<Semi2kShare128> Multiplier;
  typedef OTTripleGenerator<prep_type> TripleGenerator;
  typedef Z2kSquare<128> Rectangle;

  static const bool has_split = true;

  Semi2kShare128()
  {
  }
  template<class U>
  Semi2kShare128(const U& other) : SemiShare<SignedZ2<128>>(other)
  {
  }
//  Semi2kShare128(const T& other, int my_num, const T& alphai = {})
//  {
//    (void) alphai;
//    assign(other, my_num);
//  }

  template<class U>
  static void split(vector<U>& dest, const vector<int>& regs, int n_bits,
                    const Semi2kShare128* source, int n_inputs,
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


#endif //GARNET_SEMI2128SHARE128_H
