//
// Created by 林国鹏 on 2023/7/19.
//

#ifndef GARNET_MALREPRINGSHARE128_H
#define GARNET_MALREPRINGSHARE128_H

#include "Math/Z2k.h"
#include "Protocols/MaliciousRep3Share.h"
#include "config.h"
#include "Protocols/MalRepRingShare.h"

template<class T> class MalRepRingPrepWithBits;
template<class T> class MalRepRingPrep;


class MalRepRingShare128 : public MaliciousRep3Share<SignedZ2<128>>
{
  typedef SignedZ2<128> T;
  typedef MaliciousRep3Share<T> super;
  typedef MalRepRingShare128 This;

public:
  const static int BIT_LENGTH = 128;
  const static int SECURITY = DEFAULT_SECURITY;

  typedef Beaver<MalRepRingShare128> Protocol;
  typedef HashMaliciousRepMC<MalRepRingShare128> MAC_Check;
  typedef MAC_Check Direct_MC;
  typedef ReplicatedInput<MalRepRingShare128> Input;
  typedef ReplicatedPO<This> PO;
  typedef SpecificPrivateOutput<This> PrivateOutput;
  typedef MalRepRingPrepWithBits<MalRepRingShare128> LivePrep;
  typedef MaliciousRep3Share<Z2<128 + DEFAULT_SECURITY>> prep_type;
  typedef MalRepRingShare<128 + 2, DEFAULT_SECURITY> SquareToBitShare;

  static string type_short()
  {
    return "RR";
  }

  MalRepRingShare128()
  {
  }
  MalRepRingShare128(const T& other, int my_num, T alphai = {}) :
          super(other, my_num, alphai)
  {
  }
  template<class U>
  MalRepRingShare128(const U& other) : super(other)
  {
  }
};

#endif //GARNET_MALREPRINGSHARE128_H
