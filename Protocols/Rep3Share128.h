
#ifndef GARNET_REP3SHARE128_H
#define GARNET_REP3SHARE128_H

/*
 * Rep3Share128k.h
 *
 */


#include "Rep3Share.h"
#include "ReplicatedInput.h"
#include "Math/Z2k.h"
#include "GC/square64.h"


class Rep3Share128 : public Rep3Share<Z2<128>>
{
  typedef Z2<128> T;
  typedef Rep3Share128 This;

  public:
  typedef Replicated<Rep3Share128> Protocol;
  typedef ReplicatedMC<Rep3Share128> MAC_Check;
  typedef MAC_Check Direct_MC;
  typedef ReplicatedInput<Rep3Share128> Input;
  typedef ReplicatedPO<This> PO;
  typedef SpecificPrivateOutput<This> PrivateOutput;
  typedef SemiRep3Prep<Rep3Share128> LivePrep;
  typedef SignedZ2<128> clear;

  typedef GC::SemiHonestRepSecret bit_type;

  static const bool has_split = true;

  Rep3Share128()
  {
  }

//Rep3Share128& operator=(const FixedVec<Z2<128>, 2> &other)  {
//  this->v[0] = other.v[0];
//  this->v[1] = other.v[1];
//  return *this;
//}

  template<class U>
  Rep3Share128(const FixedVec<U, 2>& other)
  {
    FixedVec<T, 2>::operator=(other);

  }



};


#endif //GARNET_REP3SHARE128_H
