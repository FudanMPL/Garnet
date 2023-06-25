#ifndef _Random_Coins
#define _Random_Coins

/*  Randomness used to encrypt */

#include "FHE/FHE_Params.h"
#include "FHE/Rq_Element.h"
#include "FHE/AddableVector.h"

class FHE_PK;

#ifndef N_LIMBS_RAND
#define N_LIMBS_RAND 1
#endif

class Int_Random_Coins : public AddableMatrix<fixint<N_LIMBS_RAND>>
{
  typedef value_type::value_type T;

  const FHE_Params* params;
public:
  typedef value_type::value_type rand_type;

  Int_Random_Coins(const FHE_Params& params) : params(&params)
  { resize(3, params.phi_m()); }

  Int_Random_Coins(const FHE_PK& pk);

  void sample(PRNG& G)
  {
    (*this)[0].from(HalfGenerator<T>(G));
    for (int i = 1; i < 3; i++)
      (*this)[i].from(GaussianGenerator<T>(params->get_DG(), G));
  }
};

class Random_Coins
{
  typedef bigint T;

  Rq_Element uu,vv,ww;
  const FHE_Params *params;

  public:

  const FHE_Params& get_params() const { return *params; }

  Random_Coins(const FHE_Params& p) 
    : uu(p.FFTD(),evaluation,evaluation),
      vv(p.FFTD(),evaluation,evaluation),
      ww(p.FFTD(),polynomial,polynomial)
      { params=&p; }

  Random_Coins(const FHE_PK& pk);
  
  // Rely on default copy assignment/constructor

  const Rq_Element& u() const { return uu; }
  const Rq_Element& v() const { return vv; }
  const Rq_Element& w() const { return ww; }

  void assign(const Rq_Element& u,const Rq_Element& v,const Rq_Element& w)
    { uu=u; vv=v; ww=w; }

  template <class T>
  void assign(const vector<T>& u,const vector<T>& v,const vector<T>& w)
    {
      uu.from(u);
      vv.from(v);
      ww.from(w);
    }

  void assign(const Int_Random_Coins& rc)
    {
      uu.from(rc[0]);
      vv.from(rc[1]);
      ww.from(rc[2]);
    }

  /* Generate a standard distribution */
  void generate(PRNG& G)
    {
      uu.from(HalfGenerator<T>(G));
      vv.from(GaussianGenerator<T>(params->get_DG(), G));
      ww.from(GaussianGenerator<T>(params->get_DG(), G));
    }

  // Generate all from Uniform in range (-B,...B)
  void generateUniform(PRNG& G,const bigint& B1,const bigint& B2,const bigint& B3)
    {
      if (B1 == 0)
        uu.assign_zero();
      else
        uu.from(UniformGenerator<T>(G,numBits(B1)));
      vv.from(UniformGenerator<T>(G,numBits(B2)));
      ww.from(UniformGenerator<T>(G,numBits(B3)));
    }


  // ans,x and y must have same params otherwise error
  friend void add(Random_Coins& ans,
                  const Random_Coins& x,const Random_Coins& y);

  void pack(octetStream& o) const { uu.pack(o); vv.pack(o); ww.pack(o); }

  size_t report_size(ReportType type)
  { return uu.report_size(type) + vv.report_size(type) + ww.report_size(type); }
};


#endif
