
#include "FHE/Plaintext.h"
#include "FHE/Ring_Element.h"
#include "FHE/PPData.h"
#include "FHE/P2Data.h"
#include "FHE/Rq_Element.h"
#include "FHE_Keys.h"
#include "FHE/AddableVector.hpp"
#include "Math/Z2k.hpp"
#include "Math/modp.hpp"



template<class T, class FD, class S>
Plaintext<T, FD, S>::Plaintext(const FHE_Params& params) :
    Plaintext(params.get_plaintext_field_data<FD>(), Both)
{
}


template<class T, class FD, class S>
unsigned int Plaintext<T, FD, S>::num_slots() const
{
  return (*Field_Data).phi_m();
}

template<class T, class FD, class S>
int Plaintext<T, FD, S>::degree() const
{
  return (*Field_Data).phi_m();
}


template<>
unsigned int Plaintext<gf2n_short,P2Data,int>::num_slots() const
{
  return (*Field_Data).num_slots();
}

template<>
int Plaintext<gf2n_short,P2Data,int>::degree() const
{
  return (*Field_Data).degree();
}


template<>
void Plaintext<gfp, FFT_Data, bigint>::from(const Generator<bigint>& source) const
{
  b.resize(degree());
  for (auto& x : b)
    {
      source.get(bigint::tmp);
      x = bigint::tmp;
    }
}


template<>
void Plaintext<gfp,FFT_Data,bigint>::from_poly() const
{
  if (type!=Polynomial) { return; }

  Ring_Element e(*Field_Data,polynomial);
  e.from(b);
  e.change_rep(evaluation);
  a.resize(num_slots());
  for (unsigned int i=0; i<a.size(); i++)
    a[i] = gfp(e.get_element(i), e.get_FFTD().get_prD());
  type=Both;
}


template<>
void Plaintext<gfp,FFT_Data,bigint>::to_poly() const
{
  if (type!=Evaluation) { return; }

  Ring_Element e(*Field_Data,evaluation);
  for (unsigned int i=0; i<a.size(); i++)
    { e.set_element(i,a[i].get()); }
  e.change_rep(polynomial);
  from(e.get_iterator());
  type=Both;
}


template<>
void Plaintext<gfp,PPData,bigint>::from_poly() const
{
  if (type!=Polynomial) { return; }
  vector<modp> aa((*Field_Data).phi_m());
  for (unsigned int i=0; i<aa.size(); i++)
    { to_modp(aa[i], bigint::tmp = b[i], (*Field_Data).prData); }
  (*Field_Data).to_eval(aa);
  a.resize(num_slots());
  for (unsigned int i=0; i<aa.size(); i++)
    a[i] = {aa[i], Field_Data->get_prD()};
  type=Both;
}


template<>
void Plaintext<gfp,PPData,bigint>::to_poly() const
{
  if (type!=Evaluation) { return; }
  cout << "This is VERY inefficient to convert a plaintext to poly representation" << endl;
  vector<modp> bb((*Field_Data).phi_m());
  for (unsigned int i=0; i<bb.size(); i++)
    { bb[i]=a[i].get(); }
  (*Field_Data).from_eval(bb);
  for (unsigned int i=0; i<bb.size(); i++)
    {
      to_bigint(bigint::tmp,bb[i],(*Field_Data).prData);
      b[i] = bigint::tmp;
    }
  type=Both;
}



template<>
void Plaintext<gf2n_short,P2Data,int>::from_poly() const
{ 
  if (type!=Polynomial) { return; }
  a.resize(num_slots());
  (*Field_Data).backward(a,b); 
  type=Both;
}


template<>
void Plaintext<gf2n_short,P2Data,int>::to_poly() const
{ 
  if (type!=Evaluation) { return; }
  (*Field_Data).forward(b,a); 
  type=Both;
}



template<class T, class FD, class S>
void Plaintext<T, FD, S>::allocate(PT_Type type) const
{
  if (type != Evaluation)
    b.resize(degree());
  if (type != Polynomial)
    a.resize(num_slots());
  this->type = type;
}


template<class T, class FD, class S>
void Plaintext<T, FD, S>::allocate_slots(const bigint& value)
{
  b.resize(degree());
  for (auto& x : b)
    x.allocate_slots(value);
}

template<>
void Plaintext<gf2n_short, P2Data, int>::allocate_slots(const bigint& value)
{
  // nothing to allocate for int
  (void)value;
}

template<>
int Plaintext<gfp, FFT_Data, bigint>::get_min_alloc()
{
  int res = 1 << 30;
  for (auto& x : b)
    res = min(res, x.get_min_alloc());
  return res;
}


void signed_mod(bigint& x, const bigint& mod, const bigint& half_mod, const bigint& dest_mod)
{
  if (x > half_mod)
    x -= mod;
  x %= dest_mod;
  if (x < 0)
    x += dest_mod;
}

template<class T, class FD, class S>
void Plaintext<T, FD, S>::set_poly_mod(const Generator<bigint>& generator,const bigint& mod)
{
  allocate(Polynomial);
  bigint half_mod = mod / 2;
  for (unsigned int i=0; i<b.size(); i++)
    {
      generator.get(bigint::tmp);
      signed_mod(bigint::tmp, mod, half_mod, Field_Data->get_prime());
      b[i] = bigint::tmp;
    }
}




template<>
void Plaintext<gf2n_short,P2Data,int>::set_poly_mod(const vector<bigint>& vv,const bigint& mod)
{
  vector<P2Data::poly_type> pol(vv.size());
  bigint te;
  for (unsigned int i=0; i<vv.size(); i++)
    { if (vv[i]>mod/2) { te=vv[i]-mod; }
      else             { te=vv[i];     }
      pol[i]=isOdd(te);
    }
  set_poly(pol);
}


template<>
void Plaintext<gf2n_short,P2Data,int>::set_poly_mod(const Generator<bigint>& generator,const bigint& mod)
{
  allocate(Polynomial);
  bigint half_mod = mod / 2;
  bigint te;
  for (unsigned int i=0; i<b.size(); i++)
    {
      generator.get(te);
      if (te > half_mod)
        te -= mod;
      b[i]=isOdd(te);
    }
}


template<class T>
void rand_poly(vector<T>& b,PRNG& G,const bigint& pr,bool positive=true)
{
  for (unsigned int i=0; i<b.size(); i++)
    {
      b[i].randomBnd(G, pr, positive);
    }
}

template<class T,class FD,class S>
void Plaintext<T,FD,S>::randomize(PRNG& G,condition cond)
{
  switch (cond)
    { case Full:
        rand_poly(b,G,(*Field_Data).get_prime());
        type=Polynomial;
        break;
      case Diagonal:
        a.resize(num_slots());
        a[0].randomize(G);
        for (unsigned int i=1; i<a.size(); i++)
           { a[i]=a[0]; }
        type=Evaluation;
        break;
      default:
        // Gen a plaintext with 0/1 in each slot
        a.resize(num_slots());
	for (unsigned int i=0; i<a.size(); i++)
           {
             if (G.get_bit())
		{ a[i].assign_one(); }
             else
	        { a[i].assign_zero(); }
           }
        type=Evaluation;
        break;
    }
    
}


template<class T,class FD,class S>
void Plaintext<T,FD,S>::randomize(PRNG& G, int n_bits, bool Diag, PT_Type t)
{
  allocate(t);
  switch(t)
  {
    case Polynomial:
      if (Diag)
        {
          assign_zero(t);
          b[0].generateUniform(G, n_bits, false);
        }
      else
        for (size_t i = 0; i < num_slots(); i++)
          b[i].generateUniform(G, n_bits, false);
      break;
    default:
      throw not_implemented();
  }
}


template<class T,class FD,class S>
void Plaintext<T,FD,S>::assign_zero(PT_Type t)
{ 
  type=t;
  allocate();
  if (type!=Polynomial)
    {
      a.resize(num_slots());
      for (unsigned int i=0; i<a.size(); i++)
        { a[i].assign_zero(); }
    }
  if (type!=Evaluation)
    { for (unsigned int i=0; i<b.size(); i++)
        { b[i]=0; }
    }

}

template<class T,class FD,class S>
void Plaintext<T,FD,S>::assign_one(PT_Type t)
{ 
  type=t;
  allocate();
  if (type!=Polynomial)
    {
      a.resize(num_slots());
      for (unsigned int i=0; i<a.size(); i++)
        { a[i].assign_one(); }
    }
  if (type!=Evaluation)
    { for (unsigned int i=1; i<b.size(); i++)
        { b[i]=0; }
      b[0]=1;
    }
}

template<class T,class FD,class S>
void Plaintext<T,FD,S>::assign_constant(T constant, PT_Type t)
{
  allocate(Evaluation);
  for (auto& x : a)
      x = constant;
  if (t != Evaluation)
      to_poly();
}

template<class T,class FD,class S>
Plaintext<T,FD,S>& Plaintext<T,FD,S>::operator+=(
    const Plaintext& y)
{
  if (Field_Data!=y.Field_Data)  { throw field_mismatch(); }

  to_poly();
  y.to_poly();

  if (b.size() != y.b.size())
    throw length_error("size mismatch");

  add(*this, *this, y);
  return *this;
}


template<>
void add(Plaintext<gfp,FFT_Data,bigint>& z,const Plaintext<gfp,FFT_Data,bigint>& x,
                                           const Plaintext<gfp,FFT_Data,bigint>& y)
{
  if (z.Field_Data!=x.Field_Data)  { throw field_mismatch(); }
  if (z.Field_Data!=y.Field_Data)  { throw field_mismatch(); }

  if (x.type==Both && y.type!=Both)      { z.type=y.type; }
  else if (y.type==Both && x.type!=Both) { z.type=x.type; }
  else if (x.type!=y.type)               { throw rep_mismatch(); }
  else                                   { z.type=x.type; }

  z.allocate();
  if (z.type!=Polynomial)
    {
      z.a.resize(z.num_slots());
      for (unsigned int i=0; i<z.a.size(); i++)
        { z.a[i] = (x.a[i] + y.a[i]); }
    }
  if (z.type!=Evaluation)
    { for (unsigned int i=0; i<z.b.size(); i++)
        { z.b[i]=x.b[i]+y.b[i];
          if (z.b[i]>(*z.Field_Data).get_prime())
	    { z.b[i]-=(*z.Field_Data).get_prime(); }
	}
    }
}


template<>
void add(Plaintext<gfp,PPData,bigint>& z,const Plaintext<gfp,PPData,bigint>& x,
                                         const Plaintext<gfp,PPData,bigint>& y)
{
  if (z.Field_Data!=x.Field_Data)  { throw field_mismatch(); }
  if (z.Field_Data!=y.Field_Data)  { throw field_mismatch(); }

  if (x.type==Both && y.type!=Both)      { z.type=y.type; }
  else if (y.type==Both && x.type!=Both) { z.type=x.type; }
  else if (x.type!=y.type)               { throw rep_mismatch(); }
  else                                   { z.type=x.type; }

  if (z.type!=Polynomial)
    {
      z.a.resize(z.num_slots());
      for (unsigned int i=0; i<z.a.size(); i++)
        { z.a[i] = (x.a[i] + y.a[i]); }
    }
  if (z.type!=Evaluation)
    { for (unsigned int i=0; i<z.b.size(); i++)
        { z.b[i]=x.b[i]+y.b[i];
          if (z.b[i]>(*z.Field_Data).get_prime())
	    { z.b[i]-=(*z.Field_Data).get_prime(); }
	}
    }
}




template<>
void add(Plaintext<gf2n_short,P2Data,int>& z,const Plaintext<gf2n_short,P2Data,int>& x,
                                       const Plaintext<gf2n_short,P2Data,int>& y)
{
  if (z.Field_Data!=x.Field_Data)  { throw field_mismatch(); }
  if (z.Field_Data!=y.Field_Data)  { throw field_mismatch(); }

  if (x.type==Both && y.type!=Both)      { z.type=y.type; }
  else if (y.type==Both && x.type!=Both) { z.type=x.type; }
  else if (x.type!=y.type)               { throw rep_mismatch(); }
  else                                   { z.type=x.type; }

  z.allocate();
  if (z.type!=Polynomial)
    {
      z.a.resize(z.num_slots());
      for (unsigned int i=0; i<z.a.size(); i++)
        { z.a[i].add(x.a[i],y.a[i]); }
    }
  if (z.type!=Evaluation)
    { for (unsigned int i=0; i<z.b.size(); i++)
        {
          z.b[i]=x.b[i] ^ y.b[i];
        }
    }
}


template<>
void sub(Plaintext<gfp,FFT_Data,bigint>& z,const Plaintext<gfp,FFT_Data,bigint>& x,
                                           const Plaintext<gfp,FFT_Data,bigint>& y)
{
  if (z.Field_Data!=x.Field_Data)  { throw field_mismatch(); }
  if (z.Field_Data!=y.Field_Data)  { throw field_mismatch(); }

  if (x.type==Both && y.type!=Both)      { z.type=y.type; }
  else if (y.type==Both && x.type!=Both) { z.type=x.type; }
  else if (x.type!=y.type)               { throw rep_mismatch(); }
  else                                   { z.type=x.type; }

  z.allocate();
  if (z.type!=Polynomial)
    {
      z.a.resize(z.num_slots());
      for (unsigned int i=0; i<z.a.size(); i++)
        { z.a[i]= (x.a[i] - y.a[i]); }
    }
  if (z.type!=Evaluation)
    { for (unsigned int i=0; i<z.b.size(); i++)
        {
          z.b[i]=x.b[i];
          z.b[i]-=y.b[i];
          if (z.b[i]<0)
            { z.b[i]+=(*z.Field_Data).get_prime(); }
        }
    }
}



template<>
void sub(Plaintext<gfp,PPData,bigint>& z,const Plaintext<gfp,PPData,bigint>& x,
                                         const Plaintext<gfp,PPData,bigint>& y)
{
  if (z.Field_Data!=x.Field_Data)  { throw field_mismatch(); }
  if (z.Field_Data!=y.Field_Data)  { throw field_mismatch(); }

  if (x.type==Both && y.type!=Both)      { z.type=y.type; }
  else if (y.type==Both && x.type!=Both) { z.type=x.type; }
  else if (x.type!=y.type)               { throw rep_mismatch(); }
  else                                   { z.type=x.type; }

  z.allocate();
  if (z.type!=Polynomial)
    {
      z.a.resize(z.num_slots());
      for (unsigned int i=0; i<z.a.size(); i++)
        { z.a[i] = (x.a[i] - y.a[i]); }
    }
  if (z.type!=Evaluation)
    { for (unsigned int i=0; i<z.b.size(); i++)
        { z.b[i]=x.b[i]-y.b[i];
          if (z.b[i]<0)
            { z.b[i]+=(*z.Field_Data).get_prime(); }
        }
    }
}





template<>
void sub(Plaintext<gf2n_short,P2Data,int>& z,const Plaintext<gf2n_short,P2Data,int>& x,
                                       const Plaintext<gf2n_short,P2Data,int>& y)
{
  if (z.Field_Data!=x.Field_Data)  { throw field_mismatch(); }
  if (z.Field_Data!=y.Field_Data)  { throw field_mismatch(); }

  if (x.type==Both && y.type!=Both)      { z.type=y.type; }
  else if (y.type==Both && x.type!=Both) { z.type=x.type; }
  else if (x.type!=y.type)               { throw rep_mismatch(); }
  else                                   { z.type=x.type; }

  z.allocate();
  if (z.type!=Polynomial)
    {
      z.a.resize(z.num_slots());
      for (unsigned int i=0; i<z.a.size(); i++)
        { z.a[i].sub(x.a[i],y.a[i]); }
    }
  if (z.type!=Evaluation)
    { for (unsigned int i=0; i<z.b.size(); i++)
        {
          z.b[i]=x.b[i] ^ y.b[i];
        }
    }
}



template<class T,class FD,class S>
void mul(Plaintext<T,FD,S>& z,const Plaintext<T,FD,S>& x,const Plaintext<T,FD,S>& y)
{
  if (z.Field_Data!=x.Field_Data)  { throw field_mismatch(); }
  if (z.Field_Data!=y.Field_Data)  { throw field_mismatch(); }

  if (y.type==Polynomial) { throw not_implemented(); }
  if (x.type==Polynomial) { throw not_implemented(); }
  z.type=Evaluation;

  z.allocate();
  for (unsigned int i=0; i<z.a.size(); i++)
    { z.a[i] = (x.a[i] * y.a[i]); }
}


template<>
void Plaintext<gfp,FFT_Data,bigint>::negate()
{
  if (type!=Polynomial)
    {
      a.resize(num_slots());
      for (unsigned int i=0; i<a.size(); i++)
        { a[i].negate(); }
    }
  if (type!=Evaluation)
    { for (unsigned int i=0; i<b.size(); i++)
        { if (b[i]!=0)
            {
              b[i]-=(*Field_Data).get_prime();
              b[i].negate();
            }
	}
    }
}

template<>
void Plaintext<gfp,PPData,bigint>::negate()
{
  if (type!=Polynomial)
    {
      a.resize(num_slots());
      for (unsigned int i=0; i<a.size(); i++)
        { a[i].negate(); }
    }
  if (type!=Evaluation)
    { for (unsigned int i=0; i<b.size(); i++)
        { if (b[i]!=0)
            { b[i]=(*Field_Data).get_prime()-b[i]; }
	}
    }
}



template<>
void Plaintext<gf2n_short,P2Data,int>::negate()
{
  return;
}



template<class T, class FD, class _>
AddableVector<typename FD::poly_type> Plaintext<T, FD, _>::mul_by_X_i(int i,
    const FHE_PK& pk) const
{
  return AddableVector<S>(get_poly()).mul_by_X_i(i, pk);
}



template<class T,class FD,class S>
bool Plaintext<T,FD,S>::equals(const Plaintext& x) const
{
  if (Field_Data!=x.Field_Data) { return false; }
  if (type!=x.type)
    { if (type==Evaluation) { x.from_poly(); }
      else                  { from_poly(); }
    }

  if (type!=Polynomial and x.type!=Polynomial)
    {
      a.resize(num_slots());
      for (unsigned int i=0; i<a.size(); i++)
       { if (!(a[i] == x.a[i])) { return false; } }
    }
  else
    { for (unsigned int i=0; i<b.size(); i++)
       { if (b[i]!=x.b[i]) { return false; } }
    }
  return true;
}

template<>
bool Plaintext<gfp, FFT_Data, bigint>::is_diagonal() const
{
  if (type != Evaluation)
    {
      for (size_t i = 1; i < b.size(); i++)
        if (b[i] != 0)
          return false;
    }

  if (type != Polynomial)
    {
      auto first = a[0];
      for (auto& x : a)
        if (x != first)
          return false;
    }

  return true;
}

template<>
bool Plaintext<gf2n_short, P2Data, int>::is_diagonal() const
{
  if (type == Polynomial)
    from_poly();

  auto first = a[0];
  for (auto& x : a)
    if (x != first)
      return false;

  return true;
}



template <class T,class FD,class S>
void Plaintext<T,FD,S>::pack(octetStream& o) const
{
  to_poly();
  o.store((unsigned int)b.size());
  for (unsigned int i = 0; i < b.size(); i++)
    o.store(b[i]);
}

template <class T,class FD,class S>
void Plaintext<T,FD,S>::unpack(octetStream& o)
{
  type = Polynomial;
  unsigned int size;
  o.get(size);
  allocate();
  if (size != b.size() and size != 0)
    throw length_error("unexpected length received");
  for (unsigned int i = 0; i < size; i++)
    b[i] = o.get<S>();
}



template <>
size_t Plaintext<gfp, FFT_Data, bigint>::report_size(ReportType type)
{
  size_t res = 0;
  if (type != MINIMAL)
    res += sizeof(gfp) * a.capacity();
  for (unsigned int i = 0; i < b.size(); i++)
    res += b[i].report_size(type);
  return res;
}


template <>
size_t Plaintext<gf2n_short, P2Data, int>::report_size(ReportType type)
{
  size_t res = 0;
  if (type != MINIMAL)
    res += sizeof(gf2n_short) * a.capacity();
  res += sizeof(int) * b.size();
  return res;
}


template<class T, class FD, class S>
void Plaintext<T, FD, S>::print_evaluation(int n_elements, string desc) const
{
  cout << desc;
  for (int i = 0; i < n_elements; i++)
    cout << " " << element(i);
  cout << endl;
}



template class Plaintext<gfp,FFT_Data,bigint>;

template void mul(Plaintext<gfp,FFT_Data,bigint>& z,const Plaintext<gfp,FFT_Data,bigint>& x,const Plaintext<gfp,FFT_Data,bigint>& y);



template class Plaintext<gfp,PPData,bigint>;

template void mul(Plaintext<gfp,PPData,bigint>& z,const Plaintext<gfp,PPData,bigint>& x,const Plaintext<gfp,PPData,bigint>& y);



template class Plaintext<gf2n_short,P2Data,int>;

template void mul(Plaintext<gf2n_short,P2Data,int>& z,const Plaintext<gf2n_short,P2Data,int>& x,const Plaintext<gf2n_short,P2Data,int>& y);

