
#include "Math/gf2n.h"
#include "Math/Bit.h"

#include "Tools/intrinsics.h"
#include "Tools/Exceptions.h"

#include <stdint.h>

const false_type ValueInterface::characteristic_two;
const false_type ValueInterface::prime_field;
const false_type ValueInterface::invertible;

template<class U>
int gf2n_<U>::l[4];
template<class U>
bool gf2n_<U>::useC;

word gf2n_short_table[256][256];

#define num_2_fields 17

/* Require
 *  2*(n-1)-64+t1<64
 */
int fields_2[num_2_fields][4] =
{
    { 4, 1, 0, 0 },
    { 5, 2, 0, 0 },
    { 6, 1, 0, 0 },
    { 7, 1, 0, 0 },
    { 8, 4, 3, 1 },
    { 9, 1, 0, 0 },
    { 10, 3, 0, 0},
    { 11, 2, 0, 0},
    { 12, 3, 0, 0},
    { 14, 5, 0, 0},
    { 15, 1, 0, 0},
    { 16, 5, 3, 1 },
    { 28, 1, 0, 0 },
    { 40, 20, 15, 10 },
    { 63, 1, 0, 0 },
    { 64, 4, 3, 1},
    { 128, 7, 2, 1 },
};

template<class U>
string gf2n_<U>::options()
{
    string res = to_string(fields_2[0][0]);
    for (int i = 1; i < num_2_fields; i++)
    {
        int n = fields_2[i][0];
        if (n <= MAX_N_BITS)
            res += ", " + to_string(n);
    }
    return res;
}

template<class U>
void gf2n_<U>::init_tables()
{
  if (sizeof(word)!=8)
    { cout << "Word size is wrong" << endl; 
      throw not_implemented();
    }
  int i,j;
  for (i=0; i<256; i++)
    { for (j=0; j<256; j++)
        { word ii=i,jj=j;
          gf2n_short_table[i][j]=0;
          while (ii!=0)
            { if ((ii&1)==1) { gf2n_short_table[i][j]^=jj; }
              jj<<=1;
              ii>>=1;
            }
         }
    }
}

template<class U>
void gf2n_<U>::init_minimum(int lower)
{
  if (lower <= n)
    return;

  for (int i = 0; i < num_2_fields; i++)
    {
      int n = fields_2[i][0];
      if (lower <= n and n <= MAX_N_BITS)
        return init_field(n);
    }
  throw runtime_error("no suitable field for minimum degree " + to_string(lower));
}

void gf2n_short::init_field(int nn)
{
  super::init_field(nn == 0 ? DEFAULT_LENGTH : nn);
}

template<class U>
void gf2n_<U>::init_field(int nn)
{
  if (nn == 0)
    {
      nn = MAX_N_BITS;
#ifdef VERBOSE
      cerr << "Using GF(2^" << nn << ")" << endl;
#endif
    }

  if (nn == n)
      return;

  assert(n == 0);

  init_tables();
  int i,j=-1;
  for (i=0; i<num_2_fields && j==-1; i++)
    { if (nn==fields_2[i][0]) { j=i; } }

  if (nn > MAX_N_BITS)
    throw runtime_error("Bit length not supported.\n"
        "You might need to compile with USE_GF2N_LONG = 1.\n"
        "Remember to run 'make clean'.");

  if (j==-1)
    {
      throw gf2n_not_supported(nn, options());
    }

  n=nn;
  l[0] = MAX_N_BITS - n;
  for (int i = 1; i < 4; i++)
    {
      if (fields_2[j][i] == 0)
        break;
      nterms = i;
      t[i] = fields_2[j][i];
      l[i] = MAX_N_BITS + t[i] - n;
    }
  assert(nterms > 0);

  mask = (n == MAX_N_BITS) ? ~U() : ~(U(~U()) << n);
  uppermask = U(~U()) << (MAX_N_BITS - t[1]);
  lowermask = ~uppermask;

  init_multiplication();

#ifdef __PCLMUL__
  useC = not cpu_has_pclmul();
#else
  useC = true;
#endif
}


template<class U>
void gf2n_<U>::init_multiplication()
{
  if (n <= 8)
    {
      word red = 1;
      for (int i = 1; i <= nterms; i++)
        red ^= (1 << t[i]);
      memset(mult_table, 0, sizeof(mult_table));
      for (int i = 1; i < 1 << n; i++)
        {
          for (int j = 1; j <= i; j++)
            {
              word tmp = mult_table[i / 2][j];
              tmp <<= 1;
              if (i & 1)
                tmp ^= j;
              if (tmp >> n)
                tmp ^= red;
              tmp &= Integer(mask).get();
              mult_table[i][j] = tmp;
              mult_table[j][i] = tmp;
          }
        }
    }
}


template<class U>
void gf2n_<U>::specification(octetStream& os)
{
  os.store(sizeof(U));
  os.store(degree());
}


/* Takes 8bit x and y and returns the 16 bit product in c1 and c0
      ans = (c1<<8)^c0
   where c1 and c0 are 8 bit
*/
void mul(octet x, octet y, octet& c0, octet& c1)
{
  auto full = gf2n_short_table[octet(x)][octet(y)];
  c0 = full;
  c1 = full >> 8;
}

/* Takes 16bit x and y and returns the 32 bit product in c1 and c0
      ans = (c1<<16)^c0
   where c1 and c0 are 16 bit
*/
inline void mul16(word x,word y,word& c0,word& c1)
{
  word a1=x&(0xFF), b1=y&(0xFF);
  word a2=x>>8,     b2=y>>8;

  c0=gf2n_short_table[a1][b1];
  c1=gf2n_short_table[a2][b2];
  word te=gf2n_short_table[a1][b2]^gf2n_short_table[a2][b1];
  c0^=(te&0xFF)<<8;
  c1^=te>>8;
}

/* Takes 16 bit x and y and returns the 32 bit product */
inline word mul16(word x,word y)
{
  word a1=x&(0xFF), b1=y&(0xFF);
  word a2=x>>8,     b2=y>>8;

  word ans=gf2n_short_table[a2][b2]<<8;
  ans^=gf2n_short_table[a1][b2]^gf2n_short_table[a2][b1];
  ans<<=8;
  ans^=gf2n_short_table[a1][b1];

  return ans;
}



template<class U>
void gf2n_<U>::reduce(U xh, U xl)
{
  if (n == 0)
    throw runtime_error("gf2n not initialized");

  if (2 * (n - 1) - MAX_N_BITS + t[1] < MAX_N_BITS)
    {
      // Deal with xh first
      a = xl;
      for (int i = 0; i < nterms + 1; i++)
        a ^= (xh << l[i]);

      // Now deal with last word
      U hi = a >> n;
      while (hi != 0)
        {
          a &= mask;

          a ^= hi;
          for (int i = 1; i < nterms + 1; i++)
            a ^= (hi << t[i]);

          hi = a >> n;
        }
    }
  else
    {
      a = xl;
      U upper, lower;
      upper = xh & uppermask;
      lower = xh & lowermask;
      // Upper part
      U tmp = 0;
      for (int i = 0; i < nterms + 1; i++)
        tmp ^= (upper >> (n - t[1] - l[i]));
      lower ^= (tmp >> (l[1]));
      a ^= (tmp << (n - l[1]));
      // Lower part
      for (int i = 0; i < nterms + 1; i++)
        a ^= (lower << l[i]);
    }
}


void mul32(word x,word y,word& ans)
{
  word a1=x&(0xFFFF),b1=y&(0xFFFF);
  word a2=x>>16,     b2=y>>16;

  word c0,c1;

  ans=mul16(a1,b1);
  word upp=mul16(a2,b2);

  mul16(a1,b2,c0,c1);
  ans^=c0<<16;       upp^=c1;

  mul16(a2,b1,c0,c1);
  ans^=c0<<16;       upp^=c1;

  ans^=(upp<<32);
}


void mul(word x, word y, word& lo, word& hi)
{
   word c,d,e,t;
   word xl=x&0xFFFFFFFF,yl=y&0xFFFFFFFF;
   word xh=x>>32,yh=y>>32;
   mul32(xl,yl,c);
   mul32(xh,yh,d);
   mul32((xl^xh),(yl^yh),e);
   t=c^e^d;
   lo=c^(t<<32);
   hi=d^(t>>32);
}


word to_word(word x)
{
  return x;
}

word to_word(int128 x)
{
  return x.get_lower();
}

template<class U>
gf2n_<U>& gf2n_<U>::mul(const gf2n_& x,const gf2n_& y)
{
  U hi,lo;

  if (n <= 8)
    {
      *this = mult_table[octet(to_word(x.a))][octet(to_word(y.a))];
      return *this;
    }
  else if (useC or n > 64)
    {
      ::mul(x.a, y.a, lo, hi);
    }
  else
    {
      int128 res = clmul<0>(int128(x.a).a, int128(y.a).a);

      if (MAX_N_BITS <= 64)
        {
          hi = res.get_upper();
          lo = res.get_lower();
        }
      else
        {
          res.to(lo);
          hi = 0;
        }
    }

  reduce(hi,lo);
  return *this;
}

template<class U>
gf2n_<U> gf2n_<U>::operator*(const Bit& x) const
{
  return x.get() ? a : 0;
}


template<class U>
gf2n_<U> gf2n_<U>::invert() const
{
  if (n < 64)
    return U(invert<word>(a));
  else
    {
      gf2n_ res;
      res.a = invert<int128>(a).get_lower();
      return res;
    }
}

template<>
gf2n_<int128> gf2n_<int128>::invert() const
{
  if (n < 64)
    return int128(invert<word>(a.get_lower()));
  if (n < 128)
    return invert<int128>(a);
  else
    return invert<bit_plus<int128>>(a).get_lower();
}

template<class U>
template<class T>
T gf2n_<U>::invert(T a) const
{
  if (is_one())  { return a; }
  if (is_zero()) { throw division_by_zero(); }

  T u,v=a,B=0,D=1,mod=1;

  mod ^= (T(1) << n);
  for (int i = 1; i <= nterms; i++)
    mod ^= (1ULL << t[i]);

  u=mod; v=a;
  
  while (u!=0)
    { while ((u&1)==0)
	    { u>>=1;
          if ((B&1)!=0) { B^=mod; }
 	      B>>=1;
        }
      while ((v&1)==0 && v!=0)
	    { v>>=1;
	      if ((D&1)!=0) { D^=mod; }
		  D>>=1;
	    }

      if (u>=v) { u=u^v; B=B^D; }
	  else      { v=v^u; D=D^B; }
   }

  return D;
}


template<class U>
gf2n_<U> gf2n_<U>::operator <<(int i) const
{
  if (i < 0)
    throw runtime_error("cannot shift by negative");
  else if (i >= n)
    return 0;
  else
    return a << i;
}

template<class U>
gf2n_<U> gf2n_<U>::operator >>(int i) const
{
  if (i < 0)
    throw runtime_error("cannot shift by negative");
  else if (i >= n)
    return 0;
  else
    return a >> i;
}


template<class U>
void gf2n_<U>::randomize(PRNG& G, int n)
{
  (void) n;
  a=G.get<U>();
  a&=mask;
}

template<>
void gf2n_<octet>::output(ostream& s,bool human) const
{
  if (human)
    s << hex << showbase << word(a) << dec;
  else
    s.write((char*) &a, sizeof(octet));
}

template<class U>
void gf2n_<U>::output(ostream& s,bool human) const
{
  if (human)
    { s << hex << showbase << a << dec; }
  else
    { s.write((char*) &a, (sizeof(U))); }
}

template<class U>
void gf2n_<U>::input(istream& s,bool human)
{
  if (s.peek() == EOF)
    { if (s.tellg() == 0)
        { cout << "IO problem. Empty file?" << endl;
          throw file_error("gf2n input");
        }
      throw end_of_file("gf2n");
    }

  if (human)
    { s >> hex >> a >> dec; } 
  else
    { s.read((char*) &a, sizeof(U)); }

  a &= mask;
}

gf2n_short gf2n_short::cut(int128 x)
{
  return x.get_lower();
}

gf2n_short::gf2n_short(const int128& a)
{
  reduce(a.get_upper(), a.get_lower());
}


// Expansion is by x=y^5+1 (as we embed GF(256) into GF(2^40)
void expand_byte(gf2n_short& a,int b)
{
  gf2n_short x,xp;
  x = (32+1);
  xp.assign_one();
  a.assign_zero();

  while (b!=0)
    { if ((b&1)==1)
        { a.add(a,xp); }
      xp *= (x);
      b>>=1;
    }
}


// Have previously worked out the linear equations we need to solve
void collapse_byte(int& b,const gf2n_short& aa)
{
  word w=aa.get();
  int e35=(w>>35)&1;
  int e30=(w>>30)&1;
  int e25=(w>>25)&1;
  int e20=(w>>20)&1;
  int e15=(w>>15)&1;
  int e10=(w>>10)&1;
  int  e5=(w>>5)&1;
  int  e0=w&1;
  int a[8];
  a[7]=e35;
  a[6]=e30^a[7];
  a[5]=e25^a[7];
  a[4]=e20^a[5]^a[6]^a[7];
  a[3]=e15^a[7];
  a[2]=e10^a[3]^a[6]^a[7];
  a[1]=e5^a[3]^a[5]^a[7];
  a[0]=e0^a[1]^a[2]^a[3]^a[4]^a[5]^a[6]^a[7];

  b=0;
  for (int i=7; i>=0; i--)
    { b=b<<1;
      b+=a[i];
    }
}

template class gf2n_<word>;
template class gf2n_<octet>;
template class gf2n_<int128> ;
