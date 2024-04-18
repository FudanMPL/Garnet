
#include "Tools/random.h"
#include "Math/bigint.h"
#include "Math/fixint.h"
#include "Math/Z2k.hpp"
#include "Math/gfp.h"
#include "Tools/Subroutines.h"
#include <stdio.h>
#include <sodium.h>

#include <iostream>
using namespace std;


PRNG::PRNG() :
    cnt(0), n_cached_bits(0), cached_bits(0), initialized(false)
{
#if defined(__AES__) || !defined(__x86_64__)
  #ifdef USE_AES
    useC=(Check_CPU_support_AES()==0);
  #endif
#endif
}

PRNG::PRNG(octetStream& seed) : PRNG()
{
  SetSeed(seed.consume(SEED_SIZE));
}

void PRNG::ReSeed()
{
  randombytes_buf(seed, SEED_SIZE);
  InitSeed();
}

void PRNG::SeedGlobally(const PlayerBase& P)
{
  octet seed[SEED_SIZE];
  Create_Random_Seed(seed, P, SEED_SIZE);
  SetSeed(seed);
}

void PRNG::SeedGlobally(const Player& P, bool secure)
{
  if (secure)
    SeedGlobally(static_cast<const PlayerBase&>(P));
  else
    {
      octetStream os;
      if (P.my_num() == 0)
        {
          randombytes_buf(seed, SEED_SIZE);
          os.append(seed, SEED_SIZE);
          P.send_all(os);
        }
      else
        {
          P.receive_player(0, os);
          os.consume(seed, SEED_SIZE);
        }
      InitSeed();
    }
}

void PRNG::SetSeed(const octet* inp)
{
  memcpy(seed,inp,SEED_SIZE*sizeof(octet));
  InitSeed();
}

void PRNG::SetSeed(PRNG& G)
{
  octet tmp[SEED_SIZE];
  G.get_octets(tmp, sizeof(tmp));
  SetSeed(tmp);
}

void PRNG::InitSeed()
{
  initialized = true;
  #ifdef USE_AES
     if (useC)
        { aes_schedule(KeyScheduleC,seed); }
     else
        { aes_schedule(KeySchedule,seed); }
     memset(state,0,RAND_SIZE*sizeof(octet));
     for (int i = 0; i < PIPELINES; i++)
         state[i*AES_BLK_SIZE] = i;
  #else
     memcpy(state,seed,SEED_SIZE*sizeof(octet));
  #endif
  hash();
  //cout << "SetSeed : "; print_state(); cout << endl;
}


void PRNG::print_state() const
{
  unsigned i;
  for (i=0; i<SEED_SIZE; i++)
    { if (seed[i]<10){ cout << "0"; }
      cout << hex << (int) seed[i]; 
    }
  cout << "\t";
  for (i=0; i<RAND_SIZE; i++)
    { if (random[i]<10) { cout << "0"; }
      cout << hex << (int) random[i]; 
    }
  cout << "\t";
  for (i=0; i<RAND_SIZE; i++)
    { if (state[i]<10) { cout << "0"; }
      cout << hex << (int) state[i];
    }
  cout << " " << dec << cnt << " : ";
}


void PRNG::hash()
{
  assert(initialized);
  #ifndef USE_AES
    unsigned char tmp[RAND_SIZE + SEED_SIZE];
    randombytes_buf_deterministic(tmp, sizeof tmp, seed);
    memcpy(random, tmp, RAND_SIZE);
    memcpy(seed, tmp + RAND_SIZE, SEED_SIZE);
  #else
    if (useC)
       { software_ecb_aes_128_encrypt<PIPELINES>((__m128i*)random,(__m128i*)state,KeyScheduleC); }
    else
       { ecb_aes_128_encrypt<PIPELINES>((__m128i*)random,(__m128i*)state,KeySchedule); }
  #endif
  // This is a new random value so we have not used any of it yet
  cnt=0;
}



void PRNG::next()
{
  // Increment state
  for (int i = 0; i < PIPELINES; i++)
    {
      int64_t* s = (int64_t*)&state[i*AES_BLK_SIZE];
      s[0] += PIPELINES;
      if (s[0] == 0)
          s[1]++;
    }
  hash();
}


unsigned int PRNG::get_uint()
{
  // We need four bytes of randomness
  if (cnt>RAND_SIZE-4) { next(); }
  unsigned int a0=random[cnt],a1=random[cnt+1],a2=random[cnt+2],a3=random[cnt+3];
  cnt=cnt+4;
  unsigned int ans=(a0+(a1<<8)+(a2<<16)+(a3<<24));
  // print_state(); cout << " UINT " << ans << endl;
  return ans;
}

unsigned int PRNG::get_uint(int upper)
{
	// adopting Java 7 implementation of bounded nextInt here
	if (upper <= 0)
		throw invalid_argument("Must be positive");
	// power of 2 case
	if ((upper & (upper - 1)) == 0) {
		unsigned int r = (upper < 255) ? get_uchar() : get_uint();
		// zero out higher order bits
		return r % upper;
	}
	// not power of 2
	unsigned int r, reduced;
	bool use_char = upper <= 128;
	do {
		r = use_char ? get_uchar() : get_uint();
		reduced = r % upper;
	} while (int(r - reduced + (upper - 1)) > (use_char ? 256 : 0));
	return reduced;
}

void PRNG::get_octetStream(octetStream& ans,int len)
{
  ans.resize(len);
  for (int i=0; i<len; i++)
    { ans.data[i]=get_uchar(); }
  ans.len=len;
  ans.ptr=0;
}


void PRNG::randomBnd(mp_limb_t* res, const mp_limb_t* B, size_t n_bytes, mp_limb_t mask)
{
  switch (n_bytes)
  {
  case 16:
    randomBnd<16>(res, B, mask);
    return;
  case 32:
    randomBnd<32>(res, B, mask);
    return;
  default:
    {
      assert(n_bytes != 0);
      size_t n_limbs = (n_bytes + sizeof(mp_limb_t) - 1) / sizeof(mp_limb_t);
      do
      {
        get_octets((octet*) res, n_bytes);
        res[n_limbs - 1] &= mask;
      }
      while (mpn_cmp(res, B, n_limbs) >= 0);
    }
  }
}

template<>
void PRNG::randomBnd(bigint& x, const bigint& B, bool positive)
{
  int i = 0;
  do
    {
      get(x, numBits(B), true);
      if (i++ > 1000)
        {
          cout << x << " - " << B << " = " << x - B << endl;
          throw runtime_error("bounded randomness error");
        }
    }
  while (x >= B);
  if (!positive)
    {
      if (get_bit())
        mpz_neg(x.get_mpz_t(), x.get_mpz_t());
    }
}

void PRNG::get(bigint& res, int n_bits, bool positive)
{
  assert(n_bits > 0);
  int n_bytes = (n_bits + 7) / 8;
  int n_words = DIV_CEIL(n_bytes, sizeof(word));
  word words[n_words];
  octet* bytes = (octet*) words;
  words[n_words - 1] = 0;
  get_octets(bytes, n_bytes);
  octet mask = -1;
  if (n_bits % 8 > 0)
    mask = (1 << (n_bits % 8)) - 1;
  bytes[n_bytes - 1] &= mask;
  mpz_import(res.get_mpz_t(), n_words, -1, sizeof(word), -1, 0, bytes);
  if (not positive and (get_bit()))
    mpz_neg(res.get_mpz_t(), res.get_mpz_t());
}
