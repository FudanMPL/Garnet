#ifndef _NTL_Subs
#define _NTL_Subs

#include "FHE/Ring.h"
#include "FHE/FFT_Data.h"
#include "FHE/P2Data.h"
#include "FHE/PPData.h"
#include "FHE/FHE_Params.h"

/* Routines to set up key sizes given the number of players n
 *  - And size lgp of plaintext modulus p for the char p case
 */

class Parameters
{
  int n_parties;
  int plaintext_length;
  int sec;
  int slack;
  bool round_up;

public:
  Ring R;
  bigint pr0, pr1;

  Parameters(int n_parties, int plaintext_length, int sec, int slack = 0,
      bool round_up = false) :
      n_parties(n_parties), plaintext_length(plaintext_length), sec(sec), slack(
          slack), round_up(round_up)
  {
  }

  template<class FD>
  void generate_setup(FHE_Params& params, FD& FTD)
  {
    SPDZ_Data_Setup(params, FTD);
    params.set(R, {pr0, pr1});
  }

  template<class FD>
  void SPDZ_Data_Setup(FHE_Params& params, FD& FTD);

  int SPDZ_Data_Setup_Char_p_Sub(int idx, int& m, bigint& p,
      FHE_Params& params);

};

// Main setup routine
void generate_setup(int nparties, int lgp, int lg2,
    int sec, bool skip_2 = false, int slack = 0, bool round_up = false);

// semi-homomorphic, includes slack
template <class FD>
int generate_semi_setup(int plaintext_length, int sec,
    FHE_Params& params, FD& FieldD, bool round_up, int n = 1);

// field-independent semi-homomorphic setup
int common_semi_setup(FHE_Params& params, int m, bigint p, int& lgp0, int lgp1,
    bool round_up);

void init(Ring& Rg, int m, bool generate_poly);
void init(P2Data& P2D,const Ring& Rg);

namespace NTL
{
class GF2X;
}

NTL::GF2X get_F(const Ring& Rg);

// For use when we want p to be a specific value 
void SPDZ_Data_Setup_Char_p_General(Ring& R, PPData& PPD, bigint& pr0,
    bigint& pr1, int n, int sec, bigint& p, FHE_Params& params);

// generate moduli according to lengths and other parameters
void generate_moduli(bigint& pr0, bigint& pr1, const int m,
        const bigint p, const int lg2p0, const int lg2p1);

void generate_modulus(bigint& pr, const int m, const bigint p, const int lg2pr,
    const string& i = "0", const bigint& pr0 = 0);

// pre-generated dimensions for characteristic 2
void char_2_dimension(int& m, int& lg2);

// try to avoid expensive generation by loading from disk if possible
void load_or_generate(P2Data& P2D, const Ring& Rg);

int phi_N(int N);

#endif
