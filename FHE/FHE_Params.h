#ifndef _FHE_Params
#define _FHE_Params

/* Class to hold the FHE Parameters 
 *
 * The idea is that there is a single global FHE_Params structure
 * called params; which holds all the data
 *
 * Likewise initialising params sets up all the data for other
 * classes such the global ring etc
 *
 */
#include "FHE/FFT_Data.h"
#include "FHE/DiscreteGauss.h"
#include "Tools/random.h"
#include "Protocols/config.h"

/**
 * Cryptosystem parameters
 */
class FHE_Params
{
  protected:
  // Pair of moduli in FFTData[0] and FFTData[1] for the levels
  vector<FFT_Data> FFTData;

  // Random generator for Multivariate Gaussian Distribution etc
  mutable DiscreteGauss Chi;

  // Data for distributed decryption
  int sec_p;
  bigint Bval;
  int matrix_dim;

  FFT_Data fd;

  public:

  /**
   * Initialization.
   * @param n_mults number of ciphertext multiplications (0/1)
   * @param drown_sec parameter for function privacy (default 40)
   */
  FHE_Params(int n_mults = 1, int drown_sec = DEFAULT_SECURITY);

  int n_mults() const { return FFTData.size() - 1; }

  void set(const Ring& R,const vector<bigint>& primes);
  void set(const vector<bigint>& primes);
  void set_sec(int sec);
  void set_min_sec(int sec);

  void set_matrix_dim(int matrix_dim);
  void set_matrix_dim_from_options();
  int get_matrix_dim() const         { return matrix_dim; }

  const vector<FFT_Data>& FFTD() const { return FFTData; }

  const bigint& p0() const           { return FFTData[0].get_prime(); }
  const bigint& p1() const           { return FFTData[1].get_prime(); }
  bigint Q() const;

  int secp() const                   { return sec_p;        }
  const bigint& B() const            { return Bval;         }
  double get_R() const               { return Chi.get_R();  }
  void set_R(double R) const         { return Chi.set(R); }
  DiscreteGauss get_DG() const       { return Chi; }

  int phi_m() const                  { return FFTData[0].phi_m(); }
  const Ring& get_ring()             { return FFTData[0].get_R(); }

  /// Append to buffer
  void pack(octetStream& o) const;

  /// Read from buffer
  void unpack(octetStream& o);

  bool operator!=(const FHE_Params& other) const;

  /**
   * Generate parameter for computation modulo a prime
   * @param plaintext_length bit length of prime
   */
  void basic_generation_mod_prime(int plaintext_length);

  template<class FD>
  const FD& get_plaintext_field_data() const;

  bigint get_plaintext_modulus() const;
};

#endif
