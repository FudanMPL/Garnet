
#include "FHE_Params.h"
#include "NTL-Subs.h"
#include "FHE/Ring_Element.h"
#include "Tools/Exceptions.h"
#include "Protocols/HemiOptions.h"
#include "Processor/OnlineOptions.h"

FHE_Params::FHE_Params(int n_mults, int drown_sec) :
    FFTData(n_mults + 1), Chi(0.7), sec_p(drown_sec), matrix_dim(1)
{
}

void FHE_Params::set(const Ring& R,
                     const vector<bigint>& primes)
{
  if (primes.size() != FFTData.size())
    throw runtime_error("wrong number of primes");

  for (size_t i = 0; i < FFTData.size(); i++)
    FFTData[i].init(R,primes[i]);

  set_sec(sec_p);
}

void FHE_Params::set_sec(int sec)
{
  assert(sec >= 0);
  sec_p=sec;
  Bval=1;  Bval=Bval<<sec_p;
  Bval=FFTData[0].get_prime()/(2*(1+Bval));
}

void FHE_Params::set_min_sec(int sec)
{
  set_sec(max(sec, sec_p));
}

void FHE_Params::set_matrix_dim(int matrix_dim)
{
  assert(matrix_dim > 0);
  if (FFTData[0].get_prime() != 0)
    throw runtime_error("cannot change matrix dimension after parameter generation");
  this->matrix_dim = matrix_dim;
}

void FHE_Params::set_matrix_dim_from_options()
{
  set_matrix_dim(
      HemiOptions::singleton.plain_matmul ?
          1 : OnlineOptions::singleton.batch_size);
}

bigint FHE_Params::Q() const
{
  bigint res = FFTData[0].get_prime();
  for (size_t i = 1; i < FFTData.size(); i++)
    res *= FFTData[i].get_prime();
  return res;
}

void FHE_Params::pack(octetStream& o) const
{
  o.store(FFTData.size());
  for(auto& fd: FFTData)
    fd.pack(o);
  Chi.pack(o);
  Bval.pack(o);
  o.store(sec_p);
  o.store(matrix_dim);
  fd.pack(o);
}

void FHE_Params::unpack(octetStream& o)
{
  size_t size;
  o.get(size);
  FFTData.resize(size);
  for (auto& fd : FFTData)
    fd.unpack(o);
  Chi.unpack(o);
  Bval.unpack(o);
  o.get(sec_p);
  o.get(matrix_dim);
  fd.unpack(o);
}

bool FHE_Params::operator!=(const FHE_Params& other) const
{
  if (FFTData != other.FFTData or Chi != other.Chi or sec_p != other.sec_p
      or Bval != other.Bval)
    {
      return true;
    }
  else
    return false;
}

void FHE_Params::basic_generation_mod_prime(int plaintext_length)
{
  if (n_mults() == 0)
    generate_semi_setup(plaintext_length, 0, *this, fd, false);
  else
    {
      Parameters parameters(1, plaintext_length, 0);
      parameters.generate_setup(*this, fd);
    }
}

template<>
const FFT_Data& FHE_Params::get_plaintext_field_data() const
{
  return fd;
}

template<>
const P2Data& FHE_Params::get_plaintext_field_data() const
{
  throw not_implemented();
}

template<>
const PPData& FHE_Params::get_plaintext_field_data() const
{
  throw not_implemented();
}

bigint FHE_Params::get_plaintext_modulus() const
{
  return fd.get_prime();
}
