#ifndef _PPData
#define _PPData

#include "Math/modp.h"
#include "Math/Zp_Data.h"
#include "Math/gfpvar.h"
#include "Math/fixint.h"
#include "FHE/Ring.h"
#include "FHE/FFT_Data.h"

/* Class for holding modular arithmetic data wrt the ring 
 *
 * It also holds the ring
 */

class PPData
{
  public:
  typedef gfp T;
  typedef bigint S;
  typedef typename FFT_Data::poly_type poly_type;

  Ring    R;
  Zp_Data prData;

  modp root;        // m'th Root of Unity mod pr 
  
  void init(const Ring& Rg,const Zp_Data& PrD);

  PPData() { ; }
  PPData(const Ring& Rg,const Zp_Data& PrD)
    { init(Rg,PrD); }

  const Zp_Data& get_prD() const   { return prData; }
  const bigint&  get_prime() const { return prData.pr; }
  int phi_m() const                { return R.phi_m(); }
  int m()     const                { return R.m();   }
  int num_slots() const { return R.phi_m(); }
  

  int p(int i)      const        { return R.p(i);     }
  int p_inv(int i)  const        { return R.p_inv(i); }
  const vector<int>& Phi() const { return R.Phi();    }

  // Convert input vector from poly to evaluation representation
  //   - Uses naive method and not FFT, we only use this rarely in any case
  void to_eval(vector<modp>& elem) const;
  void from_eval(vector<modp>& elem) const;

  // Following are used to iteratively get slots, as we use PPData when
  // we do not have an efficient FFT algorithm
  gfp thetaPow,theta;
  int pow;
  void reset_iteration();
  void next_iteration(); 
  gfp get_evaluation(const vector<bigint>& mess) const;
  
};

#endif

