
#include "Subroutines.h"
#include "Math/modp.h"

#include "Math/modp.hpp"

void Subs(modp& ans,const vector<int>& poly,const modp& x,const Zp_Data& ZpD)
{
  modp one,co;
  assignOne(one,ZpD);
  assignZero(ans,ZpD);
  for (int i=poly.size()-1; i>=0; i--)
    { Mul(ans,ans,x,ZpD);
      if (poly[i] > 0)
        {
          for (int j = 0; j < poly[i]; j++)
            Add(ans, ans, one, ZpD);
        }
      if (poly[i] < 0)
        {
          for (int j = 0; j < -poly[i]; j++)
            Sub(ans, ans, one, ZpD);
        }
    }
}



/* Find a m'th primitive root moduli the current prime
 *   This is deterministic so all players have the same root of unity
 *   poly is Phi_m(X)
 */
modp Find_Primitive_Root_m(int m,const vector<int>& poly,const Zp_Data& ZpD)
{
  modp ans,e,one,base;
  assignOne(one,ZpD);
  assignOne(base,ZpD);
  bigint   exp;
  exp=(ZpD.pr-1)/m;
  bool flag=true;
  while (flag)
    { Add(base,base,one,ZpD);   // Keep incrementing base until we hit the answer
      Power(ans,base,exp,ZpD);
      // e=Phi(ans)
      Subs(e,poly,ans,ZpD);
      if (isZero(e,ZpD)) { flag=false; }
    }
  return ans;
}



/* Find a (2m)'th primitive root moduli the current prime
 *   This is deterministic so all players have the same root of unity
 *   poly is Phi_m(X)
 */
modp Find_Primitive_Root_2m(int m,const vector<int>& poly,const Zp_Data& ZpD)
{
  // Thin out poly, where poly is Phi_m(X) by 2
  vector<int> poly2;
  poly2.resize(2*poly.size());
  for (unsigned int i=0; i<poly.size(); i++)
    { poly2[2*i]=poly[i];
      poly2[2*i+1]=0;
    }
  modp ans=Find_Primitive_Root_m(2*m,poly2,ZpD);
  return ans;
}



/* Find an mth primitive root moduli the current prime
 *   This is deterministic so all players have the same root of unity
 * This assumes m is a power of two and so the cyclotomic polynomial
 * is  F=X^{m/2}+1
 */
modp Find_Primitive_Root_2power(int m,const Zp_Data& ZpD)
{
  modp ans,e,one,base;
  assignOne(one,ZpD);
  assignOne(base,ZpD);
  bigint   exp;
  exp=(ZpD.pr-1)/m;
  bool flag=true;
  while (flag)
    { Add(base,base,one,ZpD);   // Keep incrementing base until we hit the answer
      Power(ans,base,exp,ZpD);
      // e=ans^{m/2}+1
      Power(e,ans,m/2,ZpD);
      Add(e,e,one,ZpD);
      if (isZero(e,ZpD)) { flag=false; }
    }
  return ans;
}
