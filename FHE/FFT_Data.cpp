#include "FHE/FFT_Data.h"
#include "FHE/FFT.h"

#include "FHE/Subroutines.h"

#include "Math/modp.hpp"



FFT_Data::FFT_Data() :
    twop(-1)
{
}

void FFT_Data::init(const Ring& Rg,const Zp_Data& PrD)
{
  R=Rg;
  prData=PrD;
  root.resize(2);

  // Find which case we are in 
  int hwt=Hwt(Rg.m());
  bigint prm1=PrD.pr-1;
  twop=1<<(numBits(Rg.m())+1);
  if (hwt==1)
    { twop=0; }
  else if ((prm1%twop)!=0)
    { twop=-twop; }

  if (twop==0)
    { int nb=numBits(Rg.m());
      nb=1<<(nb-1);
      if (nb==Rg.m())
        { //cout << Rg.m() << " " << PrD.pr << endl;
          root[0]=Find_Primitive_Root_2power(Rg.m(),PrD);
          Inv(root[1],root[0],PrD);
          to_modp(iphi,Rg.phi_m(),PrD);
          Inv(iphi,iphi,PrD);
          compute_roots(Rg.m());
        }
    }
  else 
    { bigint pr=PrD.pr;
      if ((pr-1)%(2*Rg.m())!=0)
	{ throw invalid_params(); }
      root[0]=Find_Primitive_Root_2m(Rg.m(),Rg.Phi(),PrD);
      Inv(root[1],root[0],PrD); 
      compute_roots(2 * Rg.m());

      int ptwop=twop; if (twop<0) { ptwop=-twop; }

      powers.resize(2,vector<modp>(Rg.m()));
      powers_i.resize(2,vector<modp>(Rg.m()));
      b.resize(2,vector<modp>(ptwop));

      modp rInv,bi;
      bigint ee=ptwop; ee=ee*Rg.m();
      for (int r=0; r<2; r++)
        { assignOne(powers[r][0],PrD);
          if (r==0)
            { to_modp(powers_i[0][0],ptwop,PrD);  }
          else
            { to_modp(powers_i[1][0],ee,PrD);  }
          Inv(powers_i[r][0],powers_i[r][0],PrD);

          Inv(rInv,root[r],PrD);
          assignOne(b[r][Rg.m()-1],PrD);
          for (long i=1; i<Rg.m(); i++)
            { long iSqr=(i*i)%(2*Rg.m());
              Power(powers[r][i],root[r],iSqr,PrD);
              Mul(powers_i[r][i],powers[r][i],powers_i[r][0],PrD);
              Power(bi,rInv,iSqr,PrD);
              b[r][Rg.m()-1+i]=bi;
              b[r][Rg.m()-1-i]=bi;
            }
       }

      if (twop>0)
        { two_root.resize(2);
          two_root[0]=Find_Primitive_Root_2power(twop,PrD);
          Inv(two_root[1],two_root[0],PrD);

          for (int r=0; r<2; r++)
            { FFT_Iter(b[r],twop,two_root[0],PrD);  }
        }
    }
}

void FFT_Data::compute_roots(int n)
{
  roots.resize(n + 1);
  assignOne(roots[0], prData);
  for (int i = 1; i < n + 1; i++)
    Mul(roots[i], roots[i - 1], root[0], prData);
}


void FFT_Data::hash(octetStream& o) const
{
  octetStream tmp;
  pack(tmp);
  o.concat(tmp.hash());
}


void FFT_Data::pack(octetStream& o) const
{
  R.pack(o);
  prData.pack(o);
  o.store(root);
  o.store(roots);
  o.store(twop);
  o.store(two_root);
  o.store(b);
  iphi.pack(o);
  o.store(powers);
  o.store(powers_i);
}


void FFT_Data::unpack(octetStream& o)
{
  R.unpack(o);
  prData.unpack(o);
  o.get(root);
  o.get(roots);
  o.get(twop);
  o.get(two_root);
  o.get(b);
  iphi.unpack(o);
  o.get(powers);
  o.get(powers_i);
}

bool FFT_Data::operator!=(const FFT_Data& other) const
{
  if (R != other.R or prData != other.prData or root != other.root
      or twop != other.twop or two_root != other.two_root or b != other.b
      or iphi != other.iphi or powers != other.powers or powers_i != other.powers_i)
    {
      return true;
    }
  else
    return false;
}
