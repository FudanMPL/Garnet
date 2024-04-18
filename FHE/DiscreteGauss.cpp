
#include "DiscreteGauss.h"
#include "math.h"

void DiscreteGauss::set(double RR)
{
  if (RR > 0 or NewHopeB < 1)
    NewHopeB = max(1, int(round(2 * RR * RR)));
  assert(NewHopeB > 0);
}



/* This uses the approximation to a Gaussian via
 * binomial distribution
 *
 * This procedure consumes 2*NewHopeB bits
 *
 */
int DiscreteGauss::sample(PRNG &G, int stretch) const
{
  int s= 0;
  // stretch refers to the standard deviation
  int B = NewHopeB * stretch * stretch;
  for (int i = 0; i < B; i++)
    {
      s += G.get_bit();
      s -= G.get_bit();
    }
  return s;
}




int sample_half(PRNG& G)
{
  int v=G.get_uchar()&3;
  if (v==0 || v==1)
    return 0;
  else if (v==2)
    return 1;
  else
    return -1;
}


bool DiscreteGauss::operator!=(const DiscreteGauss& other) const
{
  if (other.NewHopeB != NewHopeB)
    return true;
  else
    return false;
}
