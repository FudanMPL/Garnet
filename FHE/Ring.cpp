
#include "Ring.h"
#include "Tools/Exceptions.h"

void Ring::pack(octetStream& o) const
{
  o.store(mm);
  if (((mm - 1) & mm) != 0)
    {
      o.store(phim);
      o.store(pi);
      o.store(pi_inv);
      o.store(poly);
    }
}

void Ring::unpack(octetStream& o)
{
  o.get(mm);
  if (((mm - 1) & mm) != 0)
    {
      o.get(phim);
      o.get(pi);
      o.get(pi_inv);
      o.get(poly);
    }
  else if (mm != 0)
    init(*this, mm);
}

bool Ring::operator !=(const Ring& other) const
{
  if (mm != other.mm or phim != other.phim or pi != other.pi
      or pi_inv != other.pi_inv or poly != other.poly)
    return true;
  else
    return false;
}
