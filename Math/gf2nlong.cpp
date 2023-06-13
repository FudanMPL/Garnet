/*
 * gf2n_longlong.cpp
 *
 */

#include "gf2nlong.h"
#include "gf2n.h"

#include "Tools/intrinsics.h"
#include "Tools/Exceptions.h"

#include <stdint.h>


bool is_ge(__m128i a, __m128i b)
{
  word aa[2], bb[2];
  _mm_storeu_si128((__m128i*)aa, a);
  _mm_storeu_si128((__m128i*)bb, b);
//  cout << hex << "is_ge " << aa[1] << " " << bb[1] << " " << (aa[1] > bb[1]) << " ";
//  cout << aa[0] << " " << bb[0] << " " << (aa[0] >= bb[0]) << endl;
  return aa[1] == bb[1] ? aa[0] >= bb[0] : aa[1] > bb[1];
}


ostream& operator<<(ostream& s, const int128& a)
{
  word* tmp = (word*)&a.a;
  s << hex;

  if (tmp[1] != 0)
    {
      s << noshowbase;
      s.width(16);
      s.fill('0');
      s << tmp[1];
      s.width(16);
    }
  else
    s << showbase;

  s << tmp[0] << dec;
  return s;
}

istream& operator>>(istream& s, int128& a)
{
  gf2n_long tmp;
  s >> tmp;
  a = tmp.get();
  return s;
}
