/*  Modified 2005 June 27 by CM:
        Rename "round" function to "iround" to avoid conflicts  */

#include "basic.h"
#include "util.h"

void dms( double deg, int *dd, int *mm, int *ss)
{
  int sgn=1;
  
  if (deg < 0.0) {
    sgn = -1;
    deg = -deg;
  }
  *dd = (int)floor(deg);
  *mm = (int)floor((deg - (*dd))*60);
  *ss = iround((((deg - (*dd))*60) - (*mm))*60);
  *dd *= sgn;
}
