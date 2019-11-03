/***************************************************************************

								 facnorm.c

generates a facet's normal from it's 3 vertices.  Returns the surface area
of the facet.

***************************************************************************/

#include "basic.h"
#include "util.h"

double facnorm( double v0[3], double v1[3], double v2[3], double n[3])
{
  int i;
  double a[3], b[3], area;

  for (i=0;i<=2;i++) {
	a[i] = v1[i]-v0[i];
	b[i] = v2[i]-v1[i];
  }
  area = 0.5*cross( n, a, b);
  normalize( n);
  return area;
}
