/***************************************************************************

								 resamp.c

Resamples a vector using splines.  Input v1[1..N1] output v2[1..N2].

***************************************************************************/

#include "basic.h"
#include "../nr/nr.h"


void resamp( double *v1, int N1, double *v2, int N2)
{
  double *x, *y2;
  int i;

  x = vector( 1, N1);
  y2 = vector( 1, N1);
  for (i=1;i<=N1;i++)
	x[i] = (i-1.0)/(N1-1.0);
  spline( x, v1, N1, 2.0e30, 2.0e30, y2);
  for (i=1;i<=N2;i++) {
	splint( x, v1, y2, N1, (i-1.0)/(N2-1.0), &v2[i]);
  }
  free_vector( y2, 1, N1);
  free_vector( x, 1, N1);
}
