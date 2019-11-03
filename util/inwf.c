/***************************************************************************

inwf.c

given a wavefront type shape described by vertices v[1:nv][1:3] and facets
f[1:nf][1:3] this subroutine returns a 1 if the point x,y,z is inside the
shape and 0 if it is not.

Modified 2005 January 25 by CM:
    Take care of unused and uninitialized variables
***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "util.h"

int inwf( double **v, int **f, int nv, int nf, double *x)
{
  int fi, i, fmin;
  double cen[3], dmin=1.0e20, d, a[3], b[3], n[3];

  /*  Initialize variable to avoid compilation warnings  */

  fmin = 0;

  /* find nearest facet center */
  for (fi=1; fi<=nf; fi++) {
    for (i=0; i<=2; i++)
    cen[i] = (v[f[fi][1]][i+1] + v[f[fi][2]][i+1] + v[f[fi][3]][i+1])/3.0;
    if ((d = distance( x, cen)) < dmin) {
      dmin = d;
      fmin = fi;
    }
  }
  for (i=0; i<=2; i++) {
    cen[i] = x[i] - v[f[fmin][1]][i+1];
    a[i] = v[f[fmin][2]][i+1] - v[f[fmin][1]][i+1];
    b[i] = v[f[fmin][3]][i+1] - v[f[fmin][2]][i+1];
  }
  cross( n, a, b);
  if (dot( cen, n) > 0.0)
    return 0;
  else
    return 1;
}
