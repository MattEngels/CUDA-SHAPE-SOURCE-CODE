/***************************************************************************

                                                              inter_facet.c

Returns 1 if a ray (a+ru) intersects the facet f, 0 otherwise.  It is
assumed that u[3] is a unit vector.  If case "1" then the scalar distance r
is set and the vector n[3] gives the surface normal at the point of
intersection.

Modified 2005 January 25 by CM:
    Removed unused variable
***************************************************************************/

#include "basic.h"
#include "util.h"

#define TINY 1.0e-6

int intfac( double u[3], double r0[3],
		   double v0[3], double v1[3], double v2[3],
		   double *r, double n[3], double x[3])
{
  int j;
  double v[3][3], s, t, num=0.0, den=0.0, a, b, c, e, f, del[3], tmp;
  double v10[3], v21[3];

  for (j=0; j<=2; j++) {
    v[0][j] = v0[j];
    v[1][j] = v1[j];
    v[2][j] = v2[j];
    v10[j] = v[1][j] - v[0][j];
    v21[j] = v[2][j] - v[1][j];
  }
  cross( n, v10, v21);
  normalize( n);
  for (j=0; j<=2; j++) {
    num += (v[0][j] - r0[j])*n[j];
    den += u[j]*n[j];
  }
  if (fabs(den) < TINY)			/* ray paralell to facet */
    return 0;
  (*r) = num/den;
  a = b = c = e = f = 0.0;
  for (j=0; j<=2; j++)
    del[j] = r0[j] + (*r)*u[j] - v[0][j];
  for (j=0; j<=2; j++) {
    a += (v[1][j] - v[0][j])*(v[1][j] - v[0][j]);
    b += (v[2][j] - v[1][j])*(v[2][j] - v[1][j]);
    c += (v[1][j] - v[0][j])*(v[2][j] - v[1][j]);
    e += (v[1][j] - v[0][j])*del[j];
    f += (v[2][j] - v[1][j])*del[j];
  }
  s = (e*b - c*f)*(tmp = (1/(a*b - c*c)));
  t = (a*f - c*e)*tmp;
/*  printf("%d %f %f %f\n", fi, *r, s, t); */
  if ( 0 <= s && s <= 1 && 0 <= t && t <= s ) {
      for (j=0; j<=1; j++)
        x[j] = r0[j] + (*r)*u[j];
      return 1;
  } else
      return 0;
}

#undef TINY
