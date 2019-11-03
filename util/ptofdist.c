/***************************************************************************

								ptofdist.c

Given a point and a triangular facet defined by 3 vertices, this routine
returns the minimum distance between the point and the facet.  It also sets
a vector argument (x) to the value of the point of minimum distance.

***************************************************************************/

#include "basic.h"
#include "util.h"

double ptofdist( double p[3], double v0[3], double v1[3], double v2[3],
				double x[3])
{
  int i, j;
  double v[3][3], a, b, c, d, e, f, s, t, dp[3][3];
  double d1, d2, d3;

  /* translate origin to point p */
  for (j=0;j<=2;j++) {
	v[0][j] = v0[j]-p[j];
	v[1][j] = v1[j]-p[j];
	v[2][j] = v2[j]-p[j];
  }
  for (i=0;i<=2;i++)
	for (j=0;j<=2;j++)
	  dp[i][j] = dot( v[i], v[j]);
  /* form coefficients of quadratic form */
  a = dp[0][0];
  b = 2*(dp[0][1]-dp[0][0]);
  c = 2*(dp[0][2]-dp[0][1]);
  d = 2*(dp[1][2]+dp[0][1]-dp[0][2]-dp[1][1]);
  e = dp[1][1]+dp[0][0]-dp[0][1]-dp[1][0];
  f = dp[2][2]+dp[1][1]-dp[1][2]-dp[2][1];
  /* solve for point of min dist */
  s = (2*f*b-c*d)/(d*d-4*f*e);
  t = (2*e*c-b*d)/(d*d-4*f*e);
  if ((s>=0.0)&&(s<=1.0)&&(t>=0.0)&&(t<=s)) {
	for (i=0;i<=2;i++)
	  x[i] = v0[i]+s*(v1[i]-v0[i])+t*(v2[i]-v1[i]);
	return distance( x, p);
  }
  d1 = ptoldist( p, v0, v1, x);
  d2 = ptoldist( p, v1, v2, x);
  d3 = ptoldist( p, v2, v0, x);
  if ((d1<=d2)&&(d1<=d3))
	return ptoldist( p, v0, v1, x);
  if ((d2<=d1)&&(d2<=d3))
	return ptoldist( p, v1, v2, x);
  return ptoldist( p, v2, v0, x);
}
