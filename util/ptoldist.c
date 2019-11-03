/***************************************************************************

								ptofdist.c

Given a point and a line segment defined by 2 vertices, this routine
returns the minimum distance between the point and the line.  It also sets
a vector argument (x) to the value of the point of minimum distance.

***************************************************************************/

#include "basic.h"
#include "util.h"

double ptoldist( double p[3], double v0[3], double v1[3], double x[3])
{
  int i;
  double u[3], v[3];
  double u2, v2, uv, s;

  for (i=0;i<=2;i++) {
	u[i] = v0[i]-p[i];
	v[i] = v1[i]-v0[i];
  }
  u2 = dot(u,u);
  v2 = dot(v,v);
  uv = dot(u,v);
  s = (uv-u2)/(u2+v2-2*uv);
  if (s<0.0)
	s = 0.0;
  if (s>1.0)
	s = 1.0;
  for (i=0;i<=2;i++)
	x[i] = v0[i]+s*(v1[i]-v0[i]);
  return distance( x, p);
}
