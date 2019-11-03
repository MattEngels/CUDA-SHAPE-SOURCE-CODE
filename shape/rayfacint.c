/***************************************************************************
                                                                rayfacint.c

Calculates where a ray intersects the plane of a triangular model facet.
The ray is given by a + r*u where input parameters a and u are vectors and
output parameter r is a scalar: a is the base displacement; u is a
directional vector, presumably but not necessarily normalized; and r is the
distance (positive or negative) traveled parallel to u to get from a to the
intersection point, divided by the magnitude of u.  The displacement
vectors of the facet's three vertices are input as v0, v1, and v2.  To save
time, the facet normal (not necessarily a unit normal) is also input rather
than computed from v0, v1, and v2.

Output parameters s and t are scalars that locate the intersection point
within the facet plane:

    v0 + s*(v1 - v0) + t*(v2 - v1) = a + r*u

If 0.0 <= s <= 1.0 and 0.0 <= t <= s then the intersection point falls
within the facet's boundaries.  Input parameter tol is an absolute
tolerance that is applied to the upper and lower limits of those two
inequalities, thus allowing for roundoff error in cases where the ray
passes close to the facet's edge or corner: the ray is considered to
pass through the facet if -tol <= s <= 1 + tol and -tol <= t <= s + tol.

In addition to computing r, s, and t, rayfacint returns 1 if the ray
intersects the facet plane within the facet's boundaries (to within
tolerance tol) and returns 0 otherwise.

Modified 2010 September 1 by CM:
    Switch to an algorithm based on one by D. Badouel, 1990, "An efficient
        ray-polygon intersection" in A. S. Glassner (Ed.), Graphics Gems, 
        Academic Press: San Diego, pp. 390-393 and p. 735.
    Add a new parameter -- the facet normal -- rather than wasting time
        computing it here in the routine
    Improve comments

Modified 2009 August 3 by CM:
    Added the "tol" parameter to permit a bit of roundoff slop for a ray
        that intersects a facet near its edge or corner

Modified 2008 June 8 by CM:
    Check for singular matrix (ray coplanar with facet) before calling
        Numerical Recipes ludcmp routine
***************************************************************************/

#include "head.h"

int rayfacint( double *r, double *s, double *t, double u[3], double a[3],
               double v0[3], double v1[3], double v2[3], double facetnorm[3],
               double tol)
{
  int i, i0, i1, i2;
  double ra0[3], r01[3], r12[3], n_dot_u, r0p[3], abs_nx, abs_ny, abs_nz, max_absn;

  /*  Construct vectors v0 - a, v1 - v0, and v2 - v1  */

  for (i=0; i<=2; i++) {
    ra0[i] = v0[i] - a[i];
    r01[i] = v1[i] - v0[i];
    r12[i] = v2[i] - v1[i];
  }

  /*  Quit if u is a zero-length vector or
      if u runs tangent to the facet plane  */

  n_dot_u = dot( facetnorm, u);
  if (n_dot_u == 0.0) {
    *r = *s = *t = -HUGENUMBER;
    return 0;
  }

  /*  Compute r, the distance from a to the
      intersection point divided by the magnitude of u  */

  *r = dot( facetnorm, ra0) / n_dot_u;

  /*  Construct the vector pointing from v0 to the intersection point  */

  for (i=0; i<=2; i++)
    r0p[i] = (a[i] + (*r)*u[i]) - v0[i];

  /*  Find the coordinate axis i0 along which the facet normal
      has the largest projection, then the other two axes, i1 and i2  */

  abs_nx = fabs(facetnorm[0]);
  abs_ny = fabs(facetnorm[1]);
  abs_nz = fabs(facetnorm[2]);
  i0 = 0;
  max_absn = abs_nx;
  if (abs_ny > max_absn) {
    i0 = 1;
    max_absn = abs_ny;
  }
  if (abs_nz > max_absn)
    i0 = 2;
  i1 = (i0 + 1) % 3;
  i2 = (i1 + 1) % 3;

  /*  Project all vectors into the i1-i2 plane -- guaranteeing that the facet's
      projection isn't a straight line -- and solve for s and t, the parameters
      that locate the intersection point within the facet plane relative to v0:
      P = v0 + s*(v1 - v0) + t*(v2 - v1)                                         */

  if (r01[i1] == 0.0) {
      *t = r0p[i1]/r12[i1];
      *s = (r0p[i2] - (*t)*r12[i2]) / r01[i2];
  } else {
      *t = (r0p[i2]*r01[i1] - r0p[i1]*r01[i2]) / (r12[i2]*r01[i1] - r12[i1]*r01[i2]);
      *s = (r0p[i1] - (*t)*r12[i1]) / r01[i1];
  }

  /*  Check if the intersection between the ray and the facet plane falls
      within the facet boundaries (to within the specified tolerance); absent
      roundoff error, the criteria would be 0.0 <= s <= 1.0 and 0.0 <= t <= s  */

  if (((*s) >= -tol) && ((*s) <= 1.0 + tol))
    if (((*t) >= -tol) && ((*t) <= (*s) + tol))
      return 1;
  return 0;
}
