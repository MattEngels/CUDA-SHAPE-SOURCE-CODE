/***************************************************************************
                                                                 facnorm.c

Generates a facet's normal from its 3 vertices.  Returns the surface area
of the facet.

Modified 2005 January 25 by CM:
    Eliminate an unused variable
***************************************************************************/

#include "head.h"

double facnrm( struct vertices_t verts, int fi)
{
  int i;
  double a[3], b[3], area;

  for (i=0; i<=2; i++) {
    a[i] = verts.v[verts.f[fi].v[1]].x[i] - verts.v[verts.f[fi].v[0]].x[i];
    b[i] = verts.v[verts.f[fi].v[2]].x[i] - verts.v[verts.f[fi].v[1]].x[i];
  }
  area = 0.5*cross( verts.f[fi].n, a, b);
  normalize( verts.f[fi].n);
  return area;
}
