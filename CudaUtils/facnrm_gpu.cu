extern "C" {
#include "../shape/head.h"
}
__device__ double dev_facnrm( struct vertices_t verts, int fi)
{
	int i;
	double a[3], b[3], area;

	for (i=0; i<=2; i++) {
		a[i] = verts.v[verts.f[fi].v[1]].x[i] - verts.v[verts.f[fi].v[0]].x[i];
		b[i] = verts.v[verts.f[fi].v[2]].x[i] - verts.v[verts.f[fi].v[1]].x[i];
	}
	area = 0.5*dev_cross( verts.f[fi].n, a, b);
	dev_normalize( verts.f[fi].n);
	return area;
}
