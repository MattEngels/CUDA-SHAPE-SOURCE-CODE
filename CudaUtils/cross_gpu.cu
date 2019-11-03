extern "C" {
#include "../shape/head.h"
}
__device__ double dev_cross( double z[3], double x[3], double y[3])
{
	double zz[3];

	zz[0] = x[1]*y[2]-x[2]*y[1];
	zz[1] = x[2]*y[0]-x[0]*y[2];
	zz[2] = x[0]*y[1]-x[1]*y[0];
	z[0] = zz[0];
	z[1] = zz[1];
	z[2] = zz[2];
	return sqrt(z[0]*z[0]+z[1]*z[1]+z[2]*z[2]);
}
