extern "C" {
#include "../shape/head.h"
}
__device__ double dev_dot( double x[3], double y[3])
{
	return x[0]*y[0]+x[1]*y[1]+x[2]*y[2];
}

__device__ float dev_dot_d3(double3 x, double3 y)
{
	/* This version just uses two float3's and returns a float */
	return x.x*y.x + x.y*y.y + x.z*y.z;
}
