extern "C" {
#include "../shape/head.h"
}

__device__ double dev_normalize(double *u)
{
	int i;
	double norm;

	norm = 0.0;
	for (i=0; i<=2; i++)
		norm += u[i]*u[i];
	norm = sqrt(norm);
	if (norm != 0.0) {
		for (i=0; i<=2; i++)
			u[i] /= norm;
	}
	return norm;
}
__device__ double dev_normalize3(double3 *u)
{
	double norm;

	norm = 0.0;
	norm += u->x*u->x;
	norm += u->y*u->y;
	norm += u->z*u->z;

	norm = sqrt(norm);

	if (norm != 0.0) {
		u->x /= norm;
		u->y /= norm;
		u->z /= norm;
	}
	return norm;
}
