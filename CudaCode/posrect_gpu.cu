extern "C" {
#include "../shape/head.h"
#include <limits.h>
}

__device__ static float atomicMin64(double* address, double val)
{
	unsigned long long* address_as_i = (unsigned long long*) address;
	unsigned long long old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
				__double_as_longlong(::fminf(val, __longlong_as_double(assumed))));
	} while (assumed != old);
	return __longlong_as_double(old);
}
__device__ static float atomicMax64(double* address, double val)
{
	unsigned long long* address_as_i = (unsigned long long*) address;
	unsigned long long old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
				__double_as_longlong(::fmaxf(val, __longlong_as_double(assumed))));
	} while (assumed != old);
	return __longlong_as_double(old);
}

__device__ void dev_POSrect_gpu64_shared(
		double imin_dbl,
		double imax_dbl,
		double jmin_dbl,
		double jmax_dbl,
		double4 *ijminmax_overall_sh,
		int n)	{

	/* Update the POS region that contains the target without
	 * regard to whether or not it extends beyond the POS frame */
	atomicMin64(&ijminmax_overall_sh->w, imin_dbl);
	atomicMax64(&ijminmax_overall_sh->x, imax_dbl);
	atomicMin64(&ijminmax_overall_sh->y, jmin_dbl);
	atomicMax64(&ijminmax_overall_sh->z, jmax_dbl);
}
