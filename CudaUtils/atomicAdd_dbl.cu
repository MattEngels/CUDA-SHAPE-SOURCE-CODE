extern "C" {
#include "../shape/head.h"
}
/* AtomicAdd(*double, double) implementation for architectures <3.5 */
__device__ double atomicAdd_dbl(double* address, double val) {
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val+__longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}
