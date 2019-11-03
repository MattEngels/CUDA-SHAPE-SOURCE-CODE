/*______________________________________________________________________________*
 * This file implements the functions that check for presence of a CUDA-capable *
 * device in the system and enforces certain minimum requirements.				*
 * 																				*
 * Written: Thursday, July 7, 2016 by Matt Engels								*
 *______________________________________________________________________________*/
extern "C" {
#include "../shape/head.h"
}

void CUDACount(int showCUDAInfo) {
	int nDevices, i;

	/* Get number of CUDA devices in system and make sure it's > 0  */
	cudaGetDeviceCount(&nDevices);
	if (nDevices < 1)
		bailout("No CUDA-capable GPU detected.  Exiting shape-cuda.");
	if (nDevices >= 1) {
		printf("%i CUDA-capable GPU(s) detected.  shape-cuda will use device id %i.\n\n", nDevices, GPU0);
		gpuErrchk(cudaSetDevice(GPU0));
	}

	for (i = 0; i < nDevices; i++) {
#ifdef __cplusplus
		cudaDeviceProp prop;
#else
		struct cudaDeviceProp prop;
#endif
		cudaGetDeviceProperties(&prop, i);
		if(showCUDAInfo){
			printf("Device Number: %d\n", i);

			printf("  Device name: %s\n", prop.name);
			printf("  Memory Clock Rate (GHz): %f\n",
					prop.memoryClockRate/1e6);
			printf("  Memory Bus Width (bits): %d\n",
					prop.memoryBusWidth);
			printf("  Peak Memory Bandwidth (GB/s): %f\n",
					2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
			printf("  Compute Capability: %i.%i\n", prop.major, prop.minor);

			if ((i == 0 && prop.major < 3) || (i == 0 && prop.major == 3 && prop.minor < 5))
				bailout("Minimum CUDA-Compute Capability 3.5 not met. Exiting.");

			printf("  Number of multi-processors on GPU: %i\n", prop.multiProcessorCount);
			printf("  Maximum grid size: %i x %i x %i\n", prop.maxGridSize[0], prop.maxGridSize[1],
					prop.maxGridSize[2]);
			printf("  Maximum size of each dimension of a block: %i x %i x %i\n",
					prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
			printf("  Maximum threads per block: %i\n", prop.maxThreadsPerBlock);
			printf("  Maximum shared memory per block: %lu\n", (unsigned long)prop.sharedMemPerBlock);

			/*	This is an extern global, used for kernel launch parameter calculations. */
			maxThreadsPerBlock = prop.maxThreadsPerBlock;

			printf("  Warp size: %i\n", prop.warpSize);

			if (prop.integrated){
				printf("WARNING: detected GPU is integrated ('onboard').\n");
				printf("shape-cuda will perform slower than with a standard discrete GPU.\n\n");
			}
			printf("\n");
			printf("  concurrentManagedAccess = %i\n", prop.concurrentManagedAccess);
		}
	}
}
