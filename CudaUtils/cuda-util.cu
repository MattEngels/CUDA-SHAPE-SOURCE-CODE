extern "C" {
#include "../shape/head.h"
}

void checkErrorAfterKernelLaunch(const char *location) {
	cudaError_t cudaStatus;
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed in %s: %s\n", location, cudaGetErrorString(cudaStatus));
	}
}
void deviceSyncAfterKernelLaunch(const char *location) {
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching the kernel in %s.\n", cudaStatus, location);
}
__host__ void gpuAssert(cudaError_t code, const char *file, int line)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		exit(code);
	}
}
void pickGPU(int gpuid) {
	cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(gpuid);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
}
