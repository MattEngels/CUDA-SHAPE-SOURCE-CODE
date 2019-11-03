extern "C" {
#include "../shape/head.h"
}

__global__ void clrvect_krnl(struct dat_t *ddat, int size, int s, int f) {
	/* Multi-threaded kernel */
	int idel, idop, offset = blockIdx.x * blockDim.x + threadIdx.x;

	if (offset < size) {
		if (ddat->set[s].type == DELAY) {
			idel = offset % ddat->set[s].desc.deldop.frame[f].ndel + 1;
			idop = offset / ddat->set[s].desc.deldop.frame[f].ndel + 1;
			ddat->set[s].desc.deldop.frame[f].fit[idel][idop] = 0.0;
		}
		if (ddat->set[s].type == DOPPLER) {
			ddat->set[s].desc.doppler.frame[f].fit[offset+1] = 0.0;
		}
	}
}

__global__ void clrvect_krnl_af(struct dat_t *ddat, int *size, int s, int blocks) {
	/* Multi-threaded kernel */
	/* This version limits itself to the window defined by frame[f]->idellim
	 * and frame[f]->idoplim.  It also uses shared memory for variables used
	 * by each thread.  Additionally, all frames are handled by one kernel
	 * call.  Each frame gets a configurable number of blocks within which to
	 * do a grid-stride loop through all delay-Doppler space cells within the
	 * window
	 */
	int f = blockIdx.x/blocks;
	int idel, idop, offset = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ int ndel, fr;

	if (threadIdx.x==0) {
		if (ddat->set[s].type ==DELAY) {
			ndel = ddat->set[s].desc.deldop.frame[f].ndel;
			fr = (blockIdx.x%blocks) * blockDim.x;
		}
	}

	__syncthreads();

	for (offset=threadIdx.x+fr; offset<size[f]; offset+=(blocks*blockDim.x)) {
		if (ddat->set[s].type == DELAY) {

				idel = offset % ndel + 1;
				idop = offset / ndel + 1;
				ddat->set[s].desc.deldop.frame[f].fit[idel][idop] = 0.0;

		}
		if (ddat->set[s].type == DOPPLER) {
				ddat->set[s].desc.doppler.frame[f].fit[offset+1] = 0.0;
		}
	}
}
