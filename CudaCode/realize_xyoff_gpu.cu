/***************************************************************************
                                                            realize_xyoff.c

Implements the '=' state for horizontal and vertical offsets for
plane-of-sky frames (optical images):

For each plane-of-sky frame whose horizontal offset has the '=' state, go
backwards within that dataset until we find a frame whose horizontal
offset has state 'f' or 'c', and copy this value.  Do the same for
vertical offsets.  It's legal for a frame to use the '=' state for just
one of its two offsets.

There is no way to use the '=' state to copy offsets from a frame in one
plane-of-sky dataset to a frame in another plane-of-sky dataset.

Modified 2016 July 7 by Matt Engels:
	Adapted for use in shape-cuda.

Written 2005 February 24 by CM
 ***************************************************************************/
extern "C" {
#include "../shape/head.h"
}

__global__ void realize_xyoff_gpu_krnl(struct dat_t *ddat, int nsets,
		unsigned char *type) {
	/* nset-threaded kernel */
	int s = blockIdx.x * blockDim.x + threadIdx.x;
	int j, f;
	if (s < nsets) {
		if (type[s] == POS) {

			for (j=0; j<=1; j++)
				if (ddat->set[s].desc.poset.frame[0].off[j].state == '=')
					printf("can't use \"=\" state for the first frame in a plane-of-sky dataset\n");

			for (f=1; f<ddat->set[s].desc.poset.nframes; f++)
				for (j=0; j<=1; j++)
					if (ddat->set[s].desc.poset.frame[f].off[j].state == '=')
						ddat->set[s].desc.poset.frame[f].off[j].val =
								ddat->set[s].desc.poset.frame[f-1].off[j].val;

		}
	}
}

__host__ void realize_xyoff_gpu( struct dat_t *ddat, int nsets,
		unsigned char *dtype)
{
	dim3 BLK,THD64;
	THD64.x = 64;
	BLK.x = floor((THD64.x - 1 + nsets)/THD64.x);

	/* Launch nset-threaded kernel */
	realize_xyoff_gpu_krnl<<<BLK,THD64>>>(ddat, nsets, dtype);
	checkErrorAfterKernelLaunch("realize_xyoff_cuda_streams_krnl (realize_xyoff_cuda_streams");

}
