/*****************************************************************************************
                                                                                dopoffs.c
Takes the delay-correction polynomial for a Doppler dataset and figures out the COM
Doppler corrections (in units of Doppler bins) for each frame.
Modified 2015 June 3 by CM:
    Implement smearing for the "fit" and "write" actions
Modified 2006 June 21 by CM:
    Changed dopres to dop_per_bin
Modified 2003 April 26 by CM:
    Removed delay computation
*****************************************************************************************/
extern "C" {
#include "../shape/head.h"
}
__global__ void dopoffs_krnl(struct dat_t *ddat, int s, int nframes) {
	/* nframes-threaded kernel */
	int f = blockIdx.x * blockDim.x + threadIdx.x;
	int k, n;
	double dop, arg, x;

	if (f < nframes) {
		for (k=0; k<ddat->set[s].desc.doppler.nviews; k++) {
			x = 1.0;
			dop = 0.0;
			arg = ddat->set[s].desc.doppler.frame[f].view[k].t -
					ddat->set[s].desc.doppler.delcor.t0;
			for (n=1; n<=ddat->set[s].desc.doppler.delcor.n; n++) {
				dop += n*ddat->set[s].desc.doppler.delcor.a[n].val*x;

				x *= arg;
			}

			/*  dop has units of usec/day and there are 86400 sec/day  */
			ddat->set[s].desc.doppler.frame[f].view[k].dopoff =
					-dop*ddat->set[s].desc.doppler.Ftx
					/ (ddat->set[s].desc.doppler.dop_per_bin*86400.0);
		}
	}
}

__host__ void dopoffs_gpu(struct dat_t *ddat, int s, int nframes)
{
  dim3 BLK,THD;

  /* Launch nframes-threaded kernel */
  THD.x = nframes;
  dopoffs_krnl<<<1,THD>>>(ddat, s, nframes);
  checkErrorAfterKernelLaunch("dopoffs_krnl");
}
