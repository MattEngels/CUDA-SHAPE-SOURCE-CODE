/*****************************************************************************************
                                                                             deldopoffs.c
Takes the delay-correction polynomial for a delay-doppler set and figures out the COM
delay and doppler corrections (in units of image rows and columns) for each frame.
Modified 2015 June 3 by CM:
    Implement smearing for the "fit" and "write" actions
Modified 2006 June 21 by CM:
    Changed delres to del_per_pixel and dopres to dop_per_pixel
 *****************************************************************************************/
extern "C" {
#include "../shape/head.h"
}
__global__ void deldopoffs_krnl(struct dat_t *ddat, int s, int nframes) {
	/* nframes-threaded kernel */
	int f = blockIdx.x * blockDim.x + threadIdx.x;
	int k, n;
	double del, dop, arg, x;

	if (f < nframes) {
		for (k=0; k<ddat->set[s].desc.deldop.nviews; k++) {
			x = 1.0;
			dop = 0.0;
			del = ddat->set[s].desc.deldop.delcor.a[0].val;
			arg = ddat->set[s].desc.deldop.frame[f].view[k].t -
					ddat->set[s].desc.deldop.delcor.t0;

			for (n=1; n<=ddat->set[s].desc.deldop.delcor.n; n++) {
				dop += n*ddat->set[s].desc.deldop.delcor.a[n].val*x;
				del += ddat->set[s].desc.deldop.delcor.a[n].val*(x*=arg);
			}

			/* del has units of usec */
			ddat->set[s].desc.deldop.frame[f].view[k].deloff =
					del/ddat->set[s].desc.deldop.del_per_pixel;

			/* dop has units of usec/day and there are 86400 sec/day */
			ddat->set[s].desc.deldop.frame[f].view[k].dopoff =
					-dop*ddat->set[s].desc.deldop.Ftx
					/ (ddat->set[s].desc.deldop.dop_per_pixel*86400.0);
		}
	}
}
__host__ void deldopoffs_gpu(struct dat_t *ddat, int s, int nframes)
{
	dim3 BLK,THD;

	/* Launch nframes-threaded kernel */
	THD.x = nframes;
	deldopoffs_krnl<<<BLK,THD>>>(ddat, s, nframes);
	checkErrorAfterKernelLaunch("deldopoffs_cuda_krnl (deldopoffs_cuda)");
}
