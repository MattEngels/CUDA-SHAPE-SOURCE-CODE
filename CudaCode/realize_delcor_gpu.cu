/***************************************************************************
                                                           realize_delcor.c

Implements the '=' state for delay correction polynomial coefficients:

For each delay-Doppler or Doppler dataset whose coefficients have the
'=' state, go backwards in the datafile until we find a delay-Doppler
or Doppler dataset whose coefficients have state 'f' and/or 'c', and
copy its coefficient values.

Setting zeroth-order coefficients is more complicated than setting other
coefficients, since the "vary_delcor0" fitting parameter, which permits
zeroth-order coefficients to be varied jointly with shape/spin parameters
that are being fit, might be turned on.  The "delcor0_mode" parameter to
realize_delcor determines how zeroth-order coefficients are handled:

    delcor0_mode = 0:
        Zeroth-order coefficients are not being varied jointly with
        shape/spin parameters, or else we aren't fitting a shape/spin
        parameter right now; just update each dataset's delcor0_save by
        setting it equal to that dataset's zeroth-order coefficient,
        in case joint variation is needed later in the fit

    delcor0_mode = 1:
        Zeroth-order coefficients are being varied jointly with shape/spin
        parameters, and we're in the process of fitting some shape/spin
        parameter p (i.e., we're realizing the model for a trial value
        of p); set each dataset's zeroth-order coefficient equal to the sum
        of the corresponding delcor0_save and "delta_delcor0"

    delcor0_mode = 2:
        Zeroth-order coefficients are being varied jointly with shape/spin
        parameters, we've just obtained the best-fit value for shape/spin
        parameter p, and now we need to set the zeroth-order coefficients
        to their best-fit values (i.e., to the values which "go with" the
        best-fit value of p); set each dataset's zeroth-order coefficient
        equal to the sum of the corresponding delcor0_save and
        "delta_delcor0," then update delcor0_save by setting it equal to
        this same sum

Written 2003 April 23 by CM

Modified 2006 October 1 by CM:
    Add "delta_delcor0" and "delcor0_mode" arguments
    Implement the "vary_delcor0" parameter
 ***************************************************************************/
extern "C" {
#include "../shape/head.h"
}

__device__ unsigned char rd_dat_type;
__device__ int s_delcor, type_delcor, n_delcor;
__device__ double t0_delcor;

__global__ void rd_init_flags_krnl() {
	/* Single-threaded kernel */
	if (threadIdx.x == 0) {
		s_delcor = -1;
		type_delcor = -1;
		n_delcor = -1;
		t0_delcor = -1.0;
	}
}
__global__ void rd_get_dat_type_krnl(struct dat_t *ddat, int s) {
	/* Single-threaded kernel */
	if (threadIdx.x == 0)
		rd_dat_type = ddat->set[s].type;
}
__global__ void rd_deldop_krnl(struct dat_t *ddat, int s,
		double delta_delcor0, int delcor0_mode) {
	/* Single-threaded kernel */
	int i, n;
	double t0;

	if (threadIdx.x == 0) {
		n = ddat->set[s].desc.deldop.delcor.n;
		t0 = ddat->set[s].desc.deldop.delcor.t0;

		if (ddat->set[s].desc.deldop.delcor.a[0].state != '=') {
			s_delcor = s;
			type_delcor = DELAY;
			n_delcor = n;
			t0_delcor = t0;
			if (ddat->set[s].desc.deldop.delcor.a[0].state == 'f') {
				if (delcor0_mode != 0)
					ddat->set[s].desc.deldop.delcor.a[0].val =
							ddat->set[s].desc.deldop.delcor.delcor0_save + delta_delcor0;
				if (delcor0_mode != 1)
					ddat->set[s].desc.deldop.delcor.delcor0_save =
							ddat->set[s].desc.deldop.delcor.a[0].val;
			}
		} else if (s_delcor < 0)
			printf("can't use \"=\" state for the first delay polynomial\n");
		else if (n != n_delcor)
			printf("delay polynomials must have same degree if state = \"=\"\n");
		else if (fabs(t0 - t0_delcor) > HALFSECOND)
			printf("delay polynomials must have same t0 if state = \"=\"\n");
		else if (type_delcor == DELAY) {
			ddat->set[s].desc.deldop.delcor.t0 = t0_delcor;
			for (i=0; i<=n; i++)
				ddat->set[s].desc.deldop.delcor.a[i].val =
						ddat->set[s_delcor].desc.deldop.delcor.a[i].val;
		} else {
			ddat->set[s].desc.deldop.delcor.t0 = t0_delcor;
			for (i=0; i<=n; i++)
				ddat->set[s].desc.deldop.delcor.a[i].val =
						ddat->set[s_delcor].desc.doppler.delcor.a[i].val;
		}
	}
}
__global__ void rd_doppler_krnl(struct dat_t *ddat, int s,
		double delta_delcor0, int delcor0_mode) {
	/* Single-threaded kernel */
	int i, n;
	double t0;

	if (threadIdx.x == 0) {
		n = ddat->set[s].desc.doppler.delcor.n;
		t0 = ddat->set[s].desc.doppler.delcor.t0;

		if (ddat->set[s].desc.doppler.delcor.a[0].state != '=') {
			s_delcor = s;
			type_delcor = DOPPLER;
			n_delcor = n;
			t0_delcor = t0;
		} else if (s_delcor < 0)
			printf("can't use \"=\" state for the first delay polynomial\n");
		else if (n != n_delcor)
			printf("delay polynomials must have same degree if state = \"=\"\n");
		else if (fabs(t0 - t0_delcor) > HALFSECOND)
			printf("delay polynomials must have same t0 if state = \"=\"\n");
		else if (type_delcor == DELAY) {
			ddat->set[s].desc.deldop.delcor.t0 = t0_delcor;
			for (i=0; i<=n; i++)
				ddat->set[s].desc.doppler.delcor.a[i].val =
						ddat->set[s_delcor].desc.deldop.delcor.a[i].val;
		} else {
			ddat->set[s].desc.deldop.delcor.t0 = t0_delcor;
			for (i=0; i<=n; i++)
				ddat->set[s].desc.doppler.delcor.a[i].val =
						ddat->set[s_delcor].desc.doppler.delcor.a[i].val;
		}
	}
}

__host__ void realize_delcor_gpu(struct dat_t *ddat, double delta_delcor0, int delcor0_mode,
		int nsets, int *nframes)
{
	int s;
	unsigned char type;

	/* If a dataset has delay correction polynomial coefficients with state '=',
	 * go backwards in the datafile until we reach a delay-Doppler or Doppler
	 * dataset whose polynomial coefficients have states 'f' and/or 'c' rather
	 * than '='.
	 * s_delcor is # of the dataset we find,
	 * type_delcor is either delay-Doppler or Doppler,
	 * n_delcor is # of coefficients in that dataset's polynomial,
	 * t0_delcor is the reference epoch for the polynomial.  */

	/* Initialize the flags */
	rd_init_flags_krnl<<<1,1>>>();
	checkErrorAfterKernelLaunch("rd_init_flags_krnl");

	for (s=0; s<nsets; s++) {

		/* Get type */
		rd_get_dat_type_krnl<<<1,1>>>(ddat, s);
		checkErrorAfterKernelLaunch("rd_get_dat_type_krnl (realize_delcor_cuda)");
		gpuErrchk(cudaMemcpyFromSymbol(&type, rd_dat_type,
				sizeof(unsigned char),	0, cudaMemcpyDeviceToHost));

		if (type == DELAY) {

			/* Launch the delay-doppler kernel for realize_delcor_cuda */
			rd_deldop_krnl<<<1,1>>>(ddat, s, delta_delcor0, delcor0_mode);
			checkErrorAfterKernelLaunch("rd_deldop_krnl (realize_delcor_cuda)");

			/* Compute frame[*].deloff and frame[*].dopoff for delay-Doppler
            dataset: float # of delay and Doppler bins corresponding to delay
            correction polynomial at the epoch of each data frame.*/
			deldopoffs_gpu(ddat, s, nframes[s]);

		} else if (type == DOPPLER) {

			/* Launch the Doppler kernel for realize_delcor_cuda */
			rd_doppler_krnl<<<1,1>>>(ddat, s, delta_delcor0, delcor0_mode);
			checkErrorAfterKernelLaunch("rd_doppler_krnl (realize_delcor_cuda)");

			/*  Compute frame[*].dopoff for this Doppler dataset:
            the (floating-point) number of Doppler bins corresponding to
            the delay correction polynomial at the epoch of each data frame.  */
			dopoffs_gpu(ddat, s, nframes[s]);
		}
	}
}

