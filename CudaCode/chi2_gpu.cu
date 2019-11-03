/*****************************************************************************************
                                                                                   chi2.c
Compute chi-square for a model that was previously created.
Modified 2016 December 8 by ME:
	Converted to "FIT" action-only, CUDA code
Modified 2015 June 10 by CM:
    Implement smearing for the "fit" and "write" actions
Modified 2013 April 24 by CM:
    Adjust names of output files so they are in alphanumeric order if > 100 per dataset
Modified 2012 December 6 by CM:
    Fix bug introduced on December 6: take "calval" into account when determining whether
        or not model signal is greater than 'chi2fit0_thresh' sigmas
Modified 2012 December 5 by CM:
    When the "write_chi2fit0" parameter is turned on, display the number of degrees of
        freedom with zero model signal in addition to the chi2 contribution, and list
        chi2 and dof for all data combined
    Implement the 'chi2fit0_thresh' parameter
Modified 2012 March 24 by CM:
    When the root node calls chi2 with list_breakdown = 1 (e.g., for the "write" action),
        print a warning if the value of any (delay-)Doppler dataset's Doppler scaling
        factor is out of the allowed range
Modified 2010 August 25 by CM:
    Move TINYCALFACT definition to const.h
Modified 2010 August 10 by CM:
    Implement the "radfitmin" and "radobsmin" parameters: these are the
        pixel values that map to black for all fit and obs pgm images
        that are output for delay-Doppler frames
Modified 2010 March 20 by CM:
    For the "write" action for lightcurve datasets, include magnitude
        uncertainties as a new column in output files fit_MM.dat
Modified 2009 August 9 by CM:
    For the "write" action with the "listfit" parameter turned on, replace
        a ".rdf" or ".fits" or ".fit" suffix with ".fitdat"
Modified 2009 April 3 by CM:
    When the root node calls chi2 with list_breakdown = 1 (e.g., for the
        "write" action), print a warning if any plane-of-sky fit image is
        too small to "contain" all nonzero pixels in the POS sky rendering
        or if the model is too wide in delay-Doppler space for any
        (delay-)Doppler fit frame to be correctly constructed
    For MPI_Recv calls, mpi_par[0] is no longer equal to the MPI action,
        since the message tag argument already serves that purpose (as of
        2008 April 10) -- so the other mpi_par elements are renumbered
Modified 2008 June 29 by CM:
    For the "write" and "orbit" actions, zero out fit pixels/bins in
        delay-Doppler, Doppler, and plane-of-sky frames for which those
        pixels/bins have been zeroed out in the pixel mask
Modified 2008 April 10 by CM:
    Use message tag argument to MPI_Recv to identify the MPI action
Modified 2007 September 13 by CM:
    Implement "write_chi2fit0" parameter: for the "write" and "orbit"
        actions, output chi2 for delay-Doppler, Doppler, and plane-of-sky
        pixels/bins with zero model power; do this both for each individual
        frame, for all delay-Doppler data taken together, for all Doppler
        data taken together, and for all plane-of-sky data taken together
Modified 2007 August 18 by CM:
    Rename MPI_TAG to MPI_TAG_1 to avoid name conflict with mpich headers
Modified 2007 August 10 by CM:
    Create chi2_deldop, chi2_doppler, chi2_poset, and chi2_lghtcrv
        routines so that the code isn't one giant switch statement
    When a calfact value is negative, reset it to a tiny positive value
        rather than to zero in order to protect against division by zero
    Implement the "radfitmax" and "radobsmax" parameters for applying the
        same brightness scale to all fit and obs pgm images output for
        delay-Doppler frames
Modified 2007 August 4 by CM:
    Carry out "write" action steps for the "orbit" action as well
Modified 2007 January 6 by CM:
    For the "write" action for lightcurve datasets, output rotation phases
        to files fit_MM.dat and calc_MM.dat.
Modified 2006 October 1 by CM:
    For lightcurve datasets, chi-square is now computed in intensity space
        rather than in magnitude space.  (File output is still in
        magnitudes.)
Modified 2006 June 21 by CM:
    In delay-Doppler section, changed delres to del_per_pixel and dopres to
        dop_per_pixel
    For POS renderings and plane-of-sky fit frames, changed res to
        km_per_pixel
    When the root node calls chi2 with list_breakdown = 1, print a warning
        if the model extends beyond the boundaries of the POS frames, if
        any photometric parameter has an illegal value, or if any ellipsoid
        diameter is tiny or negative.  (This change will cause such
        warnings to be displayed for the "write" action.)
Modified 2006 June 18 by CM:
    Allow each delay-Doppler frame within a dataset to have different
        dimensions after vignetting
    Allow each Doppler frame within a dataset to have different
        dimensions after vignetting
    Allow plane-of-sky frames to be rectangular rather than square
    For the "write" action, adjust output for delay-Doppler, Doppler, and
        plane-of-sky frames to allow for masked-out pixels
Modified 2005 June 27 by CM:
    Renamed "round" function to "iround" to avoid conflict
Modified 2005 April 25 by CM:
    For the "write" action, compute the one-sigma percentage uncertainty
        on chi2 correctly; the expression used until now, 100*sqrt(2/dof),
        is only valid when all data are weighted equally.
Modified 2005 March 17 by CM:
    For the "fit" action with parallel processing, check that root receives
        the responses to the correct broadcast
    Compute degrees of freedom in routine read_dat rather than here, so
        that it only needs to be done once per fit rather than repeatedly
        (dof for individual data types is still computed here)
    Allow weights and degrees of freedom to be floating-point rather than
        integer; but if they are integer after all, print integer-rounded
        values to the screen rather than floating-point
    For the "write" action with "speckle" turned on, write Doppler file
        fit_MM_NN.dat with spectral values normalized to the input sdev
        value, not to the sdev value increased for self-noise.
Modified 2005 March 6 by CM:
    For the "write" action, write calibration factors for plane-of-sky
        datasets to disk if the "list_posetcal" parameter is turned on
Modified 2005 March 2 by CM:
    Rename some "sdev" and "var" values to be "oneovervar" (1/variance)
    For the "write" action, adjust arguments of revised "resampim" routine
    For the "write" action, rotate plane-of-sky fit/obs/res frames so that
        north is upward, unless poset_scaling = NONE
Modified 2005 February 22 by CM:
    For the "write" action, fix bug (incorrect array dimensions) in
        scaling fit vs. obs pgm image brightness for plane-of-sky datasets
    For the "write" action, add the new "image_rebin" argument to function
        resampim to handle output images which have much coarser resolution
        than the model POS frames from which they are constructed
        (i.e., which are greatly undersampled).  This situation probably
        won't arise often: The real importance of "image_rebin" is for
        dealing with plane-of-sky fit frames in routine calc_fits.
Modified 2005 January 25 by CM:
    Take care of uninitialized variables
Modified 2005 January 20 by CM:
    For the "write" action, implement the bilinear and bicubic
        interpolation options for the "dd_scaling" parameter
    For the "write" action, write model, data, and residual pgm images
        for POS datasets
    Add pgm file output (obs/fit/res) for POS datasets
    Correct the expression for chi-square for POS datasets
        (the calibration factor was being ignored)
    If the calibration factor for a POS frame is a floating parameter,
        don't allow it to be negative: instead set it to zero.  Also
        display a warning, unless this is a fit and we're not at the end
        of an iteration.
Modified 2005 January 12 by CM:
    For the "fit" action with parallel processing, revise the code so
        that "list_breakdown" will work: For each dataset which is handled
        by a branch node rather than by root, root broadcasts a request for
        that branch node to run chi2 for just that one dataset and to send
        the results to root.  This is necessary because otherwise each
        branch node calls chi2 just once to process *all* of its datasets
        in one shot, without keeping results for different data types
        (Doppler, delay-Doppler, etc.) separate.
Modified 2004 December 1 by CM:
    For the "write" action, adjust the "listfit" option so that
        the "listfit_path" directory can be created on the fly
        if necessary
Modified 2004 November 30 by CM:
    For the "write" action, implement the "listfit" option to write
        out the model "data" files for all delay-Doppler frames
Modified 2004 July 26 by CM:
    If the calibration factor for a Doppler or delay-Doppler frame
        is a floating parameter, don't allow it to be negative: instead
        set it to zero.  Also display a warning, unless this is a fit
        and we're not at the end of an iteration.
Modified 2004 June 27 by CM:
    Fixed bug: For 'dd_scaling' = 'block' the number of pixels per side
        for the resampled delay-Doppler obs/fit/residual images wasn't
        being defined
Modified 2004 April 3 by CM:
    Add the "list_breakdown" argument so that we can list chi2 by
        data type (Doppler, delay-Doppler, POS, lightcurves) for
        both the "fit" and the "write" action as desired
Modified 2004 March 20 by CM:
    For the "write" action, move final summary screen display from
        the main program (shape.c) to here
    For the "write" action, display summaries for each form of data
        (Doppler, delay-Doppler, POS, lightcurves) taken separately,
        in addition to the overall summary
Modified 2004 February 29 by CM:
    For lightcurve output, replace JD244 variable with jdoffset parameter
    For the "write" action, move all screen and file output which
        relies on updated calibration factors to this routine
Modified 2003 May 10 by CM:
    Account for contributions to chi-square from model (delay-)Doppler
        power which lies outside the data frame
Modified 2003 April 24 by CM:
    Display chi-square for Doppler datasets when action = "write"
    Implement the new "weight" parameter
 *****************************************************************************************/
extern "C" {
#include "../shape/head.h"
}
/* File-scope global CUDA variables */
__device__ int c2s_print_breakdown/*, dof*/;
__device__ unsigned char c2s_write_chi2fit0, c2s_badradar,
		c2s_badphoto, c2s_baddopscale, c2s_badposet, c2s_posbnd, c2s_baddiam;
__device__ double c2s_dof_deldop, c2s_dof_doppler, c2s_dof_poset, c2s_dof_lghtcrv,
		c2s_dof, c2s_chi2, c2s_chi2_set, c2s_chi2_all_doppler;
__device__ double c2s_chi2_fit0_deldop64, c2s_dof_fit0_deldop64, c2s_chi2_all_deldop64,
	c2s_chi2_fit0_doppler64, c2s_dof_fit0_doppler64;

typedef struct chi2_thread_t
{
    int thread_no;
	struct par_t *parameter;
    struct dat_t *data;
    unsigned char *htype;
    unsigned char *dtype;
    int *nframes;
    int *hlc_n;
    int *GPUID;
    int gpuid;
    int nsets;
    int list_breakdown;
    int max_frames;
    double chi2_all_deldop;
    double chi2_all_doppler;
    double chi2_all_poset;
    double chi2_all_lghtcrv;
    double chi2_fit0_deldop;
    double chi2_fit0_doppler;
    double chi2_fit0_poset;
    double dof_fit0_deldop;
    double dof_fit0_doppler;
    double dof_fit0_poset;
    double chi2;
    cudaStream_t *gpu_stream;
} chi2_data;

/* File-scope CUDA structures */

/* Function prototype declarations */
__host__ double chi2_deldop_gpu(struct par_t *dpar, struct dat_t *ddat,
		int s, int list_breakdown, double *chi2_all_deldop, double *chi2_fit0_deldop,
		double *dof_fit0_deldop, int nframes, cudaStream_t *c2s_stream);
__host__ double chi2_doppler_gpu(struct par_t *dpar, struct dat_t *ddat, int s,
		int list_breakdown, double *chi2_all_doppler, double *chi2_fit0_doppler,
		double *dof_fit0_doppler, int nframes, cudaStream_t *c2s_stream);
__host__ double chi2_lghtcrv_gpu(struct par_t *dpar, struct dat_t *ddat,
		int s, int list_breakdown, double *chi2_all_lghtcrv, int nframes, int lc_n);

__global__ void c2s_init_krnl(struct dat_t *ddat, int nsets) {
	/* Single-threaded kernel */
	if (threadIdx.x == 0) {
		ddat->chi2 =  0.0;
		c2s_chi2_set = 0.0;
		c2s_chi2_all_doppler = 0.0;
	}
}
__global__ void c2s_retrieve_chi2_krnl(struct dat_t *ddat) {
	/* Single-threaded kernel */
	if (threadIdx.x == 0)
		c2s_chi2 = ddat->chi2;
}
__global__ void c2s_deldop_init_krnl(struct dat_t *ddat, int s, int *ndel,
		int *ndop, double *o2, double *m2, double *om, double *weight, int nframes) {
	/* nframes-threaded kernelDELDOP only */
	int f = blockIdx.x * blockDim.x + threadIdx.x;
	if (f < nframes) {
		ndel[f] = ddat->set[s].desc.deldop.frame[f].ndel;
		ndop[f] = ddat->set[s].desc.deldop.frame[f].ndop;
		o2[f] = ddat->set[s].desc.deldop.frame[f].overflow_o2;
		m2[f] = ddat->set[s].desc.deldop.frame[f].overflow_m2;
		om[f] = 0.0;
		weight[f] = ddat->set[s].desc.deldop.frame[f].weight;
	}
}
__global__ void c2s_doppler_init_krnl(struct dat_t *ddat, int s, int *ndop,
		double *o2, double *m2, double *om, double *weight, int nframes) {
	/* nframes-threaded kernel */
	int f = blockIdx.x * blockDim.x + threadIdx.x;
	if (f < nframes) {
		ndop[f] = ddat->set[s].desc.doppler.frame[f].ndop;
		o2[f] = ddat->set[s].desc.doppler.frame[f].overflow_o2;
		m2[f] = ddat->set[s].desc.doppler.frame[f].overflow_m2;
		om[f] = 0.0;
		weight[f] = ddat->set[s].desc.doppler.frame[f].weight;
	}
}
__global__ void c2s_add_deldop_contributions_krnl(
		struct par_t *dpar,
		struct dat_t *ddat,
		double *o2,
		double *m2,
		double *om,
		double *weight,
		int *ndel,
		int *ndop,
		double *chi2_deldop_frame,
		int s,
		int nframes) {

	/* nframes-threaded kernel */
	int f = blockIdx.x * blockDim.x + threadIdx.x;
	if (f < nframes) {

		double calval, err, o2_fit0, dof_fit0, err_fit0, thresh_fit0;
		chi2_deldop_frame[f] = 0.0;
		int i, j;

		o2[f] += ddat->set[s].desc.deldop.frame[f].overflow_o2;
		m2[f] += ddat->set[s].desc.deldop.frame[f].overflow_m2;
		/* If this frame's calibration factor is allowed to float, set it to
		 * minimize chi-square, the sum over all pixels of
		 * 		        { (obs - calfact*fit)^2 / variance }              */

		if (ddat->set[s].desc.deldop.frame[f].cal.state == 'f') {
			if (om[f] > 0.0)	ddat->set[s].desc.deldop.frame[f].cal.val = om[f]/m2[f];
			else			ddat->set[s].desc.deldop.frame[f].cal.val = TINYCALFACT;
		}

		/*  Compute chi-square for this frame  */
		calval = ddat->set[s].desc.deldop.frame[f].cal.val;
		err = weight[f] * (o2[f] - (2 * calval * om[f]) + (calval * calval * m2[f]));
		ddat->set[s].desc.deldop.frame[f].chi2 = err;
		chi2_deldop_frame[f] += err;

		/* Compute chi-square contributions and deg. of freedom due to pixels
		 * whose model signal is less than or equal to 'chi2fit0_thresh'
		 * standard deviations of the noise in the data frame   */
		o2_fit0 = dof_fit0 = err_fit0 = 0.0;
		thresh_fit0 = dpar->chi2fit0_thresh * ddat->set[s].desc.deldop.frame[f].sdev;
		if (dpar->write_chi2fit0) {
			for (i=0; i<ndel[f]; i++)
				for (j=0; j<ndop[f]; j++)
					//off = j*ndel[f] + i; // For the unrolled fit, obs pointers
					if (calval*ddat->set[s].desc.deldop.frame[f].fit[i][j] <= thresh_fit0) {
						o2_fit0 += ddat->set[s].desc.deldop.frame[f].obs[i][j]*
								ddat->set[s].desc.deldop.frame[f].obs[i][j]*
								ddat->set[s].desc.deldop.frame[f].oneovervar[i][j];
						if (ddat->set[s].desc.deldop.frame[f].oneovervar[i][j] > 0.0)
							dof_fit0 += weight[f];
					}
			err_fit0 = weight[f] * o2_fit0;
			atomicAdd(&c2s_chi2_fit0_deldop64, err_fit0);
			atomicAdd(&c2s_dof_fit0_deldop64, dof_fit0);
		}
	}
}
__global__ void c2s_add_dop_contrbts_srl_krnl(
		struct par_t *dpar,
		struct dat_t *ddat,
		double *o2,
		double *m2,
		double *om,
		double *weight,
		int *ndop,
		double *chi2_doppler_frame,
		int s,
		int f) {

	/* Single-threaded kernel */
	if (threadIdx.x == 0) {
		int j;
		double calval, err, o2_fit0, dof_fit0, err_fit0, thresh_fit0;
		chi2_doppler_frame[f] = 0.0;

		/* If this frame's calibration factor is allowed to float, set it to
		 * minimize chi-square, the sum over all bins of
		        { (obs - calfact*fit)^2 / variance }.                     */

		if (ddat->set[s].desc.doppler.frame[f].cal.state == 'f') {
			if (om[f] > 0.0) 	ddat->set[s].desc.doppler.frame[f].cal.val = om[f]/m2[f];
			else {
				ddat->set[s].desc.doppler.frame[f].cal.val = TINYCALFACT;
				if (dpar->action != FIT )
					printf("WARNING: set %2d frame %2d had negative calfact reset to %10.4e\n",
							s, f, ddat->set[s].desc.doppler.frame[f].cal.val);	}
		}

		/* Compute chi-square for this frame  */
		calval = ddat->set[s].desc.doppler.frame[f].cal.val;
		err = weight[f] * (o2[f] - 2*calval*om[f] + calval*calval*m2[f]);
		ddat->set[s].desc.doppler.frame[f].chi2 = err;
		chi2_doppler_frame[f] += err;

		/* Compute chi-square contributions and dof due to bins whose model
		 * signal is =< 'chi2fit0_thresh' standard deviations of the noise in
		 * the data frame   */
		o2_fit0 = dof_fit0 = err_fit0 = 0.0;
		thresh_fit0 = dpar->chi2fit0_thresh * ddat->set[s].desc.doppler.frame[f].sdev;
		if (dpar->write_chi2fit0) {
			for (j=0; j<ndop[f]; j++)
				if (calval*ddat->set[s].desc.doppler.frame[f].fit[j] <= thresh_fit0) {
					o2_fit0 += ddat->set[s].desc.doppler.frame[f].obs[j] *
							ddat->set[s].desc.doppler.frame[f].obs[j]
					  *ddat->set[s].desc.doppler.frame[f].oneovervar[j];
					if (ddat->set[s].desc.doppler.frame[f].oneovervar[j] > 0.0)
						dof_fit0 += weight[f];
				}
			err_fit0 = weight[f]*o2_fit0;
			atomicAdd(&c2s_chi2_fit0_doppler64, err_fit0);
			atomicAdd(&c2s_dof_fit0_doppler64, dof_fit0);
		}
	}
}
__global__ void c2s_doppler_add_o2_krnl(struct dat_t *ddat, double *o2, double
		*m2, double *om, int *ndop, int s, int f) {
	/* ndop-threaded kernel */
	int j = blockIdx.x * blockDim.x + threadIdx.x+1;
	double temp;

	if (j <= ndop[f]) {
		/* Add contributions from power within frame limits */

		/* Next 2 lines implement: o2 += obs[j]*obs[j]*oneovervar[j];	 */
		temp = ddat->set[s].desc.doppler.frame[f].obs[j] *
				ddat->set[s].desc.doppler.frame[f].obs[j] *
				ddat->set[s].desc.doppler.frame[f].oneovervar[j];
		atomicAdd(&o2[f], temp);

		/* Next 2 lines implement: m2 += fit[j]*fit[j]*oneovervar[j];		 */
		temp = ddat->set[s].desc.doppler.frame[f].fit[j] *
				ddat->set[s].desc.doppler.frame[f].fit[j] *
				ddat->set[s].desc.doppler.frame[f].oneovervar[j];
		atomicAdd(&m2[f], temp);

		/* Next 2 lines implement: om += fit[j]*obs[j]*oneovervar[j];		 */
		temp = ddat->set[s].desc.doppler.frame[f].fit[j] *
				ddat->set[s].desc.doppler.frame[f].obs[j] *
				ddat->set[s].desc.doppler.frame[f].oneovervar[j];
		atomicAdd(&om[f], temp);
	}
}
__global__ void c2s_lghtcrv_serial_krnl(struct dat_t *ddat, int s, double *dof_chi2set,
		double3 *o2m2om) {

	/* single-threaded kernel */
	o2m2om[0].x = o2m2om[0].y = o2m2om[0].z = 0.0;
	double obs, fit, oneovervar, calval;

	if (threadIdx.x == 0) {
		for (int i=1; i<=ddat->set[s].desc.lghtcrv.n; i++) {
			obs = ddat->set[s].desc.lghtcrv.obs[i];
			fit = ddat->set[s].desc.lghtcrv.fit[i];
			oneovervar = ddat->set[s].desc.lghtcrv.oneovervar[i];
			o2m2om[0].x += obs * obs * oneovervar;
			o2m2om[0].y += fit * fit * oneovervar;
			o2m2om[0].z += fit * obs * oneovervar;
			o2m2om[1].x = o2m2om[1].y = o2m2om[1].z = 0.0;
		}


		/* If lightcurve's calibration factor is allowed to float, set it to mini-
		 * mize chi-square (sum over all points of {(obs-calfact*fit)^2/variance}*/
		if (ddat->set[s].desc.lghtcrv.cal.state == 'f') {
			if (o2m2om[0].z > 0.0)
				ddat->set[s].desc.lghtcrv.cal.val = o2m2om[0].z/o2m2om[0].y;
			else
				ddat->set[s].desc.lghtcrv.cal.val = TINYCALFACT;
		}
		calval = ddat->set[s].desc.lghtcrv.cal.val;

		/* Compute chi-square for dataset  */
		dof_chi2set[0] = ddat->set[s].desc.lghtcrv.dof;
		dof_chi2set[1] = ddat->set[s].desc.lghtcrv.weight * (o2m2om[0].x - 2 *
				calval * o2m2om[0].z + calval * calval * o2m2om[0].y);
	}

}
__global__ void c2s_get_prntflgs_krnl(struct par_t *dpar, struct dat_t *ddat) {
	/* Single-threaded kernel */
	if (threadIdx.x == 0) {
		c2s_print_breakdown = (ddat->dof_deldop > SMALLVAL || ddat->dof_doppler  > SMALLVAL
				|| ddat->dof_poset    > SMALLVAL	|| ddat->dof_lghtcrv  > SMALLVAL);
		c2s_dof_deldop = ddat->dof_deldop;
		c2s_dof_doppler = ddat->dof_doppler;
		c2s_dof_poset = ddat->dof_poset;
		c2s_dof_lghtcrv = ddat->dof_lghtcrv;
		c2s_dof = ddat->dof;
		c2s_write_chi2fit0 = dpar->write_chi2fit0;
		c2s_badradar = dpar->badradar;
		c2s_badphoto = dpar->badphoto;
		c2s_baddopscale = dpar->baddopscale;
		c2s_badposet = dpar->badposet;
		c2s_posbnd = dpar->posbnd;
		c2s_baddiam = dpar->baddiam;
	}
}
__global__ void c2_add_chi2_krnl(struct dat_t *ddat, int s) {
	/* Single-threaded kernel */
	if (threadIdx.x == 0)
		ddat->chi2 += ddat->set[s].chi2;
}
__global__ void c2_set_chi2_krnl(struct dat_t *ddat, double chi2, int s) {
	/* Single-threaded kernel */
	if (threadIdx.x == 0)
		ddat->set[s].chi2 = chi2;
}
__global__ void deldop_wrt_chi2fit0_krnl(struct par_t *dpar, struct dat_t *ddat,
		int s, int f, int *ndel, int *ndop, int nThreads, double *returns,
		double *o2_fit0_dof_fit0) {

	/* ndel*ndop-threaded kernel */
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	int i = offset % ndel[f] + 1;
	int j = offset / ndel[f] + 1;
	double temp, err_fit0 = 0.0, thresh_fit0;

	/* returns[0] = chi2_fit0_deldop
	 * returns[1] = dof_fit0_deldop	 */

	if (dpar->write_chi2fit0) {
		if (offset < nThreads) {

			thresh_fit0 = dpar->chi2fit0_thresh * ddat->set[s].desc.deldop.frame[f].sdev;

			if (ddat->set[s].desc.deldop.frame[f].cal.val *
					ddat->set[s].desc.deldop.frame[f].fit[i][j] <= thresh_fit0) {

				temp = ddat->set[s].desc.deldop.frame[f].obs[i][j] *
						ddat->set[s].desc.deldop.frame[f].obs[i][j] *
						ddat->set[s].desc.deldop.frame[f].oneovervar[i][j];
				atomicAdd(&o2_fit0_dof_fit0[0], temp);

				if (ddat->set[s].desc.deldop.frame[f].oneovervar[i][j] > 0.0) {
					temp = ddat->set[s].desc.deldop.frame[f].weight;
					atomicAdd(&o2_fit0_dof_fit0[1], temp);
				}
			}
		}
		__syncthreads();

		if (offset == 0) {
			temp = ddat->set[s].desc.deldop.frame[f].weight;
			err_fit0 = temp*o2_fit0_dof_fit0[0];
			returns[0] = err_fit0;
			returns[1] = o2_fit0_dof_fit0[1];
		}
	}
	else
		if (offset==0)
			returns[0] = returns[1] = 0.0;

}
__global__ void dop_wrt_chi2fit0_krnl(struct par_t *dpar, struct dat_t *ddat,
		int s, int f, int nThreads, double *returns, double *o2_fit0_dof_fit0) {

	/* ndop-threaded kernel */
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	double temp;
	double err_fit0 = 0.0, thresh_fit0;
	returns[0] = returns[1] = 0.0;
	/* returns[0] = chi2_fit0_deldop
	 * returns[1] = dof_fit0_deldop	 */

	if (dpar->write_chi2fit0) {
		if (offset < nThreads) {

			thresh_fit0 = dpar->chi2fit0_thresh * ddat->set[s].desc.doppler.frame[f].sdev;

			if (ddat->set[s].desc.doppler.frame[f].cal.val *
					ddat->set[s].desc.doppler.frame[f].fit_s[offset] <= thresh_fit0) {

				temp = ddat->set[s].desc.doppler.frame[f].obs[offset] *
						ddat->set[s].desc.doppler.frame[f].obs[offset] *
						ddat->set[s].desc.doppler.frame[f].oneovervar[offset];
				atomicAdd(&o2_fit0_dof_fit0[0], temp);

				if (ddat->set[s].desc.doppler.frame[f].oneovervar[offset] > 0.0) {
					temp = ddat->set[s].desc.doppler.frame[f].weight;
					atomicAdd(&o2_fit0_dof_fit0[1], temp);
				}
			}
		}
		__syncthreads();

		if (offset == 0) {
			temp = ddat->set[s].desc.doppler.frame[f].weight;
			err_fit0 = temp* o2_fit0_dof_fit0[0];
			returns[0] = err_fit0;
			returns[1] = o2_fit0_dof_fit0[1];
		}
	}
	else
		if (offset==0)
			returns[0] = returns[1] = 0.0;
}

__host__ double chi2_gpu(
		struct par_t *dpar,
		struct dat_t *ddat,
		unsigned char *htype,
		unsigned char *dtype,
		int *hnframes,
		int *hlc_n,
		int list_breakdown,
		int nsets,
		cudaStream_t *c2s_stream,
		int max_frames)
{
	/* This version of chi2 accepts an existing streams array from the calling
	 * function. Number of streams in array is guaranteed to be as large as
	 * the the largest nframes for all sets	 */
	int s, print_breakdown;
	unsigned char write_chi2fit0, baddiam, badphoto, badposet, baddopscale,
			posbnd, badradar;
	double chi2_all_doppler, chi2_all_deldop, chi2_all_poset, chi2_all_lghtcrv,
		   chi2_fit0_doppler, chi2_fit0_deldop, chi2_fit0_poset,
		   dof_fit0_doppler, dof_fit0_deldop, chi2, dof;
	double dof_deldop, dof_doppler, dof_poset, dof_lghtcrv;
	char dofstring[MAXLEN], dof0string[MAXLEN];

	/*  Initialize variables that accumulate chi-square values  */
	chi2_all_deldop = chi2_all_doppler = chi2_all_poset = chi2_all_lghtcrv =
			chi2_fit0_deldop = chi2_fit0_doppler /*= chi2_fit0_poset*/ = 0.0;
	dof_fit0_deldop = dof_fit0_doppler/* = dof_fit0_poset*/ = 0.0;

	/* Initialize variables that accumulate chi-square values  */
	c2s_init_krnl<<<1,1>>>(ddat, nsets);
	checkErrorAfterKernelLaunch("c2s_init_krnl2, chi2_cuda_streams");

	/* Loop through all datasets, carry out chi-square computations, and
	 * provide screen and image output                            */
	for (s=0; s<nsets; s++) {
		switch (htype[s]) {
		case DELAY:
			chi2 = chi2_deldop_gpu(dpar, ddat, s, list_breakdown,
					&chi2_all_deldop, &chi2_fit0_deldop, &dof_fit0_deldop,
					hnframes[s], c2s_stream);
			c2_set_chi2_krnl<<<1,1>>>(ddat, chi2, s);
			checkErrorAfterKernelLaunch("c2_set_chi2_krnl");
			break;
		case DOPPLER:
			chi2 = chi2_doppler_gpu(dpar, ddat, s,list_breakdown,
					&chi2_all_doppler, &chi2_fit0_doppler, &dof_fit0_doppler,
					hnframes[s], c2s_stream);
			c2_set_chi2_krnl<<<1,1>>>(ddat, chi2, s);
			checkErrorAfterKernelLaunch("c2_set_chi2_krnl");
			break;
		case POS:
			printf("\nWrite chi2_poset_cuda!\n");
			//			dat->set[s].chi2 = chi2_poset(dpar, s);
			break;
		case LGHTCRV:
			chi2 = chi2_lghtcrv_gpu(dpar, ddat, s, list_breakdown,
					&chi2_all_lghtcrv, hnframes[s], hlc_n[s]);
			c2_set_chi2_krnl<<<1,1>>>(ddat, chi2, s);
			checkErrorAfterKernelLaunch("c2_set_chi2_krnl, chi2_cuda");
			break;
		default:
			printf("chi2_gpu.cu: can't handle this type yet\n");
		}

		/* Single-thread kernel adds ddat->set[s].chi2 to ddat->chi2 */
		c2_add_chi2_krnl<<<1,1>>>(ddat, s);
		checkErrorAfterKernelLaunch("c2_add_chi2_krnl");

	}  /* end for loop over datasets */
	/* Launch single-threaded kernel to retrieve ddat->chi2 to return it */
	c2s_retrieve_chi2_krnl<<<1,1>>>(ddat);
	checkErrorAfterKernelLaunch("c2s_retrieve_chi2_krnl");
	gpuErrchk(cudaMemcpyFromSymbol(&chi2, c2s_chi2, sizeof(double),
			0, cudaMemcpyDeviceToHost));
	/*.......................................................................*/


	/* Call kernel to get flags from ddat */
	c2s_get_prntflgs_krnl<<<1,1>>>(dpar, ddat);
	checkErrorAfterKernelLaunch("c2s_get_prntflgs_krnl");
	gpuErrchk(cudaMemcpyFromSymbol(&print_breakdown, c2s_print_breakdown,
			sizeof(int), 0, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpyFromSymbol(&dof_deldop, c2s_dof_deldop, sizeof(double),
			0, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpyFromSymbol(&dof_doppler, c2s_dof_doppler, sizeof(double),
			0, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpyFromSymbol(&dof_lghtcrv, c2s_dof_lghtcrv, sizeof(double),
			0, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpyFromSymbol(&write_chi2fit0, c2s_write_chi2fit0,
			sizeof(unsigned char), 0, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpyFromSymbol(&dof, c2s_dof, sizeof(double),
			0, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpyFromSymbol(&baddiam, c2s_baddiam, sizeof(unsigned char),
			0, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpyFromSymbol(&badphoto, c2s_badphoto, sizeof(unsigned char),
			0, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpyFromSymbol(&posbnd, c2s_posbnd, sizeof(unsigned char),
			0, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpyFromSymbol(&badposet, c2s_badposet, sizeof(unsigned char),
			0, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpyFromSymbol(&badradar, c2s_badradar, sizeof(unsigned char),
			0, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpyFromSymbol(&baddopscale, c2s_baddopscale, sizeof(unsigned char),
			0, cudaMemcpyDeviceToHost));

	if (list_breakdown) {
		if (print_breakdown) {
			printf("#\n");
			if (dof_deldop > SMALLVAL) {
				intifpossible( dofstring, MAXLEN, dof_deldop, SMALLVAL, "%f");
				printf("delay   chi2 = %e for %s dof (reduced chi2 = %f)\n",
						chi2_all_deldop, dofstring, chi2_all_deldop/dof_deldop);
				if (write_chi2fit0) {
					intifpossible( dof0string, MAXLEN, dof_fit0_deldop, SMALLVAL, "%f");
					printf("              (%e outside model for %s dof)\n",
							chi2_fit0_deldop, dof0string);
				}
			}
			if (dof_doppler > SMALLVAL) {
				intifpossible( dofstring, MAXLEN, dof_doppler, SMALLVAL, "%f");
				printf("Doppler chi2 = %e for %s dof (reduced chi2 = %f)\n",
						chi2_all_doppler, dofstring, chi2_all_doppler/dof_doppler);
				if (write_chi2fit0) {
					intifpossible( dof0string, MAXLEN, dof_fit0_doppler, SMALLVAL, "%f");
					printf("              (%e outside model for %s dof)\n",
							chi2_fit0_doppler, dof0string);
				}
			}
			if (dof_lghtcrv > SMALLVAL) {
				intifpossible( dofstring, MAXLEN, dof_lghtcrv, SMALLVAL, "%f");
				printf("lghtcrv chi2 = %e for %s dof (reduced chi2 = %f)\n",
						chi2_all_lghtcrv, dofstring, chi2_all_lghtcrv/dof_lghtcrv);
			}
			intifpossible( dofstring, MAXLEN, dof, SMALLVAL, "%f");
			printf("ALLDATA chi2 = %e for %s dof (reduced chi2 = %f)",
					chi2, dofstring, chi2/dof);
		} else {
			intifpossible( dofstring, MAXLEN, ddat->dof, SMALLVAL, "%f");
			printf("        chi2 = %e for %s dof (reduced chi2 = %f)",
					chi2, dofstring, chi2/dof);
		}

	if (baddiam)		printf("  (BAD DIAMS)");
	if (badphoto)		printf("  (BAD PHOTO) (chi2_cuda)");
	if (posbnd)			printf("  (BAD POS)");
	if (badposet)		printf("  (BAD POSET)");
	if (badradar)		printf("  (BAD RADAR)");
	if (baddopscale)	printf("  (BAD DOPSCALE)");
	printf("\n");
	if (print_breakdown &&  write_chi2fit0) {
		intifpossible( dof0string, MAXLEN,
				dof_fit0_deldop + dof_fit0_doppler/* + dof_fit0_poset*/,
				SMALLVAL, "%f");
		printf("              (%e outside model for %s dof)\n",
				chi2_fit0_deldop + chi2_fit0_doppler/* + chi2_fit0_poset*/, dof0string);
	}
	printf("#\n");
	fflush(stdout);
	}
	/*.......................................................................*/

	return chi2;
}

__host__ double chi2_deldop_gpu(
		struct par_t *dpar,
		struct dat_t *ddat,
		int s,
		int list_breakdown,
		double *chi2_all_deldop,
		double *chi2_fit0_deldop,
		double *dof_fit0_deldop,
		int nframes,
		cudaStream_t *c2s_stream)
{
	int f, *ndel, *ndop, hndel[nframes], hndop[nframes], nThreads[nframes];
	double chi2_set, *chi2_deldop_frame, h_chi2_deldop_frame[nframes];
	dim3 BLK[nframes],THD, BLKfrm, THD64;
	THD.x = maxThreadsPerBlock;	THD64.x = 64;
	BLKfrm.x = floor((THD64.x -1 + nframes)/THD64.x);
	double *o2_fit0_dof_fit0;
	/* o2, m2, and om are per-frame radar variables */
	double *o2, *m2, *om, *weight, *returns, *hreturns;
	chi2_set = 0.0;

	gpuErrchk(cudaMalloc((void**)&o2, 				sizeof(double)*nframes));
	gpuErrchk(cudaMalloc((void**)&m2, 				sizeof(double)*nframes));
	gpuErrchk(cudaMalloc((void**)&om, 				sizeof(double)*nframes));
	gpuErrchk(cudaMalloc((void**)&weight, 			sizeof(double)*nframes));
	gpuErrchk(cudaMalloc((void**)&chi2_deldop_frame,sizeof(double)*nframes));
	gpuErrchk(cudaMalloc((void**)&ndel,				sizeof(int)*nframes));
	gpuErrchk(cudaMalloc((void**)&ndop,				sizeof(int)*nframes));
	gpuErrchk(cudaMalloc((void**)&returns, 			sizeof(double)*2));
	gpuErrchk(cudaMalloc((void**)&o2_fit0_dof_fit0,	sizeof(double)*2));
	hreturns = (double *) malloc(2*sizeof(double));
	hreturns[0] = hreturns[1] = 0.0;

	/* Get values for ndel and ndop, and the overflow parameters o2, m2, om */
	c2s_deldop_init_krnl<<<BLKfrm,THD64>>>(ddat, s, ndel, ndop, o2, m2, om,
			weight, nframes);
	checkErrorAfterKernelLaunch("c2s_deldop_init_krnl");
	gpuErrchk(cudaMemcpy(&hndel, ndel, sizeof(int)*nframes, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&hndop, ndop, sizeof(int)*nframes, cudaMemcpyDeviceToHost));

	/* Calculate launch parameters */
	for (f=0; f<nframes; f++) {
		nThreads[f] = hndel[f]*hndop[f];
		BLK[f].x = floor((THD.x - 1 + nThreads[f]) / THD.x);
	}

	/* Add contributions from power within limits of data frame. This kernel
	 * also takes care of the frame's calibration factor and  computes chi2
	 * for this frame */
	sum_o2m2om_gpu(ddat, o2, m2, om, nframes, hndel[0]*hndop[0], s, c2s_stream);

	c2s_add_deldop_contributions_krnl<<<BLKfrm,THD64>>>(dpar, ddat, o2, m2,
			om, weight, ndel, ndop, chi2_deldop_frame, s, nframes);
	checkErrorAfterKernelLaunch("c2s_add_deldop_contributions_krnl");

	gpuErrchk(cudaMemcpy(&h_chi2_deldop_frame, chi2_deldop_frame,
			sizeof(double) * nframes, cudaMemcpyDeviceToHost));

	/* Add all frames from device memory to host memory */
	for (f=0; f<nframes; f++) {
		chi2_set += h_chi2_deldop_frame[f];
		if (list_breakdown)
			*chi2_all_deldop += h_chi2_deldop_frame[f];
	}

	if (list_breakdown) {
		for (f=0; f<nframes; f++) {
			/* WARNING WARNING WARNING */
			/* This kernel is for accuracy testing only. Must develop a proper
			 * parallel reduction for this instead */
			deldop_wrt_chi2fit0_krnl<<<BLK[f],THD, 0, c2s_stream[f]>>>(dpar,
					ddat, s, f, ndel, ndop, nThreads[f], returns, o2_fit0_dof_fit0);
			checkErrorAfterKernelLaunch("deldop_wrt_chi2fit0_krnl64");
		}
		gpuErrchk(cudaMemcpy(hreturns, returns, sizeof(double) * 2,
				cudaMemcpyDeviceToHost));
		*chi2_fit0_deldop = hreturns[0];
		*dof_fit0_deldop = hreturns[1];
	}

	cudaFree(o2);
	cudaFree(m2);
	cudaFree(om);
	cudaFree(ndel);
	cudaFree(ndop);
	cudaFree(weight);
	cudaFree(returns);
	cudaFree(chi2_deldop_frame);
	cudaFree(o2_fit0_dof_fit0);
	free(hreturns);
	return chi2_set;
}

__host__ double chi2_doppler_gpu(struct par_t *dpar, struct dat_t *ddat, int s,
		int list_breakdown, double *chi2_all_doppler, double *chi2_fit0_doppler,
		double *dof_fit0_doppler, int nframes, cudaStream_t *c2s_stream)
{
	int f, *ndop, hndop[nframes];
	double chi2_set, *chi2_doppler_frame, h_chi2_doppler_frame[nframes];
	dim3 BLK[nframes], THD, BLKfrm, THD64;
	THD.x = maxThreadsPerBlock;	THD64.x = 64;
	double *o2, *m2, *om, *weight, hreturns[2], *returns, *o2_fit0_dof_fit0;
	chi2_set = 0.0;
	BLKfrm.x = floor((THD64.x -1 + nframes)/THD64.x);

	gpuErrchk(cudaMalloc((void**)&o2, 				sizeof(double)*nframes));
	gpuErrchk(cudaMalloc((void**)&m2, 				sizeof(double)*nframes));
	gpuErrchk(cudaMalloc((void**)&om, 				sizeof(double)*nframes));
	gpuErrchk(cudaMalloc((void**)&ndop, 			sizeof(int)*nframes));
	gpuErrchk(cudaMalloc((void**)&weight, 			sizeof(double)*nframes));
	gpuErrchk(cudaMalloc((void**)&chi2_doppler_frame,sizeof(double)*nframes));
	gpuErrchk(cudaMalloc((void**)&returns, 			sizeof(double)*2));
	gpuErrchk(cudaMalloc((void**)&o2_fit0_dof_fit0,	sizeof(double)*2));
	hreturns[0] = hreturns[1] = 0.0;

	/* Get values for ndel and ndop, and the overflow parameters o2, m2, om */
	c2s_doppler_init_krnl<<<BLKfrm,THD64>>>(ddat, s, ndop, o2, m2, om, weight,
			nframes);
	checkErrorAfterKernelLaunch("c2s_doppler_init_krnl64");
	gpuErrchk(cudaMemcpy(&hndop, ndop, sizeof(int)*nframes, cudaMemcpyDeviceToHost));

	/* Calculate launch parameters */
	for (f=0; f<nframes; f++)
		BLK[f].x = floor((THD.x - 1 + hndop[f]) / THD.x);

	/*  Loop through all frames for this dataset  */
	for (f=0; f<nframes; f++) {
		/* Add contributions from power within data frame limits. This kernel
		 * also considers frame's calibration factor & computes frame chi2  */
		c2s_doppler_add_o2_krnl<<<BLK[f],THD,0,c2s_stream[f]>>>(ddat, o2, m2,
				om, ndop, s, f);
	}
	checkErrorAfterKernelLaunch("c2_doppler_add_o2_krnl64");

	for (f=0; f<nframes; f++) {
		c2s_add_dop_contrbts_srl_krnl<<<1,1>>>(dpar, ddat, o2, m2, om, weight,
				ndop, chi2_doppler_frame, s, f);
	}
	checkErrorAfterKernelLaunch("c2_add_dop_contrbts_krnl64");
	gpuErrchk(cudaMemcpy(&h_chi2_doppler_frame, chi2_doppler_frame,
			sizeof(double) * nframes, cudaMemcpyDeviceToHost));

	/* Add all frames from device memory to host memory*/
	for (f=0; f<nframes; f++) {
		chi2_set += h_chi2_doppler_frame[f];
		if (list_breakdown)
			*chi2_all_doppler += h_chi2_doppler_frame[f];
	}

	/* Compute the chi-square contributions and number of degrees of freedom
	 * due to bins whose model signal is less than or equal to 'chi2fit0_thresh'
	 *  standard deviations of the noise in the data frame   */
	if (list_breakdown) {
		for (f=0; f<nframes; f++) {
			dop_wrt_chi2fit0_krnl<<<BLK[f],THD, 0, c2s_stream[f]>>>(dpar,
					ddat, s, f, hndop[f], returns, o2_fit0_dof_fit0);
			checkErrorAfterKernelLaunch("dop_wrt_chi2fit0_krnl64");
		}
		gpuErrchk(cudaMemcpy(&hreturns, returns, sizeof(double) * 2,
				cudaMemcpyDeviceToHost));
		*chi2_fit0_doppler = hreturns[0];
		*dof_fit0_doppler  = hreturns[1];
	}

	cudaFree(o2);
	cudaFree(m2);
	cudaFree(om);
	cudaFree(ndop);
	cudaFree(weight);
	cudaFree(returns);
	cudaFree(chi2_doppler_frame);
	cudaFree(o2_fit0_dof_fit0);
	return chi2_set;
}
__host__ double chi2_lghtcrv_gpu(
		struct par_t *dpar,
		struct dat_t *ddat,
		int s,
		int list_breakdown,
		double *chi2_all_lghtcrv,
		int nframes,
		int lc_n)		{

	double *dof_chi2set, h_dof_chi2set[2];
	double3 *o2m2om, *h_o2m2om;

	gpuErrchk(cudaMalloc((void**)&dof_chi2set, sizeof(double) * 2));
	gpuErrchk(cudaMalloc((void**)&o2m2om, sizeof(double3) * 2));
	c2s_lghtcrv_serial_krnl<<<1,1>>>(ddat, s, dof_chi2set, o2m2om);
	gpuErrchk(cudaMemcpy(&h_dof_chi2set, dof_chi2set, sizeof(double)*2,
			cudaMemcpyDeviceToHost));

	if (list_breakdown)
		*chi2_all_lghtcrv += h_dof_chi2set[1];

	cudaFree(dof_chi2set);
	cudaFree(o2m2om);
	return h_dof_chi2set[1];
}
