/*****************************************************************************************
                                                                              calc_fits.c
As the name implies, this routine calculates the fits to each data frame for the current
set of model parameters.  For example, for each delay-Doppler frame it calls routine
posvis to create the model plane-of-sky image and then routine pos2deldop to create the
model delay-Doppler image from this POS image.
calc_fits also performs some of the screen and file output required by the "write" action;
in particular, it carries out tasks that require information associated with plane-of-sky
renderings, since such information is quickly overwritten if the "pos_scope" parameter is
set to "global" (i.e., if all frames and lightcurve points share the same memory for their
"pos" structures).
Modified 2015 June 10 by CM:
    Implement smearing for the "fit" and "write" actions
Modified 2014 February 14 by CM:
    Add "ilaw" argument to the apply_photo routine
Modified 2013 July 28 by CM:
    For the "write" action, output ppm POS images when the "write_highlight" parameter is
        turned on
Modified 2013 July 7 by CM:
    For the "write" action for lightcurve points and plane-of-sky frames, display the
        body-fixed longitude and latitude of the phase-angle bisector
Modified 2013 June 25 by CM:
    Allow POS images written for optical data to be annotated with principal-axis shafts
        and the angular momentum vector
    For POS images (sky renderings), display the name of the image file and the maximum
        pixel value in the plot_surface routine (called by the write_pos routine) rather
        than here
Modified 2013 April 24 by CM:
    Implement the "listpos_deldop" "listpos_opt" and "listpos_path" parameters
    Adjust names of output images so they are in alphanumeric order if > 100 per dataset
Modified 2012 April 2 by CM:
    Correct instantaneous maximum breadth calculation for Doppler scaling factor
Modified 2011 August 14 by CM:
    Display sidereal spin vector at each epoch, even for a PA rotator, if
        any spin impulses are used
Modified 2010 September 1 by CM:
    Initialize variables to avoid compilation warnings
Modified 2010 July 29 by CM:
    Fix bug introduced in calc_lghtcrv: rotation phases weren't being
        displayed for the "write" action
    For the "write" action for lightcurve datasets, include shadowed
        regions in projected area (and geometric albedo calculation)
        and display percentage of projected area that's shadowed
Modified 2010 June 15 by CM:
    Revise arguments to pos2deldop and pos2doppler routines
Modified 2010 May 28 by CM:
    Fix bug introduced with preceding change: in calc_lghtcrv, only
        deallocate memory for the "write" action (since it wasn't
        allocated in the first place for other actions)
Modified 2010 May 24 by CM:
    For the "write" action for lightcurves, output the projected area and
        (for absolute photometry) geometric albedo
Modified 2010 April 12 by CM:
    For the "write" action, include overflow region when computing
        cross sections
Modified 2009 July 29 by CM:
    For the "write" action, fix bug: output ppm images rather than pgm
        images if the "plot_angmom" parameter is turned on
    For the "write" action, pass an argument to the "write_pos" routine
        explicitly telling it whether or not to produce a colored image
Modified 2009 April 3 by CM:
    Initialize the "posbnd_logfactor" parameter and later set it for
        models that extend beyond the POS frame
    Add "badposet" and "badposet_logfactor" parameters: initialize them
        here and then use the new "checkposet" routine to adjust them for
        plane-of-sky fit images that are too small to "contain" the
        target
    Add "badradar" and "badradar_logfactor" parameters: initialize them
        here and then use the "pos2deldop" and "pos2doppler" routines
        (which are now int rather than void) to adjust them for models that
        are too wide in delay-Doppler space for the routines to handle
    Add "warn_badradar" argument to pos2deldop and pos2doppler routines
    For the "write" action, display each plane-of-sky fit frame's linear
        dimensions, the linear dimensions of the rectangular subset that
        contains the target, and the linear COM offsets
Modified 2008 December 12 by CM:
    For the "write" action for NPA rotators, list Euler angles (giving
        the body-fixed axes' orientations in ecliptic coordinates) and
        spin vector components (in body-fixed coordinates) for each
        observation epoch
    For the "write" action for NPA rotators, ensure that maximum breadth
        is nonnegative
Modified 2007 August 10 by CM:
    Eliminated unused variables and cleaned up a printf format
    For POS model frames (sky renderings) associated with lightcurve points
        and with plane-of-sky data frames, don't display the maximum pixel
        value unless the "optposmax" parameter is nonzero
Modified 2007 August 4 by CM:
    Add comp matrix for POS frames
    Add orbit_offset and body arguments to posvis routine and remove
        facet argument
    Add orbit_xoff, orbit_yoff, orbit_dopoff and body parameters to
        pos2deldop and pos2doppler routines
    Add body argument to apply_photo routine
Modified 2007 January 17 by CM:
    For the "write" action, display instantaneous folded zero-crossing
        bandwidth for Doppler and delay-Doppler frames
Modified 2007 January 11 by CM:
    In calc_lghtcrv for the "write" action, count lightcurve points
        from 0 rather than 1, as is already done for lightcurve POS images
        (and for Doppler, delay-Doppler, and plane-of-sky frames)
Modified 2007 January 6 by CM:
    In calc_lghtcrv for the "write" action, save rotation phase for each
        calculated lightcurve point so they can be output by routine chi2,
        and use cubic spline interpolation to obtain rotation phase at
        each observation epoch.  Also display range of rotation phases
        if only one calculated point per lightcurve is displayed in full
Modified 2006 October 1 by CM:
    In calc_lghtcrv, model lightcurve points are now intensities
        (relative to the solar intensity) rather than magnitudes
    In calc_lghtcrv and calc_poset, apply_photo routine has been revised
        to account for the POS pixel area and the 1 AU Sun-target distance
Modified 2006 September 1 by CM and MCN:
    When "exclude_seen" parameter is used, add check that facet number
        pos->f[i][j] is nonnegative
    For the "write" action, don't display cross sections and albedos
        for uncalibrated (delay-)Doppler frames
Modified 2006 June 21 by CM:
    In calc_deldop, changed delres to del_per_pixel and dopres to
        dop_per_pixel
    In calc_doppler, changed dopres to dop_per_bin
    For POS renderings and plane-of-sky fit frames, changed res to
        km_per_pixel
Modified 2006 June 18 by CM:
    Allow each delay-Doppler frame within a dataset to have different
        dimensions after vignetting
    Allow each Doppler frame within a dataset to have different
        dimensions after vignetting
    Allow plane-of-sky frames to be rectangular rather than square,
        and no longer require an odd number of pixels per side
    Eliminate range datasets
Modified 2006 March 10 by CM:
    Add "speckle" argument to pos2deldop and pos2doppler routines
Modified 2005 October 6 by CM:
    For lightcurve datasets, replace SUNMAG constant by "sun_appmag"
        parameter, so that absolute photometry with filters other than
        V band can be used
Modified 2005 July 25 by CM:
    For "write" action, display the model radar cross section and albedo
        for each delay-Doppler and Doppler frame
Modified 2005 July 22 by CM:
    Created five separate routines for writing POS frames as images
        so that they can be called separately if the "mark_unseen"
        parameter is turned on for the "write" action (since in this
        case we must first process all datasets to see which model
        facets were "seen" and only then can write the POS images)
Modified 2005 July 14 by CM:
    Fix bug in computing LE-to-COM delay and distance, LE-to-TE
        delay and distance, and instantantaneous bandwidth and breadth
Modified 2005 July 13 by CM:
    For "write" action for lightcurve points and plane-of-sky frames,
        display the body-fixed longitude and latitude of the
        Sun-to-asteroid line
Modified 2005 July 5 by CM:
    Remove the "dir" argument from pos2deldop and pos2doppler and add
        the "set" argument
Modified 2005 July 3 by CM:
    For "write" action for lightcurve datasets, implement the
        "lcrv_writeall" parameter, which produces screen display for
        every model lightcurve point rather than just the one point
        which falls closest to the midpoint of the observations.
Modified 2005 June 25 by CM:
    For "write" action for delay-Doppler frames, display the delay and
        distance between the leading edge and the center of mass and
        between the leading edge and the trailing edge;
        for delay-Doppler and Doppler frames, display the instantaneous
        zero-crossing bandwidth and maximum breadth.  All of the above
        are obtained from the model's delay-Doppler limits as
        determined PRIOR to convolution with the delay and Doppler
        response functions.
Modified 2005 June 22 by CM:
    Keep track of which model facets have been "seen" (i.e., are visible
        from Earth, are unshadowed, and have sufficiently low scattering
        and incidence angles) in at least one data frame or lightcurve
        point
Modified 2005 April 23 by CM:
    For the "write" action, list whether or not epochs have been corrected
        for one-way light travel time
Modified 2005 March 1 by CM:
    Adjust arguments to the revised "resampim" routine to permit rotation
        of resampled plane-of-sky frames
    Initialize the "posbnd" parameter (flag indicating that the model
        extends beyond the model POS frame) to 0 here rather than in
        bestfit.c so that it can used for actions other than "fit"
    Fix bug in calc_poset which was incorrectly flagging the model as
        being too small for the model POS frame
Modified 2005 February 21 by CM:
    Use the new "poset_resample" parameter to allow interpolation methods
        other than bilinear for constructing plane-of-sky fit images for
        plane-of-sky data frames
    Add the new "image_rebin" argument to function resampim to handle
        plane-of-sky fit frames which have much coarser resolution
        than the model POS frames from which they are constructed
        (i.e., which are greatly undersampled)
    For "write" action, display maximum pixel value for model POS images
        for plane-of-sky frames and calculated lightcurve images
        (in case someone wants to use the "optposmax" parameter to
        truncate the image brightness)
Modified 2005 February 6 by CM:
    For "write" action, display rotation phase
    For "write" action, fix bug in computing the angular body-fixed
        coordinates of the line of sight for lightcurve datasets
Modified 2005 January 25 by CM:
    Take care of unused and uninitialized variables
Modified 2005 January 24 by CM:
    Add "calc_poset" routine to handle POS datasets
    For "write" action, display the angular body-fixed coordinates of
        the line of sight
    For "write" action, display calendar dates in addition to Julian dates
    For "write" action, display the date for range datasets
Modified 2004 December 19 by CM:
    For "write" action, display the projected area for each Doppler and
        delay-Doppler frame
Modified 2004 May 3 by CM:
    For "write" action, display the (delay-)Doppler corrections for each
        frame
Modified 2004 April 9 by CM:
    For "write" action, display the solar azimuth angles (N->E in the POS)
Modified 2004 March 27 by CM:
    Eliminate output of range (rng) plane-of-sky images for
        delay-Doppler frames
    For "write" action, display the epoch, solar phase angle and
        apparent spin vector direction at the midpoint of lightcurve
        datasets
    For "write" action, if "plot_spinvec" parameter is turned on, 
        POS pgm images include an arrow indicating the target's
        intrinsic spin vector.
    For "write" action, if "plot_subradar" parameter is turned on, 
        POS pgm images for (delay-)Doppler datasets include an X
        indicating the target's subradar point.
    For "write" action, if "plot_com" parameter is turned on, 
        POS pgm images for (delay-)Doppler datasets include a cross
        indicating the target's projected COM.
    For "write" action, if "plot_pa" parameter vector has any
        component(s) turned on, POS ppm images for (delay-)Doppler
        datasets include colored cylindrical shaft(s) indicating the
        positive end of the corresponding principal axis/axes.
Modified 2004 Feb 29 by CM:
    Add comments for lightcurves
    Remove "sdev" argument to routine gamma_trans
    Compute lightcurve magnitudes rather than negative magnitudes
    Eliminate the "curve_mm" lightcurve output file, since it nearly
        duplicates the "fit.mm" file (except that the cal factor
        isn't included)
    Move the lightcurve calculations to the new "calc_lghtcrv" routine
    Eliminate the unused dat argument to calc_deldop, calc_doppler,
        and calc_range
    Eliminate "type" argument to the "apply_photo" routine, and
        add the "phase" (solar phase angle) argument
    Label lightcurve POS images as 0 through (ncalc-1) rather than
        1 through ncalc, similar to (delay-)Doppler pgm images
Modified 2003 July 30 by CM:
    Add three parameters for rotating/flipping output pgm files
        for delay-Doppler images (fit, data, residuals)
Modified 2003 May 16 by CM:
    Add listres parameter for producing output files containing
        residual matrices
Modified 2003 May 13 by CM:
    Don't resample and recenter residual pgm images if dd_scaling = none
    Correct a bug in normalizing file output for Doppler fits
Modified 2003 May 10 by CM:
    Add scalefitobs parameter so that user can choose whether to scale
        the data and fit pgm images separately (default), to the maximum
        value of the two taken together, to the maximum fit value, or to
        the maximum data value
Modified 2003 May 7 by CM:
    Add sinc2width argument to pos2deldop and pos2doppler
Modified 2003 April 29 by CM:
    Don't truncate residuals to integer values before making pgm images
    Add nsinc2 argument to pos2deldop and pos2doppler
Modified 2003 April 28 by CM:
    Display two angles for the spin vector, not just one
Modified 2003 April 24 by CM:
    Move "delcom" from delay-Doppler datasets to individual frames
Modified 2003 April 23 by CM:
    Removed "deldopoffs" call from calc_deldop and "dopoffs" call from
        calc_deldop, since these calls are now included in realize_delcor
 *****************************************************************************************/
extern "C" {
#include "../shape/head.h"
}

__host__ void calc_deldop_gpu(struct par_t *dpar, struct mod_t *dmod,
		struct dat_t *ddat, struct vertices_t **verts, int s, int nframes,
		int nviews, unsigned char type, int nf, struct pos_t **pos, struct
		deldopfrm_t **frame, struct deldopview_t **view0, double **fit_store,
		int *ndel, int *ndop, int *posn, int4 *xylim, double
		*overflow, int *outbndarr, cudaStream_t *cf_stream, double **fit_overflow);
__host__ void calc_doppler_gpu(struct par_t *dpar, struct mod_t *dmod,
		struct dat_t *ddat, struct vertices_t **verts, int s, int nframes, int
		nviews, unsigned char type, int nf, struct pos_t **pos, struct dopfrm_t
		**frame, struct dopview_t **view0, double **fit_store, int *ndop, int
		*posn, int4 *xylim, double *overflow, int	*outbndarr,
		cudaStream_t *cf_stream);
__host__ void calc_lghtcrv_gpu(struct par_t *dpar, struct mod_t *dmod,
		struct dat_t *ddat, struct vertices_t **verts, int s, int nframes, int
		nviews, unsigned char type, int lc_n, int nf, struct pos_t **pos,
		struct crvrend_t **rend, int *posn, int *bistatic, double *pxlpkm, double
		*overflow, int4 *xylim, int *outbndarr, double3 *so, double *u,
		cudaStream_t *cf_stream);

__device__ int cfs_v0_index, cfs_exclude_seen;
__device__ double cfs_lghtcrv_posbnd_logfactor;

__device__ void dev_splint_cfs(double *xa,double *ya,double *y2a,int n,double x,double *y)
{
	int klo,khi,k;
	double h,b,a;

	klo = 1;
	khi = n;
	while (khi-klo > 1) {
		k = (khi+klo) >> 1;
		if (xa[k] > x) 	khi=k;
		else klo = k;
	}
	h = xa[khi] - xa[klo];
	if (h == 0.0) 	printf("Bad XA input to routine SPLINT");
	a = (xa[khi] - x) / h;
	b = (x - xa[klo]) / h;
	*y = a * ya[klo] + b * ya[khi] + ((a*a*a-a) * y2a[klo] + (b*b*b-b) *
			y2a[khi]) * (h*h)/6.0;
}
__global__ void cf_init_seen_flags_krnl(struct mod_t *dmod, int nf, int c) {
	/* nf-threaded kernel */
	int f = blockIdx.x * blockDim.x + threadIdx.x;

	if (f < nf)
		dmod->shape.comp[c].real.f[f].seen = 0;
}
__global__ void cf_set_final_pars_krnl(struct par_t *dpar, struct
		dat_t *ddat) {
	/* Single-threaded kernel */
	if (threadIdx.x == 0) {
		dpar->posbnd_logfactor /= ddat->dof;
		dpar->badposet_logfactor /= ddat->dof_poset;
		dpar->badradar_logfactor /= (ddat->dof_deldop + ddat->dof_doppler);
	}
}
__global__ void cfs_init_devpar_krnl(struct par_t *dpar) {
	/* Single-threaded kernel */
	if (threadIdx.x == 0) {
		dpar->posbnd = 0;
		dpar->badposet = 0;
		dpar->badradar = 0;
		dpar->posbnd_logfactor = 0.0;
		dpar->badposet_logfactor = 0.0;
		dpar->badradar_logfactor = 0.0;
	}
}
__global__ void cfs_set_deldop_shortcuts_krnl(struct dat_t *ddat,
		struct deldopfrm_t **frame, struct pos_t **pos,
		struct deldopview_t **view0, int *ndop, int *ndel, double *overflow,
		int *posn, int s, int size) {
	/* nfrm_alloc-threaded kernel */
	/* 	overflow[0] - overflow_o2_store
	 * 	overflow[1] - overflow_m2_store
	 * 	overflow[2] - overflow_xsec_store
	 * 	overflow[3] - overflow_dopmean_store
	 * 	overflow[4] - overflow_delmean_store
	 */
	int f = blockIdx.x * blockDim.x + threadIdx.x;

	if (f < size) {
		frame[f] 	  = &ddat->set[s].desc.deldop.frame[f];
		ndop[f]  	  = frame[f]->ndop;
		ndel[f]	 	  = frame[f]->ndel;
		view0[f] 	  = &frame[f]->view[ddat->set[s].desc.deldop.v0];
		pos[f]		  = &frame[f]->pos;
		posn[f]		  = pos[f]->n;
	}
	__syncthreads();
	if (f==0) {
		cfs_v0_index  = ddat->set[s].desc.deldop.v0;
		overflow[0] = 0.0;
		overflow[1] = 0.0;
		overflow[2] = 0.0;
		overflow[3] = 0.0;
		overflow[4] = 0.0;
	}
}
__global__ void cfs_set_deldop_shortcuts_krnl_mod(struct dat_t *ddat,
		struct deldopfrm_t **frame, struct pos_t **pos,
		struct deldopview_t **view0, int *ndop, int *ndel, double *overflow,
		int *posn, int s, int size, int4 *xylim) {
	/* nfrm_alloc-threaded kernel */
	/* 	overflow[0] - overflow_o2_store
	 * 	overflow[1] - overflow_m2_store
	 * 	overflow[2] - overflow_xsec_store
	 * 	overflow[3] - overflow_dopmean_store
	 * 	overflow[4] - overflow_delmean_store
	 */
	/* This mod version adds xylim as an argument. This kernel will copy out
	 * the POS bounding box limits pos[f]->xlim/ylim so that the posclr kernel
	 * has the lesser workload of clearing out only the pixels inside the
	 * bounding box	 */
	int f = blockIdx.x * blockDim.x + threadIdx.x;

	if (f < size) {
		frame[f] 	  = &ddat->set[s].desc.deldop.frame[f];
		ndop[f]  	  = frame[f]->ndop;
		ndel[f]	 	  = frame[f]->ndel;
		view0[f] 	  = &frame[f]->view[ddat->set[s].desc.deldop.v0];
		pos[f]		  = &frame[f]->pos;
		posn[f]		  = pos[f]->n;
		xylim[f].w	  = pos[f]->xlim[0];
		xylim[f].x	  = pos[f]->xlim[1];
		xylim[f].y	  = pos[f]->ylim[0];
		xylim[f].z	  = pos[f]->ylim[1];
	}
	__syncthreads();
	if (f==0) {
		cfs_v0_index  = ddat->set[s].desc.deldop.v0;
		overflow[0] = 0.0;
		overflow[1] = 0.0;
		overflow[2] = 0.0;
		overflow[3] = 0.0;
		overflow[4] = 0.0;
	}
}
__global__ void cfs_set_doppler_shortcuts_krnl(struct dat_t *ddat,
		struct dopfrm_t **frame, struct pos_t **pos, struct dopview_t **view0,
		double *overflow, int *ndop, int *posn, int s, int size) {
	/* nframes-threaded kernel */
	int f = blockIdx.x * blockDim.x + threadIdx.x;

	if (f < size) {

		frame[f] = &ddat->set[s].desc.doppler.frame[f];
		view0[f] = &frame[f]->view[ddat->set[s].desc.doppler.v0];
		ndop[f]	 = frame[f]->ndop;
		pos[f]	 = &frame[f]->pos;
		posn[f]	= pos[f]->n;
	}
	__syncthreads();

	if (f==0) {
		cfs_v0_index  = ddat->set[s].desc.doppler.v0;
		overflow[0] = 0.0;
		overflow[1] = 0.0;
		overflow[2] = 0.0;
		overflow[3] = 0.0;
		overflow[4] = 0.0;
	}
}
__global__ void cfs_set_doppler_shortcuts_krnl_mod(struct dat_t *ddat,
		struct dopfrm_t **frame, struct pos_t **pos, struct dopview_t **view0,
		double *overflow, int *ndop, int *posn, int s, int size, int4 *xylim) {
	/* nframes-threaded kernel */
	/* This mod version adds xylim as an argument. This kernel will copy out
	 * the POS bounding box limits pos[f]->xlim/ylim so that the posclr kernel
	 * has the lesser workload of clearing out only the pixels inside the
	 * bounding box	 */
	int f = blockIdx.x * blockDim.x + threadIdx.x;

	if (f < size) {

		frame[f] = &ddat->set[s].desc.doppler.frame[f];
		view0[f] = &frame[f]->view[ddat->set[s].desc.doppler.v0];
		ndop[f]	 = frame[f]->ndop;
		pos[f]	 = &frame[f]->pos;
		posn[f]	= pos[f]->n;
		xylim[f].w	  = pos[f]->xlim[0];
		xylim[f].x	  = pos[f]->xlim[1];
		xylim[f].y	  = pos[f]->ylim[0];
		xylim[f].z	  = pos[f]->ylim[1];
	}
	__syncthreads();

	if (f==0) {
		cfs_v0_index  = ddat->set[s].desc.doppler.v0;
		overflow[0] = 0.0;
		overflow[1] = 0.0;
		overflow[2] = 0.0;
		overflow[3] = 0.0;
		overflow[4] = 0.0;
	}
}
__global__ void cfs_set_lghtcrv_shortcuts_krnl(struct dat_t *ddat,
		struct crvrend_t **rend, struct pos_t **pos, double *overflow,
		int *posn, int s, int size, int *outbndarr) {
	/* nfrm_alloc-threaded kernel */
	int f = blockIdx.x * blockDim.x + threadIdx.x + 1;
	if (f <= size) {
		rend[f] = &ddat->set[s].desc.lghtcrv.rend[f];//&cfs_lghtcrv->rend[f];
		pos[f] = &rend[f]->pos;
		posn[f] = pos[f]->n;
		outbndarr[0] = 0;
		if (f==1) {
			posn[0] = 0;
			overflow[0] = 0.0;
			overflow[1] = 0.0;
			overflow[2] = 0.0;
			overflow[3] = 0.0;
			overflow[4] = 0.0;
		}
	}
}
__global__ void cfs_set_lghtcrv_shortcuts_krnl_mod(struct dat_t *ddat,
		struct crvrend_t **rend, struct pos_t **pos, double *overflow,
		int *posn, int s, int size, int *outbndarr, int4 *xylim, int4 *xylim2) {
	/* nfrm_alloc-threaded kernel */
	/* This mod version adds xylim as an argument. This kernel will copy out
	 * the POS bounding box limits pos[f]->xlim/ylim so that the posclr kernel
	 * has the lesser workload of clearing out only the pixels inside the
	 * bounding box	 */
	int f = blockIdx.x * blockDim.x + threadIdx.x + 1;
	if (f <= size) {
		rend[f] = &ddat->set[s].desc.lghtcrv.rend[f];//&cfs_lghtcrv->rend[f];
		pos[f] = &rend[f]->pos;
		posn[f] = pos[f]->n;
		outbndarr[0] = 0;
		xylim[f].w	  = pos[f]->xlim[0];
		xylim[f].x	  = pos[f]->xlim[1];
		xylim[f].y	  = pos[f]->ylim[0];
		xylim[f].z	  = pos[f]->ylim[1];
		xylim2[f].w   = pos[f]->xlim2[0];
		xylim2[f].x   = pos[f]->xlim2[1];
		xylim2[f].y   = pos[f]->ylim2[0];
		xylim2[f].z   = pos[f]->ylim2[1];
		if (f==1) {
			posn[0] = 0;
			overflow[0] = 0.0;
			overflow[1] = 0.0;
			overflow[2] = 0.0;
			overflow[3] = 0.0;
			overflow[4] = 0.0;
			xylim[0].w = xylim[0].x = xylim[0].y = xylim[0].z = 0;
			xylim2[0].w = xylim2[0].x = xylim2[0].y = xylim[0].z = 0;
		}
	}
}
__global__ void cfs_set_pos_ae_krnl(struct dat_t *ddat, struct pos_t **pos,
		int s, int size, unsigned char type, int v, int lghtcrv) {
	/* nfrm_alloc-threaded kernel */
	int i, j, f = blockIdx.x * blockDim.x + threadIdx.x + lghtcrv;

	if (f < (size+lghtcrv)) {
		switch (type) {
		case DELAY:
			for (i=0; i<3; i++) {
				for (j=0; j<3; j++) {
					pos[f]->ae[i][j] = ddat->set[s].desc.deldop.frame[f].view[v].ae[i][j];
					pos[f]->oe[i][j] = ddat->set[s].desc.deldop.frame[f].view[v].oe[i][j];
				}
			}
			break;
		case DOPPLER:
			for (i=0; i<3; i++) {
				for (j=0; j<3; j++) {
					pos[f]->ae[i][j] = ddat->set[s].desc.doppler.frame[f].view[v].ae[i][j];
					pos[f]->oe[i][j] = ddat->set[s].desc.doppler.frame[f].view[v].oe[i][j];
				}
			}
			break;
		case LGHTCRV:
//			for (i=0; i<3; i++) {
//				for (j=0; j<3; j++) {
//					pos[f]->ae[i][j] = ddat->set[s].desc.lghtcrv.rend[f].ae[i][j];
//					pos[f]->oe[i][j] = ddat->set[s].desc.lghtcrv.rend[f].oe[i][j];
//					pos[f]->se[i][j] = ddat->set[s].desc.lghtcrv.rend[f].se[i][j];
//				}
//			}
			pos[f]->ae[0][0] = ddat->set[s].desc.lghtcrv.rend[f].ae[0][0];
			pos[f]->ae[0][1] = ddat->set[s].desc.lghtcrv.rend[f].ae[0][1];
			pos[f]->ae[0][2] = ddat->set[s].desc.lghtcrv.rend[f].ae[0][2];
			pos[f]->ae[1][0] = ddat->set[s].desc.lghtcrv.rend[f].ae[1][0];
			pos[f]->ae[1][1] = ddat->set[s].desc.lghtcrv.rend[f].ae[1][1];
			pos[f]->ae[1][2] = ddat->set[s].desc.lghtcrv.rend[f].ae[1][2];
			pos[f]->ae[2][0] = ddat->set[s].desc.lghtcrv.rend[f].ae[2][0];
			pos[f]->ae[2][1] = ddat->set[s].desc.lghtcrv.rend[f].ae[2][1];
			pos[f]->ae[2][2] = ddat->set[s].desc.lghtcrv.rend[f].ae[2][2];

			pos[f]->oe[0][0] = ddat->set[s].desc.lghtcrv.rend[f].oe[0][0];
			pos[f]->oe[0][1] = ddat->set[s].desc.lghtcrv.rend[f].oe[0][1];
			pos[f]->oe[0][2] = ddat->set[s].desc.lghtcrv.rend[f].oe[0][2];
			pos[f]->oe[1][0] = ddat->set[s].desc.lghtcrv.rend[f].oe[1][0];
			pos[f]->oe[1][1] = ddat->set[s].desc.lghtcrv.rend[f].oe[1][1];
			pos[f]->oe[1][2] = ddat->set[s].desc.lghtcrv.rend[f].oe[1][2];
			pos[f]->oe[2][0] = ddat->set[s].desc.lghtcrv.rend[f].oe[2][0];
			pos[f]->oe[2][1] = ddat->set[s].desc.lghtcrv.rend[f].oe[2][1];
			pos[f]->oe[2][2] = ddat->set[s].desc.lghtcrv.rend[f].oe[2][2];

			pos[f]->se[0][0] = ddat->set[s].desc.lghtcrv.rend[f].se[0][0];
			pos[f]->se[0][1] = ddat->set[s].desc.lghtcrv.rend[f].se[0][1];
			pos[f]->se[0][2] = ddat->set[s].desc.lghtcrv.rend[f].se[0][2];
			pos[f]->se[1][0] = ddat->set[s].desc.lghtcrv.rend[f].se[1][0];
			pos[f]->se[1][1] = ddat->set[s].desc.lghtcrv.rend[f].se[1][1];
			pos[f]->se[1][2] = ddat->set[s].desc.lghtcrv.rend[f].se[1][2];
			pos[f]->se[2][0] = ddat->set[s].desc.lghtcrv.rend[f].se[2][0];
			pos[f]->se[2][1] = ddat->set[s].desc.lghtcrv.rend[f].se[2][1];
			pos[f]->se[2][2] = ddat->set[s].desc.lghtcrv.rend[f].se[2][2];
			break;
		}
		/* Single-thread task */
		if (f == (0+lghtcrv)) {
			if ((type == LGHTCRV) || (type == POS))
				pos[f]->bistatic = 1;
			else if ((type == DELAY) || (type == DOPPLER))
				pos[f]->bistatic = 0;

		}
	}
}
__global__ void cfs_set_posbnd_krnl(struct par_t *dpar, struct dat_t *ddat,
		struct pos_t **pos, int s, int f, unsigned char type) {
	/* Single-threaded kernel */
	if (threadIdx.x == 0) {
		dpar->posbnd = 1;
		switch (type) {
		case DELAY:
			dpar->posbnd_logfactor += ddat->set[s].desc.deldop.frame[f].dof *
				pos[f]->posbnd_logfactor;
			break;
		case DOPPLER:
			dpar->posbnd_logfactor += ddat->set[s].desc.doppler.frame[f].dof *
				pos[f]->posbnd_logfactor;
			break;
		case LGHTCRV:
			if (pos[f]->bistatic)
				cfs_lghtcrv_posbnd_logfactor += 0.5 * pos[f]->posbnd_logfactor;
			else
				cfs_lghtcrv_posbnd_logfactor += pos[f]->posbnd_logfactor;
			break;
		}
	}
}
__global__ void cfs_get_exclude_seen_krnl(struct par_t *dpar, struct pos_t **pos,
		int4 *xylim, int size, int lghtcrv) {
	/* nframes-threaded kernel */
	int f = blockIdx.x * blockDim.x + threadIdx.x + lghtcrv;

	if (f<(size+lghtcrv)) {
		if (lghtcrv) {
			xylim[0].w = xylim[0].x = xylim[0].y = xylim[0].z = 0;
		}
		cfs_exclude_seen = dpar->exclude_seen;
		xylim[f].w = pos[f]->xlim[0];
		xylim[f].x = pos[f]->xlim[1];
		xylim[f].y = pos[f]->ylim[0];
		xylim[f].z = pos[f]->ylim[1];
		if (f==0)
		cfs_exclude_seen = dpar->exclude_seen;
	}
}
__global__ void cf_mark_pixels_seen_krnl(struct par_t *dpar,
		struct mod_t *dmod, struct pos_t **pos, int4 *xylim, int npixels,
		int xspan, int f) {
	/* Multi-threaded kernel */
	/* This kernel doesn't currently work as I removed the pixel facet assignments
	 * in posvis_gpu64	 */
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	int k = (offset % xspan) + xylim[f].w;
	int l = (offset / xspan) + xylim[f].y;
	int facetnum;

	if (offset < npixels) {
		if ((pos[f]->cose[k][l] > dpar->mincosine_seen)
				&& (pos[f]->f[k][l] >= 0)) {
			facetnum = pos[f]->f[k][l];
			//c = cf_pos->comp[k][l];
			dmod->shape.comp[0].real.f[facetnum].seen = 1;
		}
	}
}
__global__ void cf_set_badradar_krnl(struct par_t *dpar,
		struct dat_t *ddat, int s, int f, unsigned char type) {
	/* Single-threaded kernel */
	if (threadIdx.x == 0) {
		switch (type) {
		case DELAY:
			dpar->badradar = 1;
			dpar->badradar_logfactor += ddat->set[s].desc.deldop.frame[f].dof *
					ddat->set[s].desc.deldop.frame[f].badradar_logfactor /
					ddat->set[s].desc.deldop.nviews;
			break;
		case DOPPLER:
			dpar->badradar = 1;
			dpar->badradar_logfactor += ddat->set[s].desc.doppler.frame[f].dof *
					ddat->set[s].desc.doppler.frame[f].badradar_logfactor /
					ddat->set[s].desc.doppler.nviews;
			break;
		}
	}
}
__global__ void cf_add_fit_store_krnl1(struct dat_t *ddat, double **fit_store,
		int nThreads, int s, int f, unsigned char type, int *ndel) {
	/* ndel*ndop-threaded kernel */
	int idel, idop, offset = blockIdx.x * blockDim.x + threadIdx.x;
	idel = offset % ndel[f] + 1;
	idop = offset / ndel[f] + 1;

	if (offset < (nThreads)) {
		switch (type) {
		case DELAY:
			fit_store[f][offset] += ddat->set[s].desc.deldop.frame[f].fit[idel][idop];
			break;
		case DOPPLER:
			fit_store[f][offset] += ddat->set[s].desc.doppler.frame[f].fit[idop];
			break;
		}
	}
}
__global__ void cf_add_fit_store_krnl2(struct dat_t *ddat, int s, int f,
		double *overflow, unsigned char type) {
	/* ndel*ndop-threaded kernel */
	if (threadIdx.x == 0) {
		switch (type) {
		case DELAY:
			atomicAdd(&overflow[0], ddat->set[s].desc.deldop.frame[f].overflow_o2);
			atomicAdd(&overflow[1], ddat->set[s].desc.deldop.frame[f].overflow_m2);
			atomicAdd(&overflow[2], ddat->set[s].desc.deldop.frame[f].overflow_xsec);
			atomicAdd(&overflow[3], ddat->set[s].desc.deldop.frame[f].overflow_delmean);
			atomicAdd(&overflow[4], ddat->set[s].desc.deldop.frame[f].overflow_dopmean);
			break;
		case DOPPLER:
			atomicAdd(&overflow[0], ddat->set[s].desc.doppler.frame[f].overflow_o2);
			atomicAdd(&overflow[1], ddat->set[s].desc.doppler.frame[f].overflow_m2);
			atomicAdd(&overflow[2], ddat->set[s].desc.doppler.frame[f].overflow_xsec);
			atomicAdd(&overflow[3], ddat->set[s].desc.doppler.frame[f].overflow_dopmean);
		}
	}
}
__global__ void cf_finish_fit_store_krnl(struct dat_t *ddat, double **fit_store,
		int s, int f, int nThreads, unsigned char type, int *ndel) {
	/* multi-threaded kernel */
	int idel, idop, offset = blockIdx.x * blockDim.x + threadIdx.x;
	idel = offset % ndel[f] + 1;
	idop = offset / ndel[f] + 1;
	if (offset < nThreads) {
		switch (type) {
		case DELAY:
			ddat->set[s].desc.deldop.frame[f].fit[idel][idop] = fit_store[f][offset];
			break;
		case DOPPLER:
			ddat->set[s].desc.doppler.frame[f].fit[idop] = fit_store[f][offset];
			break;
		}
	}
}
__global__ void cf_finish_fit_krnl2(struct dat_t *ddat,
		double *overflow, int s, int f, unsigned char type) {
	/* Single-threaded Kernel */
	int nviews;
	if (threadIdx.x == 0) {
		switch (type) {
		case DELAY:
			nviews = ddat->set[s].desc.deldop.nviews;
			ddat->set[s].desc.deldop.frame[f].overflow_o2 = overflow[0] / nviews;
			ddat->set[s].desc.deldop.frame[f].overflow_m2 = overflow[1] / nviews;
			ddat->set[s].desc.deldop.frame[f].overflow_xsec = overflow[2] / nviews;
			ddat->set[s].desc.deldop.frame[f].overflow_delmean = overflow[3] / nviews;
			ddat->set[s].desc.deldop.frame[f].overflow_dopmean = overflow[4] / nviews;
			break;
		case DOPPLER:
			nviews = ddat->set[s].desc.doppler.nviews;
			ddat->set[s].desc.doppler.frame[f].overflow_o2 = overflow[0] / nviews;
			ddat->set[s].desc.doppler.frame[f].overflow_m2 = overflow[1] / nviews;
			ddat->set[s].desc.doppler.frame[f].overflow_xsec = overflow[2] / nviews;
			ddat->set[s].desc.doppler.frame[f].overflow_dopmean = overflow[3] / nviews;
		}
	}
}
__global__ void cf_gamma_trans_krnl(struct par_t *dpar, struct dat_t *ddat,
		int s, int f, int nThreads, unsigned char type, int *ndel) {
	/* Multi-threaded kernel */
	int idel, idop, offset = blockIdx.x * blockDim.x + threadIdx.x;
	idel = offset % ndel[f] + 1;
	idop = offset / ndel[f] + 1;
	if (offset < nThreads) {
		/*  Carry out a gamma transformation on the fit image if requested  */
		if (dpar->dd_gamma != 1.0) {
			switch (type) {
			case DELAY:
				dev_gamma_trans(&ddat->set[s].desc.deldop.frame[f].fit[idel][idop],
						dpar->dd_gamma);
				break;
			case DOPPLER:
				//cf_dop_frame->fit[offset] = fit[offset];
				break;
			}
		}
	}
}

__host__ void calc_fits_gpu(
		struct par_t *dpar,
		struct mod_t *dmod,
		struct dat_t *ddat,
		struct vertices_t **verts,
		int *nviews,
		int *nframes,
		int *lc_n,
		unsigned char *type,
		int nsets,
		int nf,
		cudaStream_t *cf_stream,
		int max_frames,
		double **fit_overflow)
{
	int s, nfrm_alloc, *ndel, *ndop, *posn, *bistatic, *outbndarr;
	int4 *xylim;
	double **fit_store, *overflow, *pxlpkm;
	double *u;
	double3 *so;
	nfrm_alloc = max_frames+1;

	struct pos_t **pos;
	struct deldopfrm_t **ddframe;
	struct dopfrm_t **dframe;
	struct crvrend_t **rend;
	struct deldopview_t **ddview0;
	struct dopview_t **dview0;
	dim3 BLK,THD;
	THD.x = maxThreadsPerBlock;

	/* Allocate memory for all arrays that are needed for any possible data set.
	 * This is done to avoid repeat allocations/deallocations	 */
	cudaCalloc1((void**)&pos, sizeof(pos_t*), nfrm_alloc);
	gpuErrchk(cudaMalloc((void**)&ddframe, sizeof(deldopfrm_t*) * nfrm_alloc));
	gpuErrchk(cudaMalloc((void**)&ddview0, sizeof(deldopview_t*) * nfrm_alloc));
	gpuErrchk(cudaMalloc((void**)&dframe, sizeof(dopfrm_t*) * nfrm_alloc));
	gpuErrchk(cudaMalloc((void**)&dview0, sizeof(dopview_t*) * nfrm_alloc));
	gpuErrchk(cudaMalloc((void**)&rend, sizeof(crvrend_t*) * nfrm_alloc));
	gpuErrchk(cudaMalloc((void**)&ndel, sizeof(int) * nfrm_alloc));
	gpuErrchk(cudaMalloc((void**)&ndop, sizeof(int) * nfrm_alloc));
	gpuErrchk(cudaMalloc((void**)&posn, sizeof(int) * nfrm_alloc));
	gpuErrchk(cudaMalloc((void**)&bistatic, sizeof(int) * nfrm_alloc));
	gpuErrchk(cudaMalloc((void**)&xylim, sizeof(int4) * nfrm_alloc));
	gpuErrchk(cudaMalloc((void**)&outbndarr, sizeof(int) * nfrm_alloc));
	gpuErrchk(cudaMalloc((void**)&u, sizeof(double) * nfrm_alloc));
	gpuErrchk(cudaMalloc((void**)&so, 	 	sizeof(double3) *((nfrm_alloc*3)+1)));
	gpuErrchk(cudaMalloc((void**)&fit_store, sizeof(double*) * nfrm_alloc));
	gpuErrchk(cudaMalloc((void**)&overflow, sizeof(double) * 6));
	gpuErrchk(cudaMalloc((void**)&pxlpkm,   sizeof(double) * nfrm_alloc));

	/* Initialize flags that indicate the model extends beyond POS frame, that
	 * plane-of-sky fit images are too small to "contain" the target, and that
	 * model is too wide in (delay-)Doppler space to create (delay-)Doppler fit
	 * frames.  Note that this also gets mod->shape.nf and nsets            */

	cfs_init_devpar_krnl<<<1,1>>>(dpar);
	checkErrorAfterKernelLaunch("cfs_init_devpar_krnl2");

	/* Initialize the flags that indicate whether or not each facet of each
	 * model component is ever visible and unshadowed from Earth
	 * Note:  Single component only for now.  */
	int c = 0;
	BLK.x = floor((THD.x - 1 + nf)/THD.x);
	cf_init_seen_flags_krnl<<<BLK,THD>>>(dmod,nf,c);
	checkErrorAfterKernelLaunch("cf_init_seen_flags_krnl (calc_fits_cuda_streams)");

	/* Calculate the fits for each dataset in turn  */
	for (s=0; s<nsets; s++) {

		switch (type[s]) {
		case DELAY:
			calc_deldop_gpu(dpar, dmod, ddat, verts, s, nframes[s], nviews[s],
					type[s], nf, pos, ddframe, ddview0, fit_store, ndel, ndop,
					posn, xylim, overflow, outbndarr, cf_stream, fit_overflow);
			break;
		case DOPPLER:
			calc_doppler_gpu(dpar, dmod, ddat, verts, s, nframes[s],
					nviews[s], type[s], nf, pos, dframe, dview0, fit_store,
					ndop, posn, xylim, overflow, outbndarr,
					cf_stream );
			break;
		case POS:
			printf("Write calc_poset_cuda!");
			//			calc_poset_cuda(dpar, dmod, s);
			break;
		case LGHTCRV:
			calc_lghtcrv_gpu(dpar, dmod, ddat, verts, s, nframes[s],
					nviews[s], type[s], lc_n[s],nf, pos, rend, posn, bistatic,
					pxlpkm, overflow, xylim, outbndarr, so, u, cf_stream);
			break;
		default:
			printf("calc_fits_gpu.c: can't handle this type yet\n");
		}
	}


	/* Complete calculations of values that will be used during a fit to
	 * increase the objective function for models with bad properties   */
	cf_set_final_pars_krnl<<<1,1>>>(dpar, ddat);
	checkErrorAfterKernelLaunch("cf_set_final_pars_krnl");

	cudaFree(fit_store);
	cudaFree(overflow);
	cudaFree(pxlpkm);

	cudaFree(pos);
	cudaFree(ddframe);
	cudaFree(dframe);
	cudaFree(rend);
	cudaFree(ddview0);
	cudaFree(dview0);
	cudaFree(ndel);
	cudaFree(ndop);
	cudaFree(posn);
	cudaFree(bistatic);
	cudaFree(xylim);
	cudaFree(outbndarr);
	cudaFree(u);
	cudaFree(so);
}

__host__ void calc_deldop_gpu(struct par_t *dpar, struct mod_t *dmod,
		struct dat_t *ddat, struct vertices_t **verts, int s, int nframes,
		int nviews, unsigned char type, int nf, struct pos_t **pos, struct
		deldopfrm_t **frame, struct deldopview_t **view0, double **fit_store,
		int *ndel, int *ndop, int *posn, int4 *xylim, double
		*overflow, int *outbndarr, cudaStream_t *cf_stream, double **fit_overflow)
{
	double3 orbit_off3;
	orbit_off3.x = orbit_off3.y = orbit_off3.z = 0.0;

	int v0_index, exclude_seen, f, v2, c=0, yspan, hndop[nframes], hndel[nframes],
			hposn[nframes], houtbndarr[nframes], nThreadsdd[nframes], v[nviews+1],
			*dxspan, *hxspan, *dnpixels, *hnpixels, *ddeldopsize, *hdeldopsize,
			blocks;
	int4 hxylim[nframes];
	dim3 BLKdd[nframes], BLKpx[nframes], THD, BLKfrm,THD64, THDaf, BLKaf;
	THD.x = maxThreadsPerBlock; THD64.x = 64;
	BLKfrm = floor((THD64.x - 1 + nframes)/THD64.x);

	/* The following variables are the launch parameters for the posclr_radar_krnl64af
	 * kernel, which clears all frames simultaneously with a configurable number of
	 * blocks per frame (set via "blocks").  Each block/set of blocks will perform
	 * a grid-stride loop through the pos bounding box for each frame 	 */
	THDaf.x = 1024;		blocks = 3; 	BLKaf.x = nframes*blocks;

	cudaCalloc1((void**)&dnpixels, sizeof(int), nframes);
	cudaCalloc1((void**)&dxspan, sizeof(int), nframes);
	cudaCalloc1((void**)&ddeldopsize, sizeof(int), nframes);
	hnpixels = (int *) malloc(nframes*sizeof(int));
	hxspan = (int *) malloc(nframes*sizeof(int));
	hdeldopsize = (int *) malloc(nframes*sizeof(int));

	/* Set deldop, frame, view0, and pos in nframes streamed kernels */
	cfs_set_deldop_shortcuts_krnl_mod<<<BLKfrm,THD64>>>(ddat, frame, pos,
			view0, ndop, ndel, overflow, posn, s, nframes, xylim);
	checkErrorAfterKernelLaunch("cfs_set_deldop_shortcuts_krnl");
	gpuErrchk(cudaMemcpy(&hndel, ndel, 	sizeof(int)*nframes, 	cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&hndop, ndop, 	sizeof(int)*nframes, 	cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&hposn, posn, 	sizeof(int)*nframes, 	cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&hxylim, xylim,sizeof(int4)*nframes, 	cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpyFromSymbol(&v0_index, cfs_v0_index, sizeof(int),
			0, cudaMemcpyDeviceToHost));

	for (f=0; f<nframes; f++) {
		nThreadsdd[f] = hndel[f]*hndop[f];
		hdeldopsize[f] = hndel[f]*hndop[f];
		BLKdd[f].x = floor((THD.x - 1 + hdeldopsize[f]) / THD.x);
		hxspan[f] = hxylim[f].x - hxylim[f].w + 1;
		yspan = hxylim[f].z - hxylim[f].y + 1;
		hnpixels[f] = hxspan[f] * yspan;
		BLKpx[f].x = (THD.x -1 + hnpixels[f]) / THD.x;

		/* If smearing is being modeled, initialize variables that
		 * will be used to sum results calculated for individual views.  */
		if (nviews > 1)
			/* Allocate fit_store as a single pointer, originally a double
			 * pointer. This also initializes the entire array to zero. */
			gpuErrchk(cudaMalloc((void**)&fit_store[f], sizeof(double) * nThreadsdd[f]));
	}
	gpuErrchk(cudaMemcpy(dnpixels, hnpixels, sizeof(int)*nframes,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dxspan, hxspan, sizeof(int)*nframes,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(ddeldopsize, hdeldopsize, sizeof(int)*nframes, cudaMemcpyHostToDevice));

	/*  Loop over all views for this (smeared) frame, going in an order that
        ends with the view corresponding to the epoch listed for this frame
        in the obs file; this way we can use the calculated information for
        that view in the "write" action screen and disk output that follows   */
	for (v2=v0_index+1; v2<=v0_index+nviews; v2++) {
		v[v2] = v2 % nviews;
		/* Launch 9-threaded kernel to set pos->ae,pos->oe,pos->bistatic.*/
		cfs_set_pos_ae_krnl<<<BLKfrm,THD64, 0, cf_stream[v2]>>>(ddat, pos, s, nframes,
				type, v[v2], 0);
	}

	posclr_radar_krnl_af<<<BLKaf,THDaf>>>(pos, posn, dxspan, xylim, dnpixels,
			blocks);
	checkErrorAfterKernelLaunch("posclr_radar_krnl64af ");

	/* Call posvis to get facet number, scattering angle, distance
	 * toward Earth at center of each POS pixel; set flag posbnd if any model
	 * portion extends beyond POS frame limits.
	 * NOTE: Limited to single component for now */
	for (v2=v0_index+1; v2<=v0_index+nviews; v2++)
		posvis_gpu(dpar, dmod, pos, verts, orbit_off3, hposn,
				outbndarr, s, nframes, 0, nf, 0, c, type, cf_stream, 0);

	gpuErrchk(cudaMemcpy(&houtbndarr, outbndarr, sizeof(int)*nframes,
			cudaMemcpyDeviceToHost));

	for (f=0; f<nframes; f++) {
		for (v2=v0_index+1; v2<=v0_index+nviews; v2++) {
			if ((houtbndarr[f]) && (v[v2] == v0_index)) {
				/* Call single-threaded kernel to set dpar->posbnd and dpar->posbnd_logfactor */
				cfs_set_posbnd_krnl<<<1,1,0,cf_stream[f]>>>(dpar, ddat, pos, s, f, type);
				houtbndarr[f]=0;
			}
		}
	}checkErrorAfterKernelLaunch("cfs_set_posbnd_krnl");

	/* Get xlim and ylim and exclude_seen flag */
	cfs_get_exclude_seen_krnl<<<BLKfrm,THD64>>>(dpar,pos,xylim,nframes, 0);
	checkErrorAfterKernelLaunch("cfs_get_exclude_krnl");

	/* Now copy the flag and all frame pos's xlim and ylim values back from GPU */
	gpuErrchk(cudaMemcpyFromSymbol(&exclude_seen, cfs_exclude_seen, sizeof(int),
			0, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&hxylim, xylim, sizeof(int4)*nframes, cudaMemcpyDeviceToHost));

	/* Calculate launch parameters for all frames */
	for (f=0; f<nframes; f++) {
		hxspan[f] = hxylim[f].x - hxylim[f].w + 1;
		yspan = hxylim[f].z - hxylim[f].y + 1;
		hnpixels[f] = hxspan[f] * yspan;
		BLKpx[f].x = (THD.x -1 + hnpixels[f]) / THD.x;
	}

	for (f=0; f<nframes; f++) {
		for (v2=v0_index+1; v2<=v0_index+nviews; v2++) {
			/* Go through all POS pixels which are visible with low enough
			 * scattering angle and mark the facets which project onto their
			 * centers as having been "seen" at least once                   */
			if (s != exclude_seen && v[v2] == v0_index)
				cf_mark_pixels_seen_krnl<<<BLKpx[f],THD,0,cf_stream[f]>>>(
						dpar, dmod, pos, xylim, hnpixels[f], hxspan[f], f);
		}
	} checkErrorAfterKernelLaunch("clrvect_krnl and cf_mark_pixels_seen_streams_krnl");

	clrvect_krnl_af<<<BLKaf,THDaf>>>(ddat, ddeldopsize, s, blocks);
	checkErrorAfterKernelLaunch("clrvect_krnl and cf_mark_pixels_seen_streams_krnl");

	for (v2=v0_index+1; v2<=v0_index+nviews; v2++)
		pos2deldop_gpu(dpar, dmod, ddat, pos, frame, xylim, ndel, ndop,
				0.0,0.0,0.0,0, s, nframes, v[v2], outbndarr, cf_stream, fit_overflow);

	if (nframes < 1) printf("NFRAMES TOO SMALL\n");

	/* Copy the badradar flag returns for all frames to a host copy */
	gpuErrchk(cudaMemcpy(&houtbndarr, outbndarr, sizeof(int)*nframes,
			cudaMemcpyDeviceToHost));

	for (f=0; f<nframes; f++) {
		for (v2=v0_index+1; v2<=v0_index+nviews; v2++) {
			if (houtbndarr[f]) {
				/* Call single-threaded kernel to set badradar flag and
				 * associated badradar_logfactor			 */
				cf_set_badradar_krnl<<<1,1,0,cf_stream[f]>>>(dpar, ddat, s, f, type);
			}
		}
	}
	/* If smearing is being modeled, include delay-Doppler calculations
	 * from this view in the summed results for this frame  */
	if (nviews > 1) {
		/* Launch ndel*ndop-threaded kernel to add fit[i][j] to fit_store[i][j]*/
		cf_add_fit_store_krnl1<<<BLKdd[f],THD,0,cf_stream[f]>>>(
				ddat,fit_store,nThreadsdd[f],s,f, type, ndel);
		cf_add_fit_store_krnl2<<<1,1>>>(ddat, s, f, overflow, type);
	} checkErrorAfterKernelLaunch("cf_deldop_set_badradar_krnl (calc_fits_cuda)");

	/* If smearing is being modeled, compute mean values over all views for
	 * this frame and store them in the standard frame structure.  This kernel
	 * also carries out the gamma transformation on the fit image if the
	 * par->dd_gamma flag is not set  */
	if (nviews > 1) {
		for (f=0; f<nframes; f++) {
			cf_finish_fit_store_krnl<<<BLKdd[f],THD,0,cf_stream[f]>>>(
					ddat, fit_store, s, f, nThreadsdd[f], type, ndel);

			cf_finish_fit_krnl2<<<1,1,0,cf_stream[f]>>>(ddat, overflow, s, f, type);
		} checkErrorAfterKernelLaunch("cf_finish_fit_store_streams kernels");
		cudaFree(fit_store);
	}

	for (f=0; f<nframes; f++) {
		cf_gamma_trans_krnl<<<BLKdd[f],THD,0,cf_stream[f]>>>(dpar, ddat, s, f,
				nThreadsdd[f], type, ndel);
	} checkErrorAfterKernelLaunch("cf_gamma_trans_krnl");

	cudaFree(dxspan);
	cudaFree(dnpixels);
	cudaFree(ddeldopsize);
	free(hxspan);
	free(hnpixels);
	free(hdeldopsize);
}

__host__ void calc_doppler_gpu(struct par_t *dpar, struct mod_t *dmod,
		struct dat_t *ddat, struct vertices_t **verts, int s, int nframes,
		int nviews, unsigned char type, int nf, struct pos_t **pos, struct
		dopfrm_t **frame, struct dopview_t **view0, double **fit_store, int
		*ndop, int *posn, int4 *xylim, double *overflow, int
			*outbndarr, cudaStream_t *cf_stream)
	{
		double3 orbit_off3;
		orbit_off3.x = orbit_off3.y = orbit_off3.z = 0.0;

	int4 hxylim[nframes];
	int v0_index, exclude_seen, f, v2, c=0, yspan, blocks, hndop[nframes],
			hposn[nframes], houtbndarr[nframes], v[nviews+1], *dnpixels,
			*hnpixels, *dxspan, *hxspan;

	dim3 BLKd[nframes], BLKpx[nframes],THD,THD64,BLKfrm, THDaf, BLKaf;
	THD.x = maxThreadsPerBlock; THD64.x = 64;
	BLKfrm = floor((THD64.x - 1 + nframes)/THD64.x);
	blocks = 3;
	THDaf.x = 1024;
	BLKaf.x = blocks*nframes;

	cudaCalloc1((void**)&dnpixels, sizeof(int), nframes);
	cudaCalloc1((void**)&dxspan, sizeof(int), nframes);
	hnpixels = (int *) malloc(nframes*sizeof(int));
	hxspan = (int *) malloc(nframes*sizeof(int));

	/* Set doppler, frame, view0, and pos in nframes-threaded kernels */
	cfs_set_doppler_shortcuts_krnl_mod<<<BLKfrm,THD64>>>(ddat, frame,
			pos, view0, overflow, ndop, posn, s, nframes, xylim);
	checkErrorAfterKernelLaunch("cfs_set_doppler_shortcuts_krnl64");
	gpuErrchk(cudaMemcpy(&hndop, ndop, 	sizeof(int)*nframes,	cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&hposn, posn, 	sizeof(int)*nframes,	cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&hxylim, xylim,sizeof(int4)*nframes, 	cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpyFromSymbol(&v0_index, cfs_v0_index, sizeof(int), 0,
			cudaMemcpyDeviceToHost));

	for (f=0; f<nframes; f++) {
		/* Calculate launch parameters needed later */
		BLKd[f] = floor((THD.x - 1 + hndop[f]) / THD.x);
		hxspan[f] = hxylim[f].x - hxylim[f].w + 1;
		yspan = hxylim[f].z - hxylim[f].y + 1;
		hnpixels[f] = hxspan[f] * yspan;
		BLKpx[f].x = (THD.x -1 + hnpixels[f]) / THD.x;

		/* If smearing is being modeled, initialize variables that
		 * will be used to sum results calculated for individual views.  */
		if (nviews > 1)
			/* Allocate fit_store as a single pointer, originally a double
			 * pointer. This also initializes the entire array to zero. */
			gpuErrchk(cudaMalloc((void**)&fit_store[f], sizeof(double)*ndop[f]));
	}
	/* Now copy back the bounding box information (# of pixels and widths, for each frame) */
	gpuErrchk(cudaMemcpy(dnpixels, hnpixels, sizeof(int)*nframes,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dxspan, hxspan, sizeof(int)*nframes,cudaMemcpyHostToDevice));

	/* Loop over all views for this (smeared) frame, going in an order that
	 * ends with the view corresponding to the epoch listed for this frame
	 * in the obs file; this way we can use the calculated information for
	 * that view in the "write" action screen and disk output that follows*/
	for (v2=v0_index+1; v2<=v0_index+nviews; v2++) {
		v[v2] = v2 % nviews;
		/* Launch kernel to set pos->ae,pos->oe,pos->bistatic.*/
		cfs_set_pos_ae_krnl<<<BLKfrm,THD64,0,cf_stream[v2]>>>(ddat, pos, s,
				nframes,type,v[v2], 0);
	} checkErrorAfterKernelLaunch("cfs_set_pos_ae_krnl");

	posclr_radar_krnl_af<<<BLKaf,THDaf>>>(pos, posn, dxspan, xylim, dnpixels,
			blocks);
	checkErrorAfterKernelLaunch("posclr_radar_krnl_af ");

	/* Call posvis_cuda_2 to get facet number, scattering angle, distance
	 * toward Earth at center of each POS pixel; set flag posbnd if any model
	 * portion extends beyond POS frame limits.*/
	/* NOTE: Limited to single component for now */
	for (v2=v0_index+1; v2<=v0_index+nviews; v2++)
		posvis_gpu(dpar, dmod, pos, verts, orbit_off3, hposn,
				outbndarr, s, nframes, 0, nf, 0, c, type, cf_stream, 0);

	/* Copy the posbnd flag returns for all frames to a host copy */
	gpuErrchk(cudaMemcpy(&houtbndarr, outbndarr, sizeof(int)*nframes,
			cudaMemcpyDeviceToHost));

	for (f=0; f<nframes; f++)
		for (v2=v0_index+1; v2<=v0_index+nviews; v2++) {
			if ((houtbndarr[f]) && (v[v2] == v0_index)) {
				/* Call single-threaded kernel to set dpar->posbnd and dpar->posbnd_logfactor */
				cfs_set_posbnd_krnl<<<1,1,0,cf_stream[f]>>>(dpar, ddat, pos, s, f, type);
				houtbndarr[f]=0;
			} checkErrorAfterKernelLaunch("cfs_set_posbnd_krnl");
		}

	/* Get xlim and ylim and exclude_seen flag */
	cfs_get_exclude_seen_krnl<<<BLKfrm,THD64>>>(dpar,pos,xylim,nframes, 0);
	checkErrorAfterKernelLaunch("cfs_get_exclude_krnl");

	/* Now copy the flag and all frame pos's xlim and ylim values back from GPU */
	gpuErrchk(cudaMemcpyFromSymbol(&exclude_seen, cfs_exclude_seen, sizeof(int),
			0, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&hxylim, xylim, sizeof(int4)*nframes, cudaMemcpyDeviceToHost));

	/* Calculate launch parameters for all frames */
	for (f=0; f<nframes; f++) {
		hxspan[f] = hxylim[f].x - hxylim[f].w + 1;
		yspan = hxylim[f].z - hxylim[f].y + 1;
		hnpixels[f] = hxspan[f] * yspan;
		BLKpx[f].x = (THD.x -1 + hnpixels[f]) / THD.x;
	}

	for (f=0; f<nframes; f++) {
		for (v2=v0_index+1; v2<=v0_index+nviews; v2++) {
			/* Go through all POS pixels which are visible with low enough
			 * scattering angle and mark the facets which project onto
			 * their centers as having been "seen" at least once           */
			if (s != exclude_seen && v[v2] == v0_index)
				cf_mark_pixels_seen_krnl<<<BLKpx[f],THD,0,cf_stream[f]>>>(
						dpar, dmod, pos, xylim, hnpixels[f], hxspan[f], f);

			/* Zero out the fit delay-Doppler image, then call pos2deldop to
			 * create the fit image by mapping power from the plane of the sky
			 * to delay-Doppler space.                             */
			clrvect_krnl<<<BLKd[f],THD,0,cf_stream[f]>>>(ddat, hndop[f], s, f);
		} checkErrorAfterKernelLaunch("clrvect_krnl and cf_mark_pixels_seen_krnl");
	}

	/* Call pos2deldop to calculate the Doppler radar fit image */
	for (v2=v0_index+1; v2<=v0_index+nviews; v2++) {
		pos2doppler_gpu(dpar, dmod, ddat, pos, frame, xylim,0.0,0.0,0.0, ndop,
				0, s, nframes, v[v2], outbndarr, cf_stream);
	}
	/* Copy the badradar flag returns for all frames to a host copy */
	gpuErrchk(cudaMemcpy(&houtbndarr, outbndarr, sizeof(int)*nframes,
			cudaMemcpyDeviceToHost));

	for (f=0; f<nframes; f++) {
		for (v2=v0_index+1; v2<=v0_index+nviews; v2++) {
			if (houtbndarr[f]) {
				/* Call single-threaded kernel to set badradar flag and
				 * associated badradar_logfactor			 */
				cf_set_badradar_krnl<<<1,1,0,cf_stream[f]>>>(dpar, ddat, s, f, type);
				checkErrorAfterKernelLaunch("cf_set_badradar_krnl");
			}
		}
		/* If smearing is being modeled, include delay-Doppler calculations
		 * from this view in the summed results for this frame  */
		if (nviews > 1) {
			/* Launch ndop-threaded kernel to add fit[i][j] to
			 * fit_store[i][j]*/
			cf_add_fit_store_krnl1<<<BLKd[f],THD,0,cf_stream[f]>>>(
					ddat, fit_store, ndop[f], s, f, type, ndop);

			cf_add_fit_store_krnl2<<<1,1>>>(ddat, s, f, overflow, type);
		}
	}

	/* If smearing is being modeled, compute mean values over all views for
	 * this frame and store them in the standard frame structure     */
	/* This kernel also carries out the gamma transformation on the fit
	 * image if the par->dd_gamma flag is not set  */
	if (nviews > 1) {
		for (f=0; f<nframes; f++) {

			cf_finish_fit_store_krnl<<<BLKd[f],THD,0,cf_stream[f]>>>(
					ddat, fit_store, s, f, ndop[f], type, ndop);

			cf_finish_fit_krnl2<<<1,1,0,cf_stream[f]>>>(ddat, overflow, s, f, type);

		} checkErrorAfterKernelLaunch("cf_finish_fit_store_kernels and "
				"cf_gamma_trans_krnl");
		cudaFree(fit_store);
	}

	cudaFree(dnpixels);
	cudaFree(dxspan);
	free(hnpixels);
	free(hxspan);
}

__host__ void calc_lghtcrv_gpu(
		struct par_t *dpar,
		struct mod_t *dmod,
		struct dat_t *ddat,
		struct vertices_t **verts,
		int s,
		int nframes,
		int nviews,
		unsigned char type,
		int lc_n,
		int nf,
		struct pos_t **pos,
		struct crvrend_t **rend,
		int *posn,
		int *bistatic,
		double *pxlpkm,
		double *overflow,
		int4 *xylim,
		int *outbndarr,
		double3 *so,
		double *u,
		cudaStream_t *cf_stream)
{
	int c=0, n, nThreads, exclude_seen, f, *dnpixels_bbox,
			*hnpixels_full, *hnpixels_bbox, *dnpixels_bistatic,
			*hnpixels_bistatic,	*dxspan, *hxspan, blocks, *dxspan_bistatic,
			*hxspan_bistatic, *dnpixels_combined, *hnpixels_combined,
			*dxspan_combined, *hxspan_combined;
	int nfplus = nframes+1; /* This is to accomodate the +1 start in lghtcrv */
	int houtbndarr[nfplus],
		hposn[nfplus];

	double3 orbit_off3;
	orbit_off3.x = orbit_off3.y = orbit_off3.z = 0.0;

	int4 hxylim[nfplus], hxylim2[nfplus], hxylim_combined[nfplus], *xylim2, *xylim_combined;
	int2 span[nfplus];

	dim3 BLKpx_full[nfplus], BLKpx_bbox[nfplus], THD, THD64, BLKfrm, BLKaf, THDaf;
	THD.x = maxThreadsPerBlock; THD64.x = 64;
	BLKfrm = floor((THD64.x - 1 + nframes)/THD64.x);

	n = lc_n;

	/* The following variables are the launch parameters for the posclr_radar_krnl64af
	 * kernel, which clears all frames simultaneously with a configurable number of
	 * blocks per frame (set via "blocks").  Each block/set of blocks will perform
	 * a grid-stride loop through the pos bounding box for each frame 	 */
	THDaf.x = 1024;		blocks = 3; 	BLKaf.x = nframes*blocks;

	cudaCalloc1((void**)&dnpixels_bbox, sizeof(int), nfplus);
	cudaCalloc1((void**)&dnpixels_bistatic, sizeof(int), nfplus);
	cudaCalloc1((void**)&dnpixels_combined, sizeof(int), nfplus);
	cudaCalloc1((void**)&dxspan, sizeof(int), nfplus);
	cudaCalloc1((void**)&dxspan_bistatic, sizeof(int), nfplus);
	cudaCalloc1((void**)&dxspan_combined, sizeof(int), nfplus);
	cudaCalloc1((void**)&xylim2, sizeof(int4), nfplus);
	cudaCalloc1((void**)&xylim_combined, sizeof(int4), nfplus);
	hnpixels_full = (int *) malloc(nfplus*sizeof(int));
	hnpixels_bbox = (int *) malloc(nfplus*sizeof(int));
	hnpixels_bistatic = (int *) malloc(nfplus*sizeof(int));
	hnpixels_combined = (int *) malloc(nfplus*sizeof(int));
	hxspan = (int *) malloc(nfplus*sizeof(int));
	hxspan_bistatic = (int *) malloc(nfplus*sizeof(int));
	hxspan_combined = (int *) malloc(nfplus*sizeof(int));

	/* Set shortcuts and copy pos->n back for all frames */
	cfs_set_lghtcrv_shortcuts_krnl_mod<<<BLKfrm,THD64>>>(ddat, rend, pos, overflow,
			posn, s, nframes, outbndarr, xylim, xylim2);
	checkErrorAfterKernelLaunch("cfs_set_lghtcrv_shortcuts_krnl64mod");
	gpuErrchk(cudaMemcpy(&hxylim, xylim, sizeof(int4)*nfplus, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&hxylim2, xylim2, sizeof(int4)*nfplus, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&hposn, posn, sizeof(int)* nfplus,
			cudaMemcpyDeviceToHost));

	/* Calculate launch parameters for later */
	for (f=1; f<=nframes; f++) {
		hxspan[f] = hxylim[f].x - hxylim[f].w + 1;
		hnpixels_bbox[f] = (hxylim[f].z - hxylim[f].y +1) * hxspan[f];
		hxspan_bistatic[f] = hxylim2[f].x - hxylim2[f].w + 1;
		hnpixels_bistatic[f] = (hxylim2[f].z - hxylim2[f].y + 1) * hxspan_bistatic[f];
		hxylim_combined[f].w = min(hxylim[f].w, hxylim2[f].w);
		hxylim_combined[f].x = max(hxylim[f].x, hxylim2[f].x);
		hxylim_combined[f].y = min(hxylim[f].y, hxylim2[f].y);
		hxylim_combined[f].z = max(hxylim[f].z, hxylim2[f].z);
		hxspan_combined[f] = hxylim_combined[f].x - hxylim_combined[f].w + 1;
		hnpixels_combined[f] = (hxylim_combined[f].z - hxylim_combined[f].y + 1) * hxspan_combined[f];
		span[f].x = (2 * hposn[f] + 1);
		hnpixels_full[f] =  span[f].x * span[f].x;
		BLKpx_full[f] = floor((THD.x - 1 + hnpixels_full[f]) / THD.x);
	}
	gpuErrchk(cudaMemcpy(dxspan, hxspan, sizeof(int)*nfplus, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dnpixels_bbox, hnpixels_bbox, sizeof(int)*nfplus, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dxspan_bistatic, hxspan_bistatic, sizeof(int)*nfplus, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dnpixels_bistatic, hnpixels_bistatic, sizeof(int)*nfplus, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dxspan_combined, hxspan_combined, sizeof(int)*nfplus, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dnpixels_combined, hnpixels_combined, sizeof(int)*nfplus, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(xylim_combined, hxylim_combined, sizeof(int4)*nfplus, cudaMemcpyHostToDevice));

	/* Calculate model lightcurve values at each user-specified epochs x[i],
	 * with i=1,2,...,ncalc; these may/may not be the same as epochs t[i]
	 * (i=1,2,...,n) at which actual lightcurve observations were made.  */
	cfs_set_pos_ae_krnl<<<BLKfrm,THD64>>>(ddat, pos, s, nframes,type, 0, 1);

	posclr_lc_krnl_af<<<BLKaf,THDaf>>>(pos, posn, dxspan_combined, xylim_combined, dnpixels_combined, blocks);
	checkErrorAfterKernelLaunch("posclr_lc_krnl64af");

	/* Call routine posvis to get  facet number, scattering & incidence
	 * angle, distance toward Earth at center of each POS pixel; set posbnd
	 * parameter = 1 if any model portion extends beyond POS frame limits*/
	posvis_gpu(dpar, dmod, pos, verts, orbit_off3, hposn, outbndarr,
			s, (nframes+1), 0, nf, 0, c, type, cf_stream, 0);

	/* Copy the posbnd flag returns for all frames to a host copy */
	gpuErrchk(cudaMemcpy(&houtbndarr, outbndarr, sizeof(int)*(nfplus),
			cudaMemcpyDeviceToHost));

	/* Now check the outbndarr for the posbnd flag for each frame */
	for (f=1; f<=nframes; f++) {
		if (houtbndarr[f]) {
			/* Call single-threaded kernel to set dpar->posbnd and dpar->posbnd_logfactor */
			cfs_set_posbnd_krnl<<<1,1,0,cf_stream[f-1]>>>(dpar, ddat, pos, s, f, type);
			houtbndarr[f]=0;
		}
	}

	/* Now view model from source (sun) and get facet number and distance
	 * toward source of each pixel in this projected view; use this
	 * information to determine which POS pixels are shadowed       */
	/* Because posvis_gpu processes all frames at the same time, if
	 * any of the frames are bistatic, all of them get calculated again  */
	posvis_gpu(dpar, dmod, pos, verts, orbit_off3, hposn,
			outbndarr, s, (nframes+1), 1, nf, 0, c, type, cf_stream, 1);

	/* Copy the posbnd flag returns for all frames to a host copy */
	gpuErrchk(cudaMemcpy(&houtbndarr, outbndarr, sizeof(int)*nfplus,
			cudaMemcpyDeviceToHost));

	/* Now check the outbndarr for the posbnd flag for each frame */
	for (f=1; f<=nframes; f++) {
		if (houtbndarr[f]) {
			/* Call single-threaded kernel to set dpar->posbnd and dpar->posbnd_logfactor */
			cfs_set_posbnd_krnl<<<1,1,0,cf_stream[f-1]>>>(dpar, ddat, pos, s, f, type);
		}
	}

	/* Initialize this stream for the posmask kernel to follow */
	posmask_init_krnl<<<BLKfrm,THD64>>>(pos, so, pxlpkm, nframes);

	for (f=1; f<=nframes; f++) {
			/* Now call posmask kernel for this stream, then loop
			 * to next stream and repeat 	 */
			posmask_krnl<<<BLKpx_full[f],THD,0,cf_stream[f-1]>>>(
					dpar, pos, so, pxlpkm, posn, hnpixels_full[f], span[f].x, f);
	} checkErrorAfterKernelLaunch("posmask_krnl64");

	/* Go through all visible and unshadowed POS pixels with low enough
	 * scattering and incidence angles, and mark facets which project onto
	 * their centers as having been "seen" at least once. Get xlim and ylim
	 * and exclude_seen flag */
	cfs_get_exclude_seen_krnl<<<BLKfrm,THD64>>>(dpar,pos,xylim,nframes, 1);
	checkErrorAfterKernelLaunch("cfs_get_exclude_seen_krnl");

	/* Now copy the flag and all frame pos's xlim and ylim values back from GPU */
	gpuErrchk(cudaMemcpyFromSymbol(&exclude_seen, cfs_exclude_seen, sizeof(int),
			0, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&hxylim, xylim, sizeof(int4)*nfplus, cudaMemcpyDeviceToHost));

	/* Calculate launch parameters for all frames */
	int xspan_max = 0, yspan_max = 0, maxthds = 0;
	int4 maxxylim;
	maxxylim.w = maxxylim.y = 1e3;
	maxxylim.x = maxxylim.z = -1e3;
	for (f=1; f<=nframes; f++) {
		span[f].x = hxylim[f].x - hxylim[f].w + 1;
		span[f].y = hxylim[f].z - hxylim[f].y + 1;
		hnpixels_bbox[f] = span[f].x * span[f].y;
		BLKpx_bbox[f].x = floor ((THD.x -1 + hnpixels_bbox[f]) / THD.x);
		maxxylim.w = min(maxxylim.w, hxylim[f].w);
		maxxylim.x = max(maxxylim.x, hxylim[f].x);
		maxxylim.y = min(maxxylim.y, hxylim[f].y);
		maxxylim.z = max(maxxylim.z, hxylim[f].z);

	}
	xspan_max = maxxylim.x - maxxylim.w + 1;
	yspan_max = maxxylim.z - maxxylim.y + 1;
	maxthds = xspan_max * yspan_max;

	/* It's not entirely clear to me what this does for the fit action. As of
	 * now (12/31/2017) it doesn't work because I have (temporarily?) removed
	 * the pos->f[][] pixel assignments from the posvis_gpu64 routine 	 */
	for (f=1; f<=nframes; f++) {
		/* Go through all POS pixels which are visible with low enough
		 * scattering angle and mark the facets which project onto
		 * their centers as having been "seen" at least once           */
		if (s != exclude_seen)
			cf_mark_pixels_seen_krnl<<<BLKpx_bbox[f],THD,0,cf_stream[f-1]>>>(
					dpar, dmod, pos, xylim, hnpixels_bbox[f], span[f].x, f);
	} checkErrorAfterKernelLaunch("cf_mark_pixels_krnl");

	/* Compute model brightness for this lightcurve point then copy to device  */
	apply_photo_gpu(dmod, ddat, pos, xylim, span, BLKpx_bbox, hnpixels_bbox,
			0, s, nframes, maxthds, maxxylim, cf_stream);

	/* Now that we have calculated the model lightcurve brightnesses y at each
	 * of the epochs x, we use cubic spline interpolation (Numerical Recipes
	 * routines spline and splint) to get model lightcurve brightness fit[i] at
	 * each OBSERVATION epoch t[i], with i=1,2,...,n. This will allow us (in
	 * routine chi2) to compare model to data (fit[i] to obs[i]) to get chi-
	 * square. Note that vector y2 contains the second derivatives of the
	 * interpolating function at the calculation epochs x. Smearing is handled
	 * by interpolating the brightness at the time t of each individual view
	 * and then taking the mean of all views that correspond to a given
	 * observed lightcurve point.                         */
	/* Configure and launch an ncalc-threaded kernel that performs what NR
	 * function spline does.  Original call:
	 *
	 * spline( lghtcrv->x, lghtcrv->y, ncalc, 2.0e30, 2.0e30, lghtcrv->y2);
	 */
	nThreads = nframes;
	THD.x = maxThreadsPerBlock;
	gpuErrchk(cudaMemset(u, 0, sizeof(double)*nfplus));
	BLKpx_full[0].x = floor((THD.x - 1 + nThreads)/THD.x);
	lghtcrv_spline_krnl<<<1,1>>>(ddat, s, 2.0e30, 2.0e30, u, nframes);
	checkErrorAfterKernelLaunch("lghtcrv_spline_krnl");

	/* Change launch parameters from ncalc threads to n threads */
	BLKpx_full[0].x = floor((THD.x - 1 + n) / THD.x);
	lghtcrv_splint_krnl<<<1,1>>>(ddat, s, n, nframes);
	checkErrorAfterKernelLaunch("lghtcrv_splint_krnl");

	free(hxspan);
	free(hnpixels_full);
	free(hnpixels_bbox);
	free(hnpixels_bistatic);
	free(hnpixels_combined);
	free(hxspan_bistatic);
	free(hxspan_combined);
	cudaFree(xylim2);
	cudaFree(xylim_combined);
	cudaFree(dxspan);
	cudaFree(dxspan_bistatic);
	cudaFree(dxspan_combined);
	cudaFree(dnpixels_bbox);
	cudaFree(dnpixels_bistatic);
	cudaFree(dnpixels_combined);
}
