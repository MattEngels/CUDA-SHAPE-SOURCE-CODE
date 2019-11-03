/*****************************************************************************************
                                                                             pos2deldop.c
Takes a pos_t structure and a deldop_t structure and "fills in" a deldopfrm_t structure
indexed by frm.  In other words, pos2deldop works from a model plane-of-sky image (with an
observer z-coordinate and a scattering angle assigned to each pixel) to produce a model
delay-Doppler image corresponding to data frame frm.
In the case of an orbiting binary system (the "orbit" action), pos2deldop only computes
power contributions from the orbiting body denoted by the "body" parameter: the routine is
called twice, once for each body.
pos2deldop takes contributions only from the rectangular plane-of-sky region defined by
pos.xlim and pos.ylim -- the smallest rectangle which completely "contains" the model in
the plane of the sky.  No power is contributed by parts of the model which extend beyond
the POS window; this is why such models are heavily penalized (the objective function is
doubled -- see function "objective" in file bestfit.c).
idellim and idoplim are updated for frame frm to show the model delay-Doppler
(rectangular) region that contains nonzero power.
Modified 2015 June 3 by CM:
    Implement smearing for the "fit" and "write" actions by adding "v" (view) parameter
        and applying each run of pos2deldop to a single view rather than to an entire
        (smeared) frame
Modified 2014 February 10 by CM:
    Add "ilaw" argument to the radlaw routine to implement multiple radar scattering laws
Modified 2012 March 23 by CM:
    Implement Doppler scaling
Modified 2010 September 1 by CM:
    Add braces to an if-then-else statement to avoid compilation warning
Modified 2010 August 26 by CM:
    For the "map" action, change the "map_forward" parameter to "map_mode"
        and implement map_mode = 'facets'
    For the "map" action, implement the "map_verbose" parameter
Modified 2010 June 15 by CM:
    Pass the entire par_t parameter structure as an argument rather than
        just selected parameters
    Implement the map action
Modified 2009 November 15 by CM:
    Fix argument type in printf statement
Modified 2009 April 3 by CM:
    Add the "warn_badradar" parameter (see below)
    Rather than halting the program if the sinc^2 Doppler response function
        extends to too many Doppler columns, set a flag and compute a factor
        by which the objective function should be increased (actually the
        logarithm of this factor).  If the "warn_badradar" parameter is
        turned on, print an explicit warning.
    If the model is too wide in delay-Doppler space even for the overflow
        image, set a flag and compute a factor by which the objective
        function should be increased (actually the logarithm of this
        factor).  If the "warn_badradar" parameter is turned on, print an
        explicit warning.
    Make pos2deldop int rather than void in order to return the flag
        described above to the calling procedure
Modified 2007 October 22 by CM:
    Use the vignetted DC column rather than the DC column for the unvignetted
        image
Modified 2007 August 4 by CM:
    Add orbit_xoff, orbit_yoff, and orbit_dopoff parameters, the x offset
        (POS image rows), y offset (POS image columns), and Doppler offset
        (delay-Doppler image columns) of the center of mass due to orbital
        motion.  Note that the delay offset due to orbital motion is already
        taken care of by the posvis routine.
    Add body parameter to indicate (for the "orbit" action) which of the two
        orbiting bodies' power contributions should be computed
    Add c (component) argument to radlaw routine
Modified 2006 September 14 by CM:
    If the overflow region is too small, print a warning rather than
        halting the program
Modified 2006 June 21 by CM:
    Change delres to del_per_pixel and dopres to dop_per_pixel
    For POS renderings, change res to km_per_pixel
Modified 2006 June 18 by CM:
    Allow each delay-Doppler frame in a dataset to have different dimensions
        after vignetting
Modified 2006 March 10 by CM:
    The bug fix of 2005 July 25 inadvertently caused "codefactor" to cancel
        out, so that the sensitivity falloff of short-code images away from
        DC was not properly modeled; fix this so that the sum of each
        POS pixel's contributions to the image is equal to codefactor*sigma
        rather than to sigma, where sigma is the cross-section equivalent
        of the echo power on the sky within that POS pixel.
    Pass the "speckle" parameter so that self-noise can be included when
        computing the chi squared contribution of the overflow region
    Compute overflow_xsec, overflow_delmean and overflow_dopmean so that
        these quantities can be used by the "delcorinit" action
Modified 2005 July 25 by CM:
    Fix bug in overall cross-section scale factor: return to Scott's scheme
        of normalizing the cross-section contributions from a given POS
        pixel so that they sum to the cross section actually present on the
        sky in that pixel
Modified 2005 July 20 by CM:
    Fix bug in computing floating-point delay-Doppler limits in usec and Hz
    Add "facet" argument to radlaw routine
Modified 2005 July 5 by CM:
    Eliminate "dir" argument (since we always add power to the model image
       and never subtract it)
    Add "set" (set number) argument in order to improve error messages
Modified 2005 June 27 by CM:
    Rename INFINITY constant to HUGENUMBER to avoid conflicts
Modified 2005 June 25 by CM:
    Rename old "dellim" to "idellim" and old "doplim" to "idoplim";
        these are delay-Doppler limits in (integer) row and column numbers,
        respectively
    Add new "dellim" and "doplim" which are floating-point delay-Doppler
        limits in usec and Hz, respectively, obtained PRIOR to convolution
        with the delay and Doppler response functions
Modified 2005 January 25 by CM:
    Take care of uninitialized variables to avoid compilation warnings
Modified 2003 May 11 by CM:
    Compute contributions to chi squared by model power which lies
        outside the limits of the data frame.
Modified 2003 May 5 by CM:
    For each POS pixel, compute the entire pixel's contribution to a
        given Doppler column in the model image so long as even one point
        at which we evaluate the sinc^2 response function is less than
        sinc2width/2.0 bins away from that Doppler column.  In other words,
        err on the side of computing too many small contributions to each
        column in the model image, so as not to omit significant
        contributions just because a POS pixel's *center* isn't close
        enough in Doppler.
Modified 2003 April 29 by CM:
    Evaluate the sinc^2 Doppler response function at nsinc2 points
        per POS pixel dimension, not just at the pixel center.
        The sinc^2 function varies rapidly -- one null per Doppler bin
        away from the central peak -- so if the pixel width is more than
        about half the Doppler resolution, we want to take the mean of
        several points within the pixel.
Modified 2003 April 26 by CM:
    Zero out the sinc^2 Doppler response function beyond the
        nearest sinc2width bins rather than beyond the nearest 2 bins
Modified 2003 April 24 by CM:
    Move "delcom" (COM delay bin) from delay-Doppler datasets to
        individual frames
Modified 2003 April 14 by CM:
    Now handles multiple samples per baud.
    Now handles short code vs. long code (original reduction method)
                           vs. long code (modified reduction method)
    Now correctly scales the model delay-Doppler image to account
        for delay mismatching and correlations between image rows
        (i.e., does NOT require the cross-section contributions from
        each plane-of-sky pixel to sum to the cross section actually
        present in that POS pixel)
 *****************************************************************************************/
extern "C" {
#include "../shape/head.h"
}

__device__ int /*p2ds_any_overflow, */p2ds_codemethod, p2ds_spb,
p2ds_stride, p2ds_spb_sq, p2ds_dopfftlen, p2ds_spb_over_stride, p2ds_nsinc2_sq;
__device__ double p2ds_const1, p2ds_const2, p2ds_one_over_spb, p2ds_delfact,
p2ds_dopfact;

__device__ static float atomicMin64(double* address, double val)
{
	unsigned long long* address_as_i = (unsigned long long*) address;
	unsigned long long old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
				__double_as_longlong(::fminf(val, __longlong_as_double(assumed))));
	} while (assumed != old);
	return __longlong_as_double(old);
}
__device__ static float atomicMax64(double* address, double val)
{
	unsigned long long* address_as_i = (unsigned long long*) address;
	unsigned long long old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
				__double_as_longlong(::fmaxf(val, __longlong_as_double(assumed))));
	} while (assumed != old);
	return __longlong_as_double(old);
}

__global__ void pos2deldop_init_krnl(
		struct dat_t *ddat,
		struct deldopfrm_t **frame,
		int *idel0,
		int *idop0,
		int *ndel,
		int *ndop,
		int set,
		int size,
		int *badradararr,
		int *any_overflow) {
	/* nfrm_alloc-threaded kernel */
	int f = blockIdx.x * blockDim.x + threadIdx.x;

	if (f < size) {
		/*  Initialize variables to avoid compilation warnings  */
		idel0[f] = idop0[f] = /*p2ds_any_overflow = */0;
		frame[f] = &ddat->set[set].desc.deldop.frame[f];
		frame[f]->idellim[0] = ndel[f] + 999999;
		frame[f]->idellim[1] = frame[f]->idoplim[1] = -999999;
		frame[f]->idoplim[0] = ndop[f] + 999999;
		frame[f]->dellim[0] =  HUGENUMBER;
		frame[f]->dellim[1] = -HUGENUMBER;
		frame[f]->doplim[0] =  HUGENUMBER;
		frame[f]->doplim[1] = -HUGENUMBER;

		badradararr[f] = 0;
		any_overflow[f] = 0;
		frame[f]->badradar_logfactor = 0.0;
	}
}
__global__ void pos2deldop_data_sampling_krnl(
		struct par_t *dpar,
		struct dat_t *ddat,
		struct deldopfrm_t **frame,
		struct pos_t **pos,
		double2 *axay,
		double2 *xyincr,
		double2 *deldopshift,
		double3 *w,
		double4 *dop,
		double4 *deldoplim,
		int4 *xylim,
		int set,
		int size,
		int v,
		double orbit_dopoff,
		int *badradararr) {
	/* nfrm_alloc-threaded kernel */

	/* Get parameters related to data sampling and data reduction; then
	 * compute two more (both in units of delay bins = image rows):
	 *  	const1: half of the base width of the delay response function
	 *  	const2: half the delay difference between the first and last
	 * 				image rows within each baud  */
	int f = blockIdx.x * blockDim.x + threadIdx.x;

	if (f == 0) {
		p2ds_codemethod = ddat->set[set].desc.deldop.codemethod;
		p2ds_spb = ddat->set[set].desc.deldop.spb;
		p2ds_stride = ddat->set[set].desc.deldop.stride;
		p2ds_dopfftlen = ddat->set[set].desc.deldop.dopfftlen;
		p2ds_spb_over_stride = p2ds_spb/p2ds_stride;
		p2ds_one_over_spb = 1.0/p2ds_spb;
		p2ds_spb_sq = p2ds_spb*p2ds_spb;
		if (p2ds_codemethod != LONG_ORIG) {
			p2ds_const1 = (3*p2ds_spb - 1)/(2.0*p2ds_stride);
			p2ds_const2 = (p2ds_spb - 1)/(2.0*p2ds_stride);
		} else {
			p2ds_const1 = (double) p2ds_spb_over_stride;
			p2ds_const2 = 0.0;  /* not used with this code + reduction method */
		}
		/*  Converts from km towards radar to delay bins  */
		p2ds_delfact = -KM2US/ddat->set[set].desc.deldop.del_per_pixel;

		p2ds_dopfact = ddat->set[set].desc.deldop.dopscale.val * KM2HZFACT * pos[0]->km_per_pixel
				* ddat->set[set].desc.deldop.Ftx / ddat->set[set].desc.deldop.dop_per_pixel;
		p2ds_nsinc2_sq = dpar->nsinc2 * dpar->nsinc2;
	}
	__syncthreads();

	if (f < size) {

		dop[f].y = frame[f]->dopDC_vig;
		/*  Get w, the apparent spin vector in observer coordinates  */
		dev_cotrans5(w, frame[f]->view[v].oe, frame[f]->view[v].spin, 1, f);

		/* Compute the Doppler bin increment per plane-of-sky pixel westward
		 * (ax) and northward (ay); these values are scaled by the "dopscale"
		 * parameter for this dataset. Then compute km2Hz, the Doppler
		 * increment (Hz) per km perpendicular to the projected spin axis in
		 * the plane of the sky.     */
		axay[f].x = -w[f].y * p2ds_dopfact;
		axay[f].y =  w[f].x * p2ds_dopfact;
		frame[f]->view[v].km2Hz = sqrt(axay[f].x*axay[f].x+axay[f].y*axay[f].y) *
				ddat->set[set].desc.deldop.dop_per_pixel / pos[f]->km_per_pixel;

		/* Compute the absolute value of the difference between the maximum (or minimum)
		 * Doppler on any given POS pixel's edge and the Doppler at its center             */
		if (w[f].x != 0.0 || w[f].y != 0.0)
			dop[f].z = frame[f]->view[v].km2Hz * 0.5 * pos[f]->km_per_pixel
			* sqrt(w[f].x * w[f].x + w[f].y * w[f].y) /	MAX( fabs(w[f].x),
					fabs(w[f].y));
		else
			dop[f].z = 0.0;

		/* We may be evaluating the sinc^2 Doppler response function at more than one point
		 * per POS pixel.  xincr and yincr are the Doppler bin increments between adjacent
		 * evaluation points in the x and y directions.  dopdiff_bl is the Doppler bin difference
		 * between the bottom-leftmost (southeasternmost) evaluation point and the pixel center.
		 * dopdiff_max is the maximum positive Doppler bin difference between any evaluation point
		 * and the pixel center.
		 * 	dop.w - dopdiff_bl
		 * 	dop.x - dopdiff_max
		 *	dop.y - dopDC_vig
		 * 	dop.z - dop_extra		 */
		xyincr[f].x = axay[f].x / dpar->nsinc2;
		xyincr[f].y = axay[f].y / dpar->nsinc2;
		dop[f].w = -(dpar->nsinc2 - 1) * (xyincr[f].x + xyincr[f].y) / 2;
		dop[f].x = (dpar->nsinc2 - 1) * (fabs(xyincr[f].x) + fabs(xyincr[f].y)) / 2;

		if (2 * dop[f].x + dpar->sinc2width + 1 > MAXBINS) {
			badradararr[f] = 1;
			frame[f]->badradar_logfactor += log(
					(2 * dop[f].x + dpar->sinc2width + 1) / MAXBINS);
			if (dpar->warn_badradar) {
				printf(
						"\nWARNING in pos2deldop_cuda_af.c for set %2d frame %2d:\n",
						set, f);
				printf(
						"        sinc^2 function evaluated at %d Doppler bins, must be <= %d\n",
						(int) ceil(2 * dop[f].x + dpar->sinc2width + 1),
						MAXBINS);
			}
		}
		/* Get the COM delay and Doppler bins, corrected for ephemeris drift and adjusted for
		 * orbital motion; the delay adjustment for orbital motion is done implicitly (the
		 * posvis routine has already adjusted the "z" values for all POS pixels), whereas the
		 * Doppler adjustment for orbital motion must be done here explicitly.                  */
		deldopshift[f].x = frame[f]->delcom_vig + frame[f]->view[v].deloff;
		deldopshift[f].y = frame[f]->dopcom_vig + frame[f]->view[v].dopoff
				+ orbit_dopoff;

		/* Now get pos->xlim[0], pos->xlim[1], pos->ylim[0], pos->ylim[1] */
		xylim[f].w = pos[f]->xlim[0];
		xylim[f].x = pos[f]->xlim[1];
		xylim[f].y = pos[f]->ylim[0];
		xylim[f].z = pos[f]->ylim[1];

		/* Copy frame[frm]->dellim[2] and frame[frm]->doplim[2] to the device variables */
		deldoplim[f].w = frame[f]->dellim[0];
		deldoplim[f].x = frame[f]->dellim[1];
		deldoplim[f].y = frame[f]->doplim[0];
		deldoplim[f].z = frame[f]->doplim[1];
	}
}
__global__ void pos2deldop_clr_overflow_krnl(double **fit_overflow, int f) {

	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int total;
	int i, j;
	if (offset < 4e6) {
		i = total % 2000;
		j = total / 2000;
		fit_overflow[f][offset] = 0.0;
	}
}
__global__ void pos2deldop_pixel_krnl_moda(
		struct par_t *dpar,
		struct mod_t *dmod,
		struct dat_t *ddat,
		struct pos_t **pos,
		struct deldopfrm_t **frame,
		double4 *deldoplim,
		double4 *dop,
		double2 *deldopshift,
		double2 *axay,
		double2 *xyincr,
		int *idel0,
		int *idop0,
		int *ndel,
		int *ndop,
		int xspan,
		int nThreads,
		double orbit_xoff,
		double orbit_yoff,
		int set,
		int f,
		int *any_overflow,
		double **fit_overflow) {
	/* nThreads-threaded kernel */
	/* This kernel version uses shared memory for many variables that are used by every
	 * thread.  radlaw has been shortened to exclude body and facets.  The floating point
	 * and integer delay and doppler limits are calculated in shared memory first and then
	 * consolidated at the end of the kernel back into global memory.
	 */
	/* Loop through all POS pixels within the rectangular plane-of-sky region spanned by the
	 *  model; for each such pixel which isn't blank sky, compute the cross-section contributions
	 *  to pixels in the model delay-Doppler frame. Note that functions posclr and posvis flag
	 *  blank-sky pixels by assigning "cose" = cos(scattering angle) = 0.
	 *  Only compute contributions from POS pixels that project onto the right body, in case this
	 *  is the "orbit" action (for which this routine is called twice, once for each of the two
	 *  orbiting bodies). */

	int offset = blockIdx.x * blockDim.x + threadIdx.x;

	int x, y, i, j, k, idel, idel_min, idel_max, idop_min, idop_max,
	idop, m, m_min, m_max, in_bounds, idel1, idel2, idop1, idop2;
	double delPOS, dopPOS, codefactor, tmp, arg_sample, amp, arg_left,
	sinc2arg, sinc2_mean, arg_bl, sumweights, fit_contribution, cose;
	double del_contribution[MAXBINS], dop_contribution[MAXBINS];

	__shared__ int xlim0, ylim0, nsinc2, sinc2width, idellim[2], idoplim[2];
	__shared__ double km_per_pixel;
	__shared__ double xincr, yincr;
	__shared__ double4 deldoplim_sh;

	/* Load variables that are used by every thread into shared memory */
	if (threadIdx.x==0) {
		xincr = xyincr[f].x;
		yincr = xyincr[f].y;
		xlim0 = pos[f]->xlim[0];
		ylim0 = pos[f]->ylim[0];
		nsinc2 = dpar->nsinc2;
		sinc2width = dpar->sinc2width;
		km_per_pixel = pos[f]->km_per_pixel;

		deldoplim_sh.w = deldoplim_sh.y = idellim[0] = idoplim[0] =  999999;
		deldoplim_sh.x = deldoplim_sh.z = idellim[1] = idoplim[1] = -999999;
	}
	__syncthreads();

	if (offset < nThreads) {
		/* Calculate x and y position index in delxdop grid */
		x = offset % xspan + xlim0;
		y = offset / xspan + ylim0;

		/* Load current pixel's cose value into register memory */
		cose = pos[f]->cose[x][y];

		if (cose > 0.0 ) {

			/* Get the (floating-point) delay and Doppler bin of the POS pixel
			 * center: delPOS and dopPOS. Also get the minimum and maximum
			 * (integer) delay and Doppler bins to which this pixel contributes
			 * power: idel_min and idel_max, idop_min and idop_max. Each POS
			 * pixel contributes power to *all* Doppler columns, but here we're
			 * zeroing out the sinc^2 response function beyond the nearest
			 * sinc2width columns. Actually, if nsinc2 > 1, we'll distribute
			 * power to *at least* sinc2width Doppler bins: For pixels which
			 * span multiple bins we'll err on the side of computing more
			 * contributions rather          than fewer.       */
			delPOS = pos[f]->z[x][y] * p2ds_delfact + deldopshift[f].x;
			idel_min = (int) floor(delPOS - p2ds_const1) + 1;
			idel_max = (int) ceil(delPOS + p2ds_const1) - 1;
			dopPOS = axay[f].x*(x - orbit_xoff) + axay[f].y*
					(y - orbit_yoff) + deldopshift[f].y;
			idop_min = (int) floor(dopPOS - dop[f].x + 1 - sinc2width/2.0);
			idop_max = (int) floor(dopPOS + dop[f].x + sinc2width/2.0);

			/*  For the short code, sensitivity drops as we move away from DC. (This variation is slow,
			 *  so we can just evaluate the response at the center of the POS pixel.)
			 *  Note that the SINC2 macro multiplies its argument by pi.        */
			codefactor = (p2ds_codemethod == SHORT) ? SINC2( (dopPOS-dop[f].y)/p2ds_dopfftlen ) : 1.0;

			/*  Update rectangular delay-Doppler region (row/column numbers) with !0 power according to model  */
			atomicMin(&idellim[0], idel_min);
			atomicMax(&idellim[1], idel_max);
			atomicMin(&idoplim[0], idop_min);
			atomicMax(&idoplim[1], idop_max);

			/*  Update the model's floating-point delay-Doppler limits, as determined prior to convolution
			 *  with the delay and Doppler response functions. At this point in the code, dellim and doplim
			 *  are pairs of floating-point row and column numbers which apply to POS pixel centers; when
			 *  the loop over POS pixels is finished we will convert them to usec and Hz, and will widen
			 *  the Doppler limits to account for nonzero POS pixel width.     */
			atomicMin64(&deldoplim_sh.w, delPOS);
			atomicMax64(&deldoplim_sh.x, delPOS);
			atomicMin64(&deldoplim_sh.y, dopPOS);
			atomicMax64(&deldoplim_sh.z, dopPOS);

			/*  Check whether or not all delay-Doppler pixels which will receive power from this POS pixel
			 *  fall within the data frame; if not, initialize the "overflow" image if necessary.         */
			if ((idel_min>=1) && (idel_max<=ndel[f]) && (idop_min>=1) &&
					(idop_max<=ndop[f]))
				in_bounds = 1;
			else {
				in_bounds = 0;
				if (!any_overflow[f]) {
					atomicExch(&any_overflow[f], 1);
					/* Center the COM in the overflow image:
					 * pixel [idel][idop] in the fit frame corresponds to
					 * pixel [idel+idel0][idop+idop0] in the fit_overflow frame*/

					idel0[f] =MAXOVERFLOW/2-(int)floor(deldopshift[f].x+0.5);
					idop0[f] =MAXOVERFLOW/2-(int)floor(deldopshift[f].y+0.5);
				}
			}

			/* Loop thru all delay bins this POS pixel contributes power (cross
			 * section), and compute delay response function for each bin       */
			for (idel=idel_min; idel<=idel_max; idel++) {
				if (p2ds_codemethod != LONG_ORIG) {
					/* Get the delay response function for image row idel:
					 * sum the triangle-function contributions from each
					 * sample per baud, then divide the sum by spb and square.
					 * The triangle function for sample m  (0 <= m <= spb-1)
					 * has unit height and a half-width of spb/stride image
					 * rows,and is centered [-const2 + m/stride] rows later
					 * than the row center (idel).
					 * In the code block below, the arguments to macros TRI
					 * and TRI2 have been divided by half-width spb/stride,
					 * since those two macros are defined to give nonzero
					 * values for arguments between -1 and 1.  Each argument,
					 * then, is just (delPOS - [triangle-function center]) /
					 * half-width.
					 * Do the two most common cases (spb = 1 or 2) without
					 * loops in order to gain a bit of speed.  For the other
					 * cases, set m_min and m_max so as not to waste time
					 * computing contributions that are zero.             */
					switch (p2ds_spb) {
					case 1:
						del_contribution[idel-idel_min] = TRI2( delPOS - idel );
						break;
					case 2:
						arg_sample = (delPOS - (idel - p2ds_const2)) /
						p2ds_spb_over_stride;
						del_contribution[idel-idel_min] = TRI( arg_sample )
			                      						+ TRI( arg_sample - 0.5 );
						del_contribution[idel-idel_min] *= del_contribution[idel-idel_min]/4;
						break;
					default:
						del_contribution[idel-idel_min] = 0.0;
						m_min = MAX( (int) floor((delPOS - idel - p2ds_const2)
								* p2ds_stride) , 0 );
						m_max = MIN( (int) ceil((delPOS - idel + p2ds_const1)
								* p2ds_stride) , p2ds_spb ) - 1;
						arg_sample = (delPOS - (idel - p2ds_const2)) /
								p2ds_spb_over_stride - m_min*p2ds_one_over_spb;
						for (m=m_min; m<=m_max; m++) {
							del_contribution[idel-idel_min] += TRI( arg_sample );
							arg_sample -= p2ds_one_over_spb;
						}
						del_contribution[idel-idel_min] *=
								del_contribution[idel-idel_min]/p2ds_spb_sq;
						break;
					}
				} else {

					/*  Long code with original (Harmon) reduction method: data for
					 *  each sample per baud are reduced separately,as if datataking
					 *  were just one sample per baud; then the image rows for spb/
					 *  stride samples are interleaved.  */
					del_contribution[idel-idel_min] = TRI2( (delPOS - idel) /
							p2ds_spb_over_stride );
				}
			}

			/*  Next include the sinc^2 factor for Doppler mismatching: Take the
			 *  mean of nsinc2^2 points interior to the POS pixel. Do the two most
			 *  common cases (nsinc2 = 1 or 2) without loops in order to gain a bit
			 *  of speed. Note that the SINC2 macro multiplies its argument by pi*/
			for (idop=idop_min; idop<=idop_max; idop++) {
				switch (nsinc2) {
				case 1:
					sinc2_mean = SINC2( dopPOS - idop );
					break;
				case 2:
					arg_bl = dopPOS + dop[f].w - idop;   /* bl = bottom left */
					sinc2_mean = ( SINC2( arg_bl ) + SINC2( arg_bl+xincr) +
							SINC2( arg_bl+yincr) + SINC2( arg_bl+xincr+yincr) ) / 4;
					break;
				default:
					arg_left = dopPOS + dop[f].w - idop;
					sinc2_mean = 0.0;
					for (i=0; i<nsinc2; i++) {
						sinc2arg = arg_left;
						for (j=0; j<nsinc2; j++) {
							sinc2_mean += SINC2( sinc2arg );
							sinc2arg += xincr;
						}
						arg_left += yincr;
					}
					sinc2_mean /= p2ds_nsinc2_sq;
					break;
				}
				k = MIN( idop - idop_min, MAXBINS);
				dop_contribution[k] = sinc2_mean;
			}

			/*  Compute the sum of delay-Doppler weighting factors  */
			sumweights = 0.0;
			for (idel=idel_min; idel<=idel_max; idel++)
				for (idop=idop_min; idop<=idop_max; idop++) {
					k = MIN( idop - idop_min, MAXBINS);
					sumweights += del_contribution[idel-idel_min]*dop_contribution[k];
				}

			/* The radar cross section within this plane-of-sky pixel is
			 * [differential radar scattering law]*[POS pixel area in km^2]
			 * The differential radar scattering law (function radlaw
			 * = d[cross section]/d[area] ) includes a sec(theta) factor to
			 * account for the fact that the POS pixel area is projected area
			 * rather than physical area on the target surface.      */
			amp = dev_radlaw_mod( &dmod->photo, ddat->set[set].desc.deldop.iradlaw,
					cose) * km_per_pixel * km_per_pixel * codefactor / sumweights;

			/* Only add this POS pixel's power contributions to model delay-
			 * Doppler frame if NONE of those contributions fall outside
			 * the frame limits.                                   */
			if (in_bounds) {
				/*  Add the cross-section contributions to the model frame  */
				for (idel=idel_min; idel<=idel_max; idel++)
					for (idop=idop_min; idop<=idop_max; idop++) {
						k = MIN( idop - idop_min, MAXBINS);
						fit_contribution = amp * del_contribution[idel-idel_min]
						                                          * dop_contribution[k];
						atomicAdd(&frame[f]->fit[idel][idop], fit_contribution);
					}
			} else {
				/* Add the cross-section contributions to the "overflow" image */
				idel1 = MAX( idel_min, -idel0[f]);
				idel2 = MIN( idel_max, -idel0[f] + MAXOVERFLOW - 1);
				idop1 = MAX( idop_min, -idop0[f]);
				idop2 = MIN( idop_max, -idop0[f] + MAXOVERFLOW - 1);
				for (idel=idel1; idel<=idel2; idel++)
					for (idop=idop1; idop<=idop2; idop++) {
						k = MIN( idop - idop_min, MAXBINS);
						fit_contribution = amp * del_contribution[idel-idel_min]
						                                          * dop_contribution[k];
						atomicAdd(&fit_overflow[f][(idop+idop0[f])*MAXOVERFLOW + (idel+idel0[f])],fit_contribution);
					}
			}
		}
	}
	__syncthreads();

	/* Now consolidate the idellim/idoplim and dellim/doplim shared memory
	 * variables back into global memory	 */
	if (threadIdx.x==0)	{	/* So thread 0 for every block */
		/*  Update rectangular delay-Doppler region (row/column numbers) with !0 power according to model  */
		atomicMin(&frame[f]->idellim[0], idellim[0]);
		atomicMax(&frame[f]->idellim[1], idellim[1]);
		atomicMin(&frame[f]->idoplim[0], idoplim[0]);
		atomicMax(&frame[f]->idoplim[1], idoplim[1]);

		atomicMin64(&deldoplim[f].w, deldoplim_sh.w);
		atomicMax64(&deldoplim[f].x, deldoplim_sh.x);
		atomicMin64(&deldoplim[f].y, deldoplim_sh.y);
		atomicMax64(&deldoplim[f].z, deldoplim_sh.z);
	}
}
__global__ void pos2deldop_deldoplim_krnl(
		struct dat_t *ddat,
		struct deldopfrm_t **frame,
		double4 *deldoplim,
		double4 *dop,
		double2 *deldopshift,
		int set,
		int size) {

	/* nfrm_malloc-threaded kernel */
	int f = blockIdx.x * blockDim.x + threadIdx.x;
	double dlppxl, dpppxl;
	dlppxl = ddat->set[set].desc.deldop.del_per_pixel;
	dpppxl = ddat->set[set].desc.deldop.dop_per_pixel;

	if (f<size) {
		frame[f]->dellim[0] = deldoplim[f].w;
		frame[f]->dellim[1] = deldoplim[f].x;
		frame[f]->doplim[0] = deldoplim[f].y;
		frame[f]->doplim[1] = deldoplim[f].z;

		/*  Convert the model's floating-point delay-Doppler limits from floating-
		 *  point row and column numbers to usec and Hz, and widen the Doppler limits
		 *  to account for nonzero POS pixel width  */
		frame[f]->dellim[0] = (frame[f]->dellim[0] - deldopshift[f].x)*dlppxl;
		frame[f]->dellim[1] = (frame[f]->dellim[1] - deldopshift[f].x)*dlppxl;
		frame[f]->doplim[0] = (frame[f]->doplim[0] - deldopshift[f].y)*dpppxl
				- dop[f].z;
		frame[f]->doplim[1] = (frame[f]->doplim[1] - deldopshift[f].y)*dpppxl
				+ dop[f].z;

	}
}
__global__ void pos2deldop_overflow_krnl(
		struct par_t *dpar,
		struct deldopfrm_t **frame,
		int *idel0,
		int *idop0,
		int *ndel,
		int *ndop,
		int set,
		int size,
		int *badradararr,
		int *any_overflow,
		double **fit_overflow) {
	/* nfrm_alloc (size) -threaded kernel */
	int f = blockIdx.x * blockDim.x + threadIdx.x;
	int i, i1, i2, j, j1, j2, idell0, idopp0;
	double lookfact, sdev_sq, variance,dopfactor, delfactor, fo;
	double o2 = 0.0, m2 = 0.0, xsec = 0.0, delmean = 0.0, dopmean = 0.0;

	if (f < size) {

		/* Calculate the overflow contributions to chi squared:
		 *  o2 = obs^2 contribution, m2 = model^2 contribution.
		 *
		 *  Also compute the summed cross section and the mean delay and Doppler
		 *  bins for the overflow region, for use with the "delcorinit" action    */
		frame[f]->overflow_o2 = 0.0;
		frame[f]->overflow_m2 = 0.0;
		frame[f]->overflow_xsec = 0.0;
		frame[f]->overflow_delmean = 0.0;
		frame[f]->overflow_dopmean = 0.0;

		sdev_sq = frame[f]->sdev*frame[f]->sdev;
		variance = sdev_sq;
		lookfact = (frame[f]->nlooks > 0.0) ? 1.0/frame[f]->nlooks : 0.0;

		if (any_overflow[f]) {

			i1 = MAX( frame[f]->idellim[0] + idel0[f], 0);
			i2 = MIN( frame[f]->idellim[1] + idel0[f], MAXOVERFLOW - 1);
			j1 = MAX( frame[f]->idoplim[0] + idop0[f], 0);
			j2 = MIN( frame[f]->idoplim[1] + idop0[f], MAXOVERFLOW - 1);

			for (i=i1; i<=i2; i++)
				for (j=j1; j<=j2; j++) {
					if (fit_overflow[f][j*MAXOVERFLOW + i] !=0.0) {
						if (dpar->speckle)
							variance = sdev_sq + lookfact *
							fit_overflow[f][j*MAXOVERFLOW + i] * fit_overflow[f][j*MAXOVERFLOW + i];
						frame[f]->overflow_o2 += 1.0;
						frame[f]->overflow_m2 += fit_overflow[f][j*MAXOVERFLOW + i] *
								fit_overflow[f][j*MAXOVERFLOW + i]/variance;
						frame[f]->overflow_xsec += fit_overflow[f][j*MAXOVERFLOW + i];
						frame[f]->overflow_delmean += (i-idel0[f]) *
								fit_overflow[f][j*MAXOVERFLOW + i];
						frame[f]->overflow_dopmean += (j-idop0[f]) *
								fit_overflow[f][j*MAXOVERFLOW + i];
					}
				}
			if (frame[f]->overflow_xsec != 0.0) {
				frame[f]->overflow_delmean /= frame[f]->overflow_xsec;
				frame[f]->overflow_dopmean /= frame[f]->overflow_xsec;
			}


			/*  Print a warning if the model extends even beyond the overflow image  */
			if (((frame[f]->idellim[0] + idel0[f]) < 0)            ||
					((frame[f]->idellim[1] + idel0[f]) >= MAXOVERFLOW) ||
					((frame[f]->idoplim[0] + idop0[f]) < 0)            ||
					((frame[f]->idoplim[1] + idop0[f]) >= MAXOVERFLOW)    ) {
				badradararr[f] = 1;
				delfactor = (MAX(frame[f]->idellim[1] + idel0[f], MAXOVERFLOW)
						- MIN(frame[f]->idellim[0] + idel0[f], 0)         )
					                						  / (1.0*MAXOVERFLOW);
				dopfactor = (MAX(frame[f]->idoplim[1] + idop0[f], MAXOVERFLOW)
						- MIN(frame[f]->idoplim[0] + idop0[f], 0)         )
					                						  / (1.0*MAXOVERFLOW);
				frame[f]->badradar_logfactor += log(delfactor*dopfactor);
				if (dpar->warn_badradar) {
					printf("\nWARNING in pos2deldop_cuda_streams.c for set %2d frame %2d:\n", set, f);
					printf("        model delay-Doppler image extends too far beyond the data image\n");
					printf("             data:  rows %2d to %2d , cols %2d to %2d\n", 1, ndel[f], 1, ndop[f]);
					printf("            model:  rows %2d to %2d , cols %2d to %2d\n",
							frame[f]->idellim[0], frame[f]->idellim[1],
							frame[f]->idoplim[0], frame[f]->idoplim[1]);
				}
			}
		}

	}
}

__host__ void pos2deldop_gpu(
		struct par_t *dpar,
		struct mod_t *dmod,
		struct dat_t *ddat,
		struct pos_t **pos,
		struct deldopfrm_t **frame,
		int4 *xylim,
		int *ndel,
		int *ndop,
		double orbit_xoff,
		double orbit_yoff,
		double orbit_dopoff,
		int body,
		int set,
		int nfrm_alloc,
		int v,
		int *badradararr,
		cudaStream_t *p2d_stream,
		double **fit_overflow)
{
	int xspan[nfrm_alloc], yspan, nThreads[nfrm_alloc], *idop0, *idel0,
	*any_overflow;
	double2 *axay, *xyincr, *deldopshift;
	double3 *w;
	double4 *dop, *deldoplim;
	int4 host_xylim[nfrm_alloc];
	dim3 BLK[nfrm_alloc], THD, BLKfrm, THD64;
	THD.x = maxThreadsPerBlock;	THD64.x = 64;
	BLKfrm.x = floor((THD64.x - 1 + nfrm_alloc)/THD64.x);

	gpuErrchk(cudaMalloc((void**)&idop0, sizeof(int)*nfrm_alloc));
	gpuErrchk(cudaMalloc((void**)&idel0, sizeof(int)*nfrm_alloc));
	gpuErrchk(cudaMalloc((void**)&any_overflow, sizeof(int)*nfrm_alloc));
	gpuErrchk(cudaMalloc((void**)&w, sizeof(double3)*nfrm_alloc));
	gpuErrchk(cudaMalloc((void**)&axay, sizeof(double2)*nfrm_alloc));
	gpuErrchk(cudaMalloc((void**)&xyincr, sizeof(double2)*nfrm_alloc));
	gpuErrchk(cudaMalloc((void**)&deldopshift, sizeof(double2)*nfrm_alloc));
	gpuErrchk(cudaMalloc((void**)&deldoplim, sizeof(double4)*nfrm_alloc));
	gpuErrchk(cudaMalloc((void**)&dop, sizeof(double4)*nfrm_alloc));

	/* Launch single-thread initialization kernel */
	pos2deldop_init_krnl<<<BLKfrm,THD64>>>(ddat, frame, idel0, idop0, ndel,
			ndop, set, nfrm_alloc, badradararr, any_overflow);
	checkErrorAfterKernelLaunch("pos2deldop_init_krnl");
	/* Launch kernel to determine data sampling/radar parameters.  */
	pos2deldop_data_sampling_krnl<<<BLKfrm,THD64>>>(dpar, ddat, frame, pos,
			axay, xyincr, deldopshift, w, dop, deldoplim, xylim, set,
			nfrm_alloc, v, orbit_dopoff, badradararr);
	checkErrorAfterKernelLaunch("pos2deldop_data_sampling_krnl");
	gpuErrchk(cudaMemcpy(host_xylim, xylim, nfrm_alloc*sizeof(int4), cudaMemcpyDeviceToHost));

	/* Figure out the kernel launch parameters for every stream */
	for (int f=0; f<nfrm_alloc; f++) {
		xspan[f] = host_xylim[f].x - host_xylim[f].w + 1;
		yspan = host_xylim[f].z - host_xylim[f].y + 1;
		nThreads[f] = xspan[f]*yspan;
		BLK[f].x = floor((THD.x -1 + nThreads[f]) / THD.x);
	}

	dim3 OFBLK, OFTHD;
	OFTHD.x = 1024; OFTHD.y = 1; OFTHD.z = 1;
	OFBLK.x = floor(((THD.x - 1) + (2000*2000))/THD.x);
	/* Assign 1 stream to each frame's iteration each of the three kernels */
	for (int f=0; f<nfrm_alloc; f++) {
		pos2deldop_clr_overflow_krnl<<<OFBLK,OFTHD>>>(fit_overflow, f);

		pos2deldop_pixel_krnl_moda<<<BLK[f],THD,0,p2d_stream[f]>>>(dpar, dmod, ddat, pos,
				frame, deldoplim, dop, deldopshift, axay, xyincr, idel0, idop0,
				ndel, ndop, xspan[f], nThreads[f], orbit_xoff, orbit_yoff,
				set, f, any_overflow, fit_overflow);
	} checkErrorAfterKernelLaunch("pos2deldop_pixel_krnl_moda");

	/* Launch kernel to copy the deldop limits back to original doubles in
	 * the frame structures.	 */
	pos2deldop_deldoplim_krnl<<<BLKfrm,THD64>>>(ddat, frame,
			deldoplim, dop, deldopshift, set, nfrm_alloc);
	checkErrorAfterKernelLaunch("pos2deldop_deldoplim_krnl");

	/* Launch kernel to take care of any bin overflow */
	pos2deldop_overflow_krnl<<<BLKfrm,THD64>>>(dpar, frame, idel0, idop0, ndel,
			ndop, set, nfrm_alloc, badradararr, any_overflow, fit_overflow);
	checkErrorAfterKernelLaunch("pos2deldop_overflow_krnl");

	cudaFree(idop0);
	cudaFree(idel0);
	cudaFree(w);
	cudaFree(axay);
	cudaFree(xyincr);
	cudaFree(deldopshift);
	cudaFree(deldoplim);
	cudaFree(dop);
}
