/*****************************************************************************************
                                                                            pos2doppler.c
Takes a pos_t structure and a doppler_t structure and "fills in" a dopfrm_t structure
indexed by frm.  In other words, pos2doppler works from a model plane-of-sky image (with
an observer z-coordinate and a scattering angle assigned to each pixel) to produce a model
Doppler spectrum corresponding to data frame frm.
In the case of an orbiting binary system (the "orbit" action), pos2doppler only computes
power contributions from the orbiting body denoted by the "body" parameter: the routine is
called twice, once for each body.
pos2doppler takes contributions only from the rectangular plane-of-sky region defined by
pos.xlim and pos.ylim -- the smallest rectangle which completely "contains" the model in
the plane of the sky.  No power is contributed by parts of the model which extend beyond
the POS window; this is why such models are heavily penalized (the objective function is
doubled -- see function f in file bestfit.c).
idoplim is updated for frame frm to show the model Doppler region that contains nonzero
power.
Modified 2016 November 14 by ME:
	Implemented an all-GPU version of pos2doppler.
Modified 2015 June 3 by CM:
    Implement smearing for the "fit" and "write" actions by adding "v" (view) parameter
        and applying each run of pos2doppler to a single view rather than to an entire
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
    Fix argument type in a printf statement
Modified 2009 April 3 by CM:
    Add the "warn_badradar" parameter (see below)
    If the sinc^2 Doppler response function extends to too many Doppler
        bins, set a flag and compute a factor by which the objective
        function should be increased (actually the logarithm of this
        factor).  If the "warn_badradar" parameter is turned on, print an
        explicit warning.
    If the model is too wide in Doppler space even for the overflow image,
        set a flag and compute a factor by which the objective function
        should be increased (actually the logarithm of this factor).  If
        the "warn_badradar" parameter is turned on, print an explicit
        warning.
    Make pos2doppler int rather than void in order to return the flag
        described above to the calling procedure
Modified 2007 August 4 by CM:
    Add orbit_xoff, orbit_yoff, and orbit_dopoff parameters, the x offset
        (POS image rows), y offset (POS image columns), and Doppler offset
        (spectral bins) of the center of mass due to orbital motion.
    Add body parameter to indicate (for the "orbit" action) which of the two
        orbiting bodies' power contributions should be computed
    Add c (component) argument to radlaw routine
Modified 2006 September 14 by CM:
    If the overflow region is too small, print a warning rather than
        halting the program
Modified 2006 June 21 by CM:
    Change dopres to dop_per_bin
    For POS renderings, change res to km_per_pixel
Modified 2006 June 18 by CM:
    Allow each Doppler frame in a dataset to have different dimensions
        after vignetting
Modified 2006 March 10 by CM:
    Pass the "speckle" parameter so that self-noise can be included when
        computing the chi squared contribution of the overflow region
    Compute overflow_xsec and overflow_dopmean so that these quantities
        can be used by the "delcorinit" action
Modified 2005 July 25 by CM:
    Fix bug in overall cross-section scale factor: return to Scott's scheme
        of normalizing the cross-section contributions from a given POS 
        pixel so that they sum to the cross section actually present on the 
        sky in that pixel
Modified 2005 July 20 by CM:
    Fix bug in computing floating-point Doppler limits in Hz
    Add "facet" argument to radlaw routine
Modified 2005 July 5 by CM:
    Eliminate "dir" argument (since we always add power to the model image
       and never subtract it)
    Add "set" (set number) argument in order to improve error messages
Modified 2005 June 27 by CM:
    Rename INFINITY constant to HUGENUMBER to avoid conflicts
Modified 2005 June 25 by CM:
    Rename old "doplim" to "idoplim"; this is the Doppler limits in
        (integer) bin numbers
    Add new "doplim" which is the floating-point Doppler limits in Hz,
        obtained PRIOR to convolution with the Doppler response function
Modified 2005 January 25 by CM:
    Take care of uninitialized variable
Modified 2003 May 11 by CM:
    Compute contributions to chi squared by model power which lies
    outside the limits of the data frame.
Modified 2003 May 5 by CM:
    For each POS pixel, compute the entire pixel's contribution to a
    given Doppler bin in the model spectrum so long as even one point
    at which we evaluate the sinc^2 response function is less than
    sinc2width/2.0 bins away from that Doppler bin.  In other words,
    err on the side of computing too many small contributions to each
    bin in the model spectrum, so as not to omit significant contributions
    just because a POS pixel's *center* isn't close enough in Doppler.
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
Modified 2003 April 17 by CM:
    Now correctly scales the model Doppler spectrum to account for
    Doppler mismatching
 *****************************************************************************************/
extern "C" {
#include "../shape/head.h"
}

/* Declare __device__ vars and structs, which have file scope */
__device__ int pds_nsinc2_sq, pds_any_overflow, pds_in_bounds, pds_badradar_global;

/*	Note that both pos2deldop_cuda.cu and posvis_cuda.cu have the atomicMaxf
 * 	and atomicMinf device functions defined separately.  This is done due to
 * 	the way static device functions I handled I guess.  I tried putting them
 * 	into a separate file and a declaration in the shape-cuda.h header file,
 * 	but to no avail.  So here they are, duplicated in both files.			*/

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

__global__ void pos2doppler_init_krnl(
		struct dat_t *ddat,
		struct dopfrm_t **frame,
		struct pos_t **pos,
		int4 *xylim,
		double3 *w,
		double2 *doplim,
		int *ndop,
		int *idop0,
		int set,
		int v,
		int size,
		int *badradararr) {

	/* nfrm_alloc-threaded kernel */
	int f = blockIdx.x * blockDim.x + threadIdx.x;
	if (f < size) {
		/* Initialize variables  */
		idop0[f] = 0;
		if (f==0)	pds_any_overflow = 0;
		frame[f]->idoplim[0] = ndop[f] + 999999;
		frame[f]->idoplim[1] = -999999;
		frame[f]->doplim[0] =  HUGENUMBER;
		frame[f]->doplim[1] = -HUGENUMBER;
		badradararr[f] = 0;
		frame[f]->badradar_logfactor = 0.0;

		/*  Get w, the apparent spin vector in observer coordinates  */
		dev_cotrans5(w, frame[f]->view[v].oe, frame[f]->view[v].spin, 1, f);

		/* Copy frame->doplim over to the float device variable */
		doplim[f].x = frame[f]->doplim[0];
		doplim[f].y = frame[f]->doplim[1];

		xylim[f].w = pos[f]->xlim[0];
		xylim[f].x = pos[f]->xlim[1];
		xylim[f].y = pos[f]->ylim[0];
		xylim[f].z = pos[f]->ylim[1];

	}
}
__global__ void pos2doppler_radar_parameters_krnl(
		struct par_t *dpar,
		struct dat_t *ddat,
		struct dopfrm_t **frame,
		struct pos_t **pos,
		double4 *dop,
		double3 *w,
		double2 *axay,
		double2 *xyincr,
		double *dopshift,
		double orbit_dopoff,
		int set,
		int v,
		int size,
		int *badradararr) {

	/* nfrm_alloc-threaded kernel */
	int f = blockIdx.x * blockDim.x + threadIdx.x;
	double dopfact;
	int dpb, nsinc2, sinc2width;
	double kmppxl;

	dpb = ddat->set[set].desc.doppler.dop_per_bin;
	nsinc2 = dpar->nsinc2;
	sinc2width = dpar->sinc2width;

	if (f < size) {
		kmppxl = pos[f]->km_per_pixel;

		if (f == 0) 	pds_nsinc2_sq = nsinc2 * nsinc2;
		/*  Compute the Doppler bin increment per plane-of-sky pixel westward (ax)
      and northward (ay); these values are scaled by the "dopscale" parameter
      for this dataset.  Then compute km2Hz, the Doppler increment (Hz) per
      km perpendicular to the projected spin axis in the plane of the sky.     */

		dopfact = ddat->set[set].desc.doppler.dopscale.val * KM2HZFACT *
				kmppxl	* ddat->set[set].desc.doppler.Ftx / dpb;

		axay[f].x = -w[f].y*dopfact;
		axay[f].y =  w[f].x*dopfact;
		frame[f]->view[v].km2Hz = sqrt(axay[f].x*axay[f].x +
				axay[f].y*axay[f].y) * dpb	/ kmppxl;

	/* Compute absolute value of the difference between maximum (or minimum)
	 * Doppler on any given POS pixel's edge and the Doppler at its center                               */
	/* dop.w - dopdiff_bl
	 * dop.x - dopdiff_max
	 * dop.y - dopDC_vig (not used in Doppler)
	 * dop.z - dop_extra		 */

	if (w[f].x != 0.0 || w[f].y != 0.0)
		dop[f].z = frame[f]->view[v].km2Hz * 0.5 * kmppxl *
			sqrt(w[f].x*w[f].x + w[f].y*w[f].y) /
			MAX( fabs(w[f].x), fabs(w[f].y));
	else
		dop[f].z = 0.0;

	/* We may be evaluating the sinc^2 Doppler response function at more than
	 * one point per POS pixel.
	 * 	xincr & yincr: Doppler bin increments between adjacent evaluation
	 * 			points in the x and y directions.
	 * 	dopdiff_bl: Doppler bin difference between bottom-leftmost (SE-most)
	 * 			evaluation point and the pixel center.
	 * 	dopdiff_max: max. positive Doppler bin difference between any
	 * 			evaluation point and the     pixel center.             */
	xyincr[f].x = axay[f].x / nsinc2;
	xyincr[f].y = axay[f].y / nsinc2;
	dop[f].w = -(nsinc2 - 1)*(xyincr[f].x + xyincr[f].y)/2;
	dop[f].x = (nsinc2 - 1)*(fabs(xyincr[f].x) + fabs(xyincr[f].x))/2;
	if (2*dop[f].x + sinc2width + 1 > MAXBINS) {
		badradararr[f] = 1;
		frame[f]->badradar_logfactor += log((2*dop[f].x + sinc2width + 1) / MAXBINS);
		if (dpar->warn_badradar) {
			printf("\nWARNING in pos2doppler.c for set %2d frame %2d:\n", set, f);
			printf("        sinc^2 function evaluated at %d Doppler bins, must be <= %d\n",
					(int) ceil(2*dop[f].x + sinc2width + 1), MAXBINS);
		}
	}
	/* Get the COM Doppler bin, corrected for ephemeris drift and adjusted for
	 * orbital motion                         */
	dopshift[f] = frame[f]->dopcom_vig + frame[f]->view[v].dopoff + orbit_dopoff;
	}
}
__global__ void pos2doppler_pixel_krnl(
		struct par_t *dpar,
		struct mod_t *dmod,
		struct dat_t *ddat,
		struct pos_t **pos,
		struct dopfrm_t **frame,
		double4 *dop,
		double2 *axay,
		double2 *doplim,
		double2 *xyincr,
		int *ndop,
		int *idop0,
		double *dopshift,
		int xspan,
		int set,
		int nframes,
		int frame_size,
		int body,
		double orbit_xoff,
		double orbit_yoff,
		int f) {

	/* Multi-threaded kernel */
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	int x = offset % xspan + pos[f]->xlim[0];
	int y = offset / xspan + pos[f]->ylim[0];
	int n, nsinc2, sinc2width ;

	int idop, idop_min, idop_max, idop1, idop2, i, j, c, fac, k;
	double tmp, amp, arg_left, sinc2arg, sinc2_mean, arg_bl, fit_contribution,
	sumweights, dop_contribution[MAXBINS], dopPOS;
	n = pos[f]->n;
	nsinc2 = dpar->nsinc2;
	sinc2width = dpar->sinc2width;

	if (offset < frame_size) {
		/* Loop through all POS pixels within the rectangular plane-of-sky
		 * region spanned by the model; for each such pixel which isn't blank
		 * sky, compute the cross-section contributions to pixels in the model
		 * Doppler spectrum. Note that functions posclr and posvis flag
		 * blank-sky pixels by assigning "cose" = cos(scattering angle) = 0.
		 * Only compute contributions from POS pixels that project onto the
		 * right body, in case this is the "orbit" action (for which this
		 * routine is called twice, once for each of the 2 orbiting bodies).*/

		if (pos[f]->cose[x][y] > 0.0 ) {// && pos[f]->body[x][y] == body) {

			/* Get the fp Doppler bin of POS pixel center: dopPOS. Also get the
			 * min and max int Doppler bins to which this pixel contributes
			 * power: idop_min and idop_max. Each POS pixel contributes power
			 * to *all* Doppler bins, but here we're zeroing out the sinc^2
			 * response function beyond the nearest sinc2width bins.
			 * Actually, if nsinc2 > 1, we'll distribute power to *at least*
			 * sinc2width Doppler bins: For pixels which span multiple bins
			 * we'll err on the side of computing more contributions rather
			 * than fewer. */
			/* dop.w - dopdiff_bl
			 * dop.x - dopdiff_max
			 * dop.y - dopDC_vig
			 * dop.z - dop_extra		 */
			dopPOS = axay[f].x*(x - orbit_xoff) + axay[f].y*(y - orbit_yoff) + dopshift[f];
			idop_min = (int) floor(dopPOS - dop[f].x + 1 - sinc2width/2.0);
			idop_max = (int) floor(dopPOS + dop[f].x + sinc2width/2.0);

			/* Update the rectangular delay-Doppler region with nonzero power
			 * according to the model              */
			atomicMin(&frame[f]->idoplim[0], idop_min);
			atomicMax(&frame[f]->idoplim[1], idop_max);

			/* Update model's fp Doppler limits, as determined prior to
			 * convolution with the Doppler response function. At this point
			 * in the code, doplim is a pair of floating-point bin numbers
			 * which applies to POS pixel centers; when the loop over POS
			 * pixels is finished we will convert these limits to Hz and will
			 * widen the limits to account for nonzero POS pixel width. */
			/* Note that p2d_doplim[2] is a single-precision (float) copy of
			 * the original p2d_frame->doplim[2] (double-precision). This is
			 * necessary to get atomic operations to work.		 */
			atomicMin64(&doplim[f].x, dopPOS);
			atomicMax64(&doplim[f].y, dopPOS);

			/* Check if all Doppler bins which will receive power from this POS
			 * pixel fall within the data frame; if not, initialize the
			 * "overflow" spectrum if necessary.  */
			if ( (idop_min >= 1) && (idop_max <= ndop[f]) )
				pds_in_bounds = 1;
			else {
				pds_in_bounds = 0;
				if (!pds_any_overflow)
					pds_any_overflow = 1;
			}

			/* Compute the sinc^2 factors for Doppler mismatching: Take the
			 * mean of nsinc2^2 points interior to the POS pixel. Do the two
			 * most common cases (nsinc2 = 1 or 2) without loops to gain speed.
			 * Note the SINC2 macro multiplies its argument by pi. Then add the
			 * cross-section contributions to the model spectrum.  */

			for (idop=idop_min; idop<=idop_max; idop++) {
				switch (nsinc2) {
				case 1:
					sinc2_mean = SINC2(dopPOS - idop);
					break;
				case 2:
					arg_bl = dopPOS + dop[f].w - idop;   /* bl = bottom left */
					sinc2_mean = (SINC2(arg_bl) +
							SINC2(arg_bl+xyincr[f].x) +
							SINC2(arg_bl+xyincr[f].y) +
							SINC2(arg_bl+xyincr[f].x+xyincr[f].y)) / 4;
					break;
				default:
					arg_left = dopPOS + dop[f].w - idop;
					sinc2_mean = 0.0;
					for (i=0; i<nsinc2; i++) {
						sinc2arg = arg_left;
						for (j=0; j<nsinc2; j++) {
							sinc2_mean += SINC2(sinc2arg);
							sinc2arg += xyincr[f].x;
						}
						arg_left += xyincr[f].y;
					}
					sinc2_mean /= pds_nsinc2_sq;
					break;
				}
				k = MIN( idop - idop_min, MAXBINS);
				dop_contribution[k] = sinc2_mean;
			}

			/* Compute the sum of Doppler weighting factors  */
			sumweights = 0.0;
			for (idop=idop_min; idop<=idop_max; idop++) {
				k = MIN( idop - idop_min, MAXBINS);
				sumweights += dop_contribution[k];
			}

			/* The radar cross section within this plane-of-sky pixel is
			 * [differential radar scattering law]*[POS pixel area in km^2].
			 * The differential radar scattering law (function radlaw
			 * = d[cross section]/d[area] ) includes a sec(theta) factor to
			 * account for the fact that the POS pixel area is projected area
			 * rather than physical area on the target surface.	 */
			amp = dev_radlaw(&dmod->photo, ddat->set[set].desc.doppler.iradlaw,
					pos[f]->cose[x][y], pos[f]->comp[x][y], pos[f]->f[x][y])
		    		 * pos[f]->km_per_pixel * pos[f]->km_per_pixel / sumweights;

			/* Only add POS pixel's power contributions to model Doppler spect-
			 * rum if NONE of those contributions fall outside spectrum limits*/
			if (pds_in_bounds) {
				/* Add the cross-section contributions to the model frame  */
				for (idop=idop_min; idop<=idop_max; idop++) {
					k = MIN( idop - idop_min, MAXBINS);
					fit_contribution = amp * dop_contribution[k];

					atomicAdd(&frame[f]->fit[idop], fit_contribution);
				}
			}
		}  /* if cos(scattering angle) > 0 and POS pixel projects onto the right body */
	}
}

__global__ void pos2doppler_finish_krnl(
		struct par_t *dpar,
		struct dat_t *ddat,
		struct dopfrm_t **frame,
		double4 *dop,
		double2 *doplim,
		double *dopshift,
		int *idop0,
		int *ndop,
		int set,
		int size,
		int *badradararr) {

	/* nfrm_alloc-threaded kernel */
	int f = blockIdx.x * blockDim.x + threadIdx.x;
	int j, j1, j2;
	double lookfact, sdev_sq, variance, dopfactor;

	if (f < size) {
		/* Copy float device variable over to the frame->doplim   */
		frame[f]->doplim[0] = doplim[f].x;
		frame[f]->doplim[1] = doplim[f].y;

		/* Convert model's Doppler limits from float bin numbers to Hz and
		 * widen the limits to account for nonzero POS pixel width    */
		frame[f]->doplim[0] = (frame[f]->doplim[0] - dopshift[f])*
				ddat->set[set].desc.doppler.dop_per_bin - dop[f].z;
		frame[f]->doplim[1] = (frame[f]->doplim[1] - dopshift[f])*
				ddat->set[set].desc.doppler.dop_per_bin + dop[f].z;

		/* Calculate overflow contributions to chi squared:
		 *   o2 = obs^2 contribution, m2 = model^2 contribution.
		 * Also compute summed cross section and mean Doppler bin for overflow
		 * region, for use with the "delcorinit" action   */
		frame[f]->overflow_o2 = 0.0;
		frame[f]->overflow_m2 = 0.0;
		frame[f]->overflow_xsec = 0.0;
		frame[f]->overflow_dopmean = 0.0;
		sdev_sq = frame[f]->sdev*frame[f]->sdev;
		variance = sdev_sq;
		lookfact = (frame[f]->nlooks > 0.0) ? 1.0/frame[f]->nlooks : 0.0;
		if (pds_any_overflow) {

			badradararr[f] = 1;
			dopfactor = (MAX(frame[f]->idoplim[1] + idop0[f], MAXOVERFLOW)
					- MIN(frame[f]->idoplim[0]+idop0[f],0))/(1.0*MAXOVERFLOW);
			frame[f]->badradar_logfactor += log(dopfactor);
			if (dpar->warn_badradar) {
				printf("\nWARNING in pos2doppler.c for set %2d frame %2d:\n", set, f);
				printf("        model Doppler spectrum extends too far beyond the data spectrum\n");
				printf("             data:  bins %2d to %2d\n", 1, ndop[f]);
				printf("            model:  bins %2d to %2d\n",
						frame[f]->idoplim[0], frame[f]->idoplim[1]);
			}
		}
		if (badradararr[f]) pds_badradar_global = 0;
	}
}

__host__ int pos2doppler_gpu(
		struct par_t *dpar,
		struct mod_t *dmod,
		struct dat_t *ddat,
		struct pos_t **pos,
		struct dopfrm_t **frame,
		int4 *xylim,
		double orbit_xoff,
		double orbit_yoff,
		double orbit_dopoff,
		int *ndop,
		int body,
		int set,
		int nfrm_alloc,
		int v,
		int *badradararr,
		cudaStream_t *pds_stream)
{
	int badradar, xspan[nfrm_alloc], yspan, nThreads[nfrm_alloc], f, *idop0;
	dim3 BLK[nfrm_alloc], THD, BLKfrm, THD64;
	THD.x = maxThreadsPerBlock; THD64.x = 64;
	double *dopshift;
	double2 *doplim, *axay, *xyincr;
	double3 *w;
	double4 *dop;
	int4 host_xylim[nfrm_alloc];
	BLKfrm.x = floor((THD64.x - 1 + nfrm_alloc)/THD64.x);

	gpuErrchk(cudaMalloc((void**)&dopshift, sizeof(double)*nfrm_alloc));
	gpuErrchk(cudaMalloc((void**)&axay, 	sizeof(double2)*nfrm_alloc));
	gpuErrchk(cudaMalloc((void**)&xyincr,   sizeof(double2)*nfrm_alloc));
	gpuErrchk(cudaMalloc((void**)&doplim,   sizeof(double2)*nfrm_alloc));
	gpuErrchk(cudaMalloc((void**)&w, 	   	sizeof(double3)*nfrm_alloc));
	gpuErrchk(cudaMalloc((void**)&dop, 	   	sizeof(double4)*nfrm_alloc));
	gpuErrchk(cudaMalloc((void**)&idop0,	sizeof(int)*nfrm_alloc));

	/* Launch initialization kernels */
	pos2doppler_init_krnl<<<BLKfrm,THD64>>>(ddat, frame, pos, xylim, w, doplim,
			ndop, idop0, set, v, nfrm_alloc, badradararr);
	checkErrorAfterKernelLaunch("pos2doppler_init_krnl");

	pos2doppler_radar_parameters_krnl<<<BLKfrm,THD64>>>(dpar,
			ddat, frame, pos, dop, w, axay, xyincr, dopshift, orbit_dopoff,
			set, v, nfrm_alloc, badradararr);
	checkErrorAfterKernelLaunch("pos2doppler_radar_parameters_krnl");

	/* Figure out the kernel launch parameters for every stream */
	gpuErrchk(cudaMemcpy(host_xylim, xylim, nfrm_alloc*sizeof(int4), cudaMemcpyDeviceToHost));
	for (f=0; f<nfrm_alloc; f++) {
		xspan[f] = host_xylim[f].x - host_xylim[f].w + 1;
		yspan = host_xylim[f].z - host_xylim[f].y + 1;
		nThreads[f] = xspan[f]*yspan;
		BLK[f].x = floor((THD.x -1 + nThreads[f]) / THD.x);
	}

	for (f=0; f<nfrm_alloc; f++) {
		/* Loop through all pixels and calculate contributed power */
		pos2doppler_pixel_krnl<<<BLK[f],THD,0, pds_stream[f]>>>(dpar,dmod,ddat,
				pos, frame, dop, axay, doplim, xyincr, ndop, idop0, dopshift,
				xspan[f], set, nfrm_alloc, nThreads[f], body, orbit_xoff, orbit_yoff, f);
	}
	checkErrorAfterKernelLaunch("pos2doppler_pixel_krnl");

	/* Synchronize streams to default stream */
	for (f=0; f<nfrm_alloc; f++)
		cudaStreamSynchronize(pds_stream[f]);

	/* Finish up Doppler calculations */
	pos2doppler_finish_krnl<<<BLKfrm,THD64>>>(dpar,ddat,frame,
			dop, doplim, dopshift, idop0, ndop, set, nfrm_alloc, badradararr);
	checkErrorAfterKernelLaunch("pos2doppler_finish_krnl");
	gpuErrchk(cudaMemcpyFromSymbol(&badradar, pds_badradar_global, sizeof(int),
			0, cudaMemcpyDeviceToHost));

	cudaFree(dopshift);
	cudaFree(axay);
	cudaFree(xyincr);
	cudaFree(doplim);
	cudaFree(w);
	cudaFree(dop);
	cudaFree(idop0);
	return badradar;
}
