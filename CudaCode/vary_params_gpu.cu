/*****************************************************************************************
                                                                            vary_params.c
This routine is called by every processing node for every trial value of every floating
parameter during a fit, in order to implement the "vary_radalb" "vary_optalb"
"vary_delcor0" and "vary_dopscale" parameters.  The code, which is essentially lifted from
calc_fits.c, computes up to four means:
a) mean distance towards Earth of the subradar point relative to the COM,
   for delay-Doppler frames whose 0th-order delay correction polynomial coefficient is not
   held constant; this is used to adjust the 0th-order delay correction polynomial
   coefficient if the "vary_delcor0" parameter is turned on.
b) mean "radar" projected area for (delay-)Doppler frames that are treated as absolute
   photometry; this is used to adjust the radar albedo (R) if the "vary_radalb" parameter
   is turned on.
c) mean "optical" unshadowed projected area for calculated lightcurve points that are
   treated as absolute photometry; this is used to adjust the optical albedo (R or w) if
   the "vary_optalb" parameter is turned on.  Note that plane-of-sky datasets are not used
   here, since these frames are always treated as relative photometry.
d) mean cos(subradar latitude) for (delay-)Doppler frames in datasets whose Doppler
   scaling parameter is allowed to float; this is used to adjust those parameters if the
   "vary_dopscale" parameter is turned on.
When a branch node calls this routine, it returns its datasets' summed contributions (NOT
mean contributions) to the four output parameters, deldop_zmax, rad_xsec, opt_brightness,
and cos_subradarlat.
When the root node calls this routine, it first computes its datasets' summed
contributions to these four parameters; then it receives and adds in the contributions
from the branch nodes; and finally it returns the mean (NOT summed) parameters.
Before calling vary_params, the model's size/shape and spin states must be realized
(realize_mod and realize_spin); if albedos are being varied jointly with other parameters,
the photometric state must also be realized (realize_photo); and in either case the
0th-order delay correction polynomial coefficients and the Doppler scaling factors must be
reset to their saved values via the appropriate calls to realize_delcor and
realize_dopscale, respectively.
Modified 2017 March 27 by ME:
	Split off again from previous CUDA code to create a cudaStreams version.
	cudaStreams provide another level of parallelism by executing functions
	inside a stream in that specific order, but other streams are independent
	and can thus perform their own tasks at the same time.  The CUDA runtime
	driver will keep loading the GPU with parallel streamed tasks until
	capacity is reached or the tasks run out.
	A special note on the code structure in this version:  There are a lot of
	if blocks and for loops over frames that may seem weirdly placed or
	inefficient, i.e. multiple for-loops through frames right after one another
	with just one or two lines of code inside each loop.  This is done
	deliberately to launch streamed kernels in parallel.  Alteration to the
	code could break this, resulting in a loss of parallelism and therefore:
	speed.
Modified 2016 November 6 by ME:
	Split off from vary_params to create a version that performs almost exclusively
	on the GPU
Modified 2015 June 10 by CM:
    Implement smearing
Modified 2014 February 12 by CM:
    Add "ilaw" argument to the apply_photo routine
Modified 2012 March 23 by CM:
    Implement Doppler scaling
Modified 2011 September 10 by CM:
    Two small aesthetic changes in the lightcurve section of the code
Modified 2010 June 15 by CM:
    Revise arguments to pos2deldop and pos2doppler routines
Modified 2010 April 12 by CM:
    Include overflow region when computing cross sections
    Added comment about calling realize_delcor before calling vary_params
Modified 2009 March 29 by CM:
    For MPI_Recv calls, mpi_par[0] is no longer equal to the MPI action,
        since the message tag argument already serves that purpose (as of
        2008 April 10) -- so the other mpi_par elements are renumbered
    Add "warn_badradar" argument to pos2deldop and pos2doppler routines
Modified 2008 April 10 by CM:
    Use message tag argument to MPI_Recv to identify the MPI action
Modified 2007 August 18 by CM:
    Rename MPI_TAG to MPI_TAG_1 to avoid name conflict with mpich headers
Modified 2007 August 4 by CM:
    Add orbit_offset and body arguments to posvis routine and remove
        facet argument
    Add orbit_xoff, orbit_yoff, orbit_dopoff, and body arguments to
        pos2deldop and pos2doppler routines
    Add body argument to apply_photo routine
Written 2006 October 1 by CM
 *****************************************************************************************/

extern "C" {
#include "../shape/head.h"
}
__device__ double gpu_deldop_cross_section64=0.0, gpu_doppler_cross_section64=0.0,
		gpu_sum_rad_xsec64=0.0, gpu_sum_cos_subradarlat64=0.0, gpu_sum_deldop_zmax64,
		gpu_sum_opt_brightness64;
__device__ double gpu_deldop_zmax=0.0, gpu_rad_xsec=0.0, gpu_opt_brightness=0.0,
		gpu_cos_subradarlat=0.0;

__global__ void init_krnl(
		struct par_t *dpar,
		struct dat_t *ddat,
		int *compute_zmax,
		int *compute_cosdelta,
		int *compute_brightness,
		unsigned char *dtype,
		int nsets) {
	/* Single-threaded kernel, to be performed by GPU0 */
	int s;
	if (threadIdx.x == 0) {
		/* Initialize __device__ (file scope) variables to zero */
		gpu_deldop_cross_section64 = 0.0;
		gpu_doppler_cross_section64 = 0.0;
		gpu_sum_rad_xsec64 = 0.0;
		gpu_sum_cos_subradarlat64 = 0.0;
		gpu_sum_deldop_zmax64 = 0.0;
		gpu_sum_opt_brightness64  = 0.0;


		gpu_deldop_zmax = 0.0;
		gpu_rad_xsec = 0.0;
		gpu_opt_brightness = 0.0;
		gpu_cos_subradarlat = 0.0;

		for (s=0; s<nsets; s++) {
			switch(dtype[s]) {
			case DELAY:
				compute_cosdelta[s] = (dpar->vary_dopscale != VARY_NONE
						&& ddat->set[s].desc.deldop.dopscale.state != 'c');
				compute_zmax[s] = (dpar->vary_delcor0 != VARY_NONE
						&& ddat->set[s].desc.deldop.delcor.a[0].state != 'c');
				compute_brightness[s] = 0;
				break;
			case DOPPLER:
				compute_cosdelta[s] = (dpar->vary_dopscale != VARY_NONE &&
						ddat->set[s].desc.doppler.dopscale.state != 'c');
				compute_zmax[s] = 0;
				compute_brightness[s] = 0;
				break;
			case POS:
				break;
			case LGHTCRV:
				compute_cosdelta[s] = 0;
				compute_zmax[s] = 0;
				compute_brightness[s] = (dpar->vary_optalb != VARY_NONE
						&& ddat->set[s].desc.lghtcrv.cal.state == 'c');
			}
		}
	}
}
__global__ void zmax_final_krnl(double value) {
	/* Single-threaded kernel */
	if (threadIdx.x == 0)
		gpu_sum_deldop_zmax64 += value;
}
__global__ void xsec_doppler_krnl(struct dat_t *ddat, double frm_xsec,
		int s, int f) {
	/* Single-threaded kernel */
	if (threadIdx.x == 0) {
		gpu_doppler_cross_section64 = ddat->set[s].desc.doppler.frame[f].overflow_xsec;
		gpu_doppler_cross_section64 += frm_xsec;
		gpu_doppler_cross_section64 *= ddat->set[s].desc.doppler.frame[f].cal.val;
		gpu_sum_rad_xsec64 += gpu_doppler_cross_section64 *
				ddat->set[s].desc.doppler.frame[f].weight;
	}
}
__global__ void xsec_deldop_krnl(double value) {
	/* Single-threaded kernel */
	if (threadIdx.x == 0)
		gpu_sum_rad_xsec64 += value;
}
__global__ void cosdelta_krnl(struct dat_t *ddat, int s, int size) {

	/* nfrm_alloc-threaded kernel */
	int f = blockIdx.x * blockDim.x + threadIdx.x;

	if (f < size) {
		int j, view;
		double weight, cos_delta, oa[3][3], to_earth[3];

		switch(ddat->set[s].type){
		case DELAY:
			view = ddat->set[s].desc.deldop.v0;
			/* oa = matrix to transform body-fixed to observer coordinates  */
			/* to_earth = normalized target-to-Earth vector in body-fixed coords  */
			dev_mtrnsps(oa, ddat->set[s].desc.deldop.frame[f].view[view].ae);
			dev_mmmul(oa, ddat->set[s].desc.deldop.frame[f].view[view].oe, oa);
			for (j=0; j<=2; j++)
				to_earth[j] = oa[2][j];
			cos_delta = sqrt(to_earth[0]*to_earth[0] + to_earth[1]*to_earth[1]);
			weight = ddat->set[s].desc.deldop.frame[f].weight;
			gpu_sum_cos_subradarlat64 += cos_delta*weight;
			break;
		case DOPPLER:
			view = ddat->set[s].desc.doppler.v0;
			/* oa = matrix to transform body-fixed to observer coordinates  */
			/* to_earth = normalized target-to-Earth vector in body-fixed coords  */
			dev_mtrnsps(oa, ddat->set[s].desc.doppler.frame[f].view[view].ae);
			dev_mmmul(oa, ddat->set[s].desc.doppler.frame[f].view[view].oe, oa);
			for (j=0; j<=2; j++)
				to_earth[j] = oa[2][j];
			cos_delta = sqrt(to_earth[0]*to_earth[0] + to_earth[1]*to_earth[1]);
			weight = ddat->set[s].desc.doppler.frame[f].weight;
			gpu_sum_cos_subradarlat64 += cos_delta*weight;
		}
	}
}
__global__ void finalize_krnl(struct dat_t *ddat) {
	/* Single-threaded kernel */
	if (threadIdx.x == 0) {
		if (ddat->sum_deldop_zmax_weights > 0.0)
			gpu_deldop_zmax = gpu_sum_deldop_zmax64 / ddat->sum_deldop_zmax_weights;
		else
			gpu_deldop_zmax = 0.0;
		if (ddat->sum_rad_xsec_weights > 0.0) {
			gpu_rad_xsec = gpu_sum_rad_xsec64 / ddat->sum_rad_xsec_weights;			}
		else
			gpu_rad_xsec = 0.0;
		if (ddat->sum_opt_brightness_weights > 0.0)
			gpu_opt_brightness = gpu_sum_opt_brightness64 / ddat->sum_opt_brightness_weights;
		else
			gpu_opt_brightness = 0.0;
		if (ddat->sum_cos_subradarlat_weights > 0.0)
			gpu_cos_subradarlat = gpu_sum_cos_subradarlat64 / ddat->sum_cos_subradarlat_weights;
		else
			gpu_cos_subradarlat = 0.0;
	}
}
__global__ void delay_params_krnl(struct par_t *dpar, struct dat_t *ddat,
		struct pos_t **pos, struct deldopfrm_t **frame, int *compute_xsec,
		int *posn, int *ndel, int *ndop, int s, int size, int4 *xylim) {
	/* nframes-threaded kernel */
	int f = blockIdx.x * blockDim.x + threadIdx.x;

	if (f < size) {
		compute_xsec[f] = (dpar->vary_radalb != VARY_NONE
				&& ddat->set[s].desc.deldop.frame[f].cal.state == 'c');

		pos[f] = &ddat->set[s].desc.deldop.frame[f].pos;
		posn[f] = pos[f]->n;
		ndel[f] = ddat->set[s].desc.deldop.frame[f].ndel;
		ndop[f] = ddat->set[s].desc.deldop.frame[f].ndop;
		frame[f] = &ddat->set[s].desc.deldop.frame[f];
		xylim[f].w = pos[f]->xlim[0];
		xylim[f].x = pos[f]->xlim[1];
		xylim[f].y = pos[f]->ylim[0];
		xylim[f].z = pos[f]->ylim[1];
	}
}
__global__ void dop_params_krnl(struct par_t *dpar, struct dat_t *ddat,
		struct pos_t **pos, struct dopfrm_t **frame, int *compute_xsec,
		int *posn, int *ndop, int s, int size, int4 *xylim) {
	/* nframes-threaded kernel */
	int f = threadIdx.x;

	if (f < size) {
		compute_xsec[f] = (dpar->vary_radalb != VARY_NONE &&
				ddat->set[s].desc.doppler.frame[f].cal.state == 'c');
		pos[f] = &ddat->set[s].desc.doppler.frame[f].pos;
		posn[f] = pos[f]->n;
		ndop[f] = ddat->set[s].desc.doppler.frame[f].ndop;
		frame[f] = &ddat->set[s].desc.doppler.frame[f];
		xylim[f].w = pos[f]->xlim[0];
		xylim[f].x = pos[f]->xlim[1];
		xylim[f].y = pos[f]->ylim[0];
		xylim[f].z = pos[f]->ylim[1];
	}
}
__global__ void lghtcrv_params_krnl(struct par_t *dpar, struct dat_t *ddat,
		struct pos_t **pos, int *posn, int *bistatic, int s, int size,
		int4 *xylim) {
	/* nframes-threaded kernel */
	int f = blockIdx.x * blockDim.x + threadIdx.x + 1;

	if (f <= size) {
		pos[f] = &ddat->set[s].desc.lghtcrv.rend[f].pos;
		posn[f] = pos[f]->n;
		bistatic[f] = pos[f]->bistatic;
		xylim[f].w = pos[f]->xlim[0];
		xylim[f].x = pos[f]->xlim[1];
		xylim[f].y = pos[f]->ylim[0];
		xylim[f].z = pos[f]->ylim[1];
	}
}
__global__ void set_ae_oe_krnl(struct dat_t *ddat,
		struct pos_t **pos,
		unsigned char type,
		int s,
		int size) {

	/* nfrm_alloc-threaded kernel*/
	int indx, i, j, f = blockIdx.x * blockDim.x + threadIdx.x;

	if (f < size) {
		switch (type) {
		case DELAY:
			indx = ddat->set[s].desc.deldop.v0;
			for (i=0; i<3; i++) {
				for (j=0; j<3; j++) {
					pos[f]->ae[i][j] = ddat->set[s].desc.deldop.frame[f].view[indx].ae[i][j];
					pos[f]->oe[i][j] = ddat->set[s].desc.deldop.frame[f].view[indx].oe[i][j];
					if (i==1 && j==1)
						pos[f]->bistatic = 0;
				}
			}
			break;
		case DOPPLER:
			indx = ddat->set[s].desc.doppler.v0;
			for (i=0; i<3; i++) {
				for (j=0; j<3; j++) {
					pos[f]->ae[i][j] =	ddat->set[s].desc.doppler.frame[f].view[indx].ae[i][j];
					pos[f]->oe[i][j] =	ddat->set[s].desc.doppler.frame[f].view[indx].oe[i][j];
					if (i==1 && j==1)
						pos[f]->bistatic = 0;
				}
			}
			break;
		case LGHTCRV:
			f++; /* Lightcurve offset */
			for (i=0; i<3; i++) {
				for (j=0; j<3; j++) {
					pos[f]->ae[i][j] = ddat->set[s].desc.lghtcrv.rend[f].ae[i][j];
					pos[f]->oe[i][j] = ddat->set[s].desc.lghtcrv.rend[f].oe[i][j];
					pos[f]->se[i][j] = ddat->set[s].desc.lghtcrv.rend[f].se[i][j];
					if (i==1 && j==1)
						pos[f]->bistatic = 1;
				}
			}
			break;
		}
	}
}
__global__ void get_xylim_krnl(struct par_t *dpar, struct pos_t **pos,
		int4 *xylim, int nframes) {
	/* nframes-threaded kernel */
	int f = blockIdx.x * blockDim.x + threadIdx.x;
	if (f < nframes) {
		xylim[f].w = pos[f]->xlim[0];
		xylim[f].x = pos[f]->xlim[1];
		xylim[f].y = pos[f]->ylim[0];
		xylim[f].z = pos[f]->ylim[1];
	}
}
__global__ void posclr_radar_krnl_af(struct pos_t **pos, int *posn,
		int *xspan, int4 *xylim, int *npixels, int factor)
{
	/* This kernel launches as many blocks as needed for one frame.  Each thread handles
	 * just one pixel.  All frames in one set are done with this kernel and assigned by
	 * block ID (so each thread block is responsible for one frame).  Each block will
	 * do a grid-stride loop through the frame's POS bounding box pixels */
	int offset, i, j, f = blockIdx.x/factor;
	__shared__ int n, fr, xlim, ylim, xspan_sh;

	if (threadIdx.x==0) {
		n = posn[f];
		pos[f]->xlim[0] = pos[f]->ylim[0] =  n;
		pos[f]->xlim[1] = pos[f]->ylim[1] = -n;
		fr = (blockIdx.x%factor) * blockDim.x;
		xlim = xylim[f].w;
		ylim = xylim[f].y;
		xspan_sh = xspan[f];
	}
	__syncthreads();

	for (offset=threadIdx.x+fr; offset<npixels[f]; offset+=(factor*blockDim.x)) {
		i = (offset % xspan_sh) + xlim;
		j = (offset / xspan_sh) + ylim;
		/* For each POS pixel, zero out the optical brightness (b) and
		 * cos(scattering angle), reset the z coordinate (distance from COM towards
		 * Earth) to a dummy value, and reset the body, component, and facet onto
		 * which the pixel center projects to  dummy values                  */
		if (pos[f]->z[i][j] != -HUGENUMBER) {
			pos[f]->cose[i][j] = 0.0;
			pos[f]->z[i][j] = -HUGENUMBER;
		}
	}
}
__global__ void posclr_lc_krnl(struct pos_t **pos, int *posn, int f)
{
	/* This kernel launches as many blocks as needed for one frame.  Each thread handles
	 * just one pixel.  Frames are streamed. */
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	int i, j;

	__shared__ int n, nx, npixels, bistatic;

	if (threadIdx.x==0) {
		n = posn[f];
		nx = 2*n+1;
		npixels = nx*nx;
		bistatic = pos[f]->bistatic;

		if (offset==0) {
			pos[f]->xlim[0] = pos[f]->ylim[0] =  n;
			pos[f]->xlim[1] = pos[f]->ylim[1] = -n;

			if (bistatic) {
				pos[f]->xlim2[0] = pos[f]->ylim2[0] =  n;
				pos[f]->xlim2[1] = pos[f]->ylim2[1] = -n;
			}
		}
	}
	__syncthreads();

	if (offset < npixels) {
		i = (offset % nx) - n;
		j = (offset / nx) - n;

		/* For each POS pixel, zero out the optical brightness (b) and
		 * cos(scattering angle), reset the z coordinate (distance from COM towards
		 * Earth) to a dummy value, and reset the body, component, and facet onto
		 * which the pixel center projects to  dummy values                  */
		if (pos[f]->z[i][j]!=-HUGENUMBER) {
			pos[f]->b[i][j] = 0.0;
			pos[f]->cose[i][j] = 0.0;
			pos[f]->cosi[i][j] = 0.0;
			pos[f]->z[i][j] = -HUGENUMBER;

			/* For a bistatic situation (lightcurve or plane-of-sky dataset), zero out
			 * cos(incidence angle) and reset the distance towards the sun, the body,
			 * component, and facet numbers as viewed from the sun, and the model's
			 * maximum projected extent as viewed from the sun to dummy values    */
			if (bistatic) {
				pos[f]->cosill[i][j] = 0.0;
				pos[f]->zill[i][j] = -HUGENUMBER;
			}
		}
	}
}
__global__ void posclr_lc_krnl_af(struct pos_t **pos, int *posn, int *xspan, int4 *xylim, int *npixels, int blocks)
{
	/* This kernel launches as many blocks as needed for one frame.  Each thread handles
	 * just one pixel.  All frames in one set are done with this kernel and assigned by
	 * block ID (so each thread block is responsible for one frame).  Each block will
	 * do a grid-stride loop through the frame's POS bounding box pixels */
	int offset, i, j, f = blockIdx.x/blocks + 1;
	__shared__ int n, fr, xlim, ylim, xspan_sh, bistatic;

	if (threadIdx.x==0) {
		n = posn[f];
		pos[f]->xlim[0] = pos[f]->ylim[0] =  n;
		pos[f]->xlim[1] = pos[f]->ylim[1] = -n;
		fr = (blockIdx.x%blocks) * blockDim.x;
		xlim = xylim[f].w;
		ylim = xylim[f].y;
		xspan_sh = xspan[f];
		bistatic = pos[f]->bistatic;

		if (bistatic) {
			pos[f]->xlim2[0] = pos[f]->ylim2[0] =  n;
			pos[f]->xlim2[1] = pos[f]->ylim2[1] = -n;
		}
	}
	__syncthreads();

	for (offset=threadIdx.x+fr; offset<npixels[f]; offset+=(blocks*blockDim.x)) {
		i = (offset % xspan_sh) + xlim;
		j = (offset / xspan_sh) + ylim;
		/* For each POS pixel, zero out the optical brightness (b) and
		 * cos(scattering angle), reset the z coordinate (distance from COM towards
		 * Earth) to a dummy value, and reset the body, component, and facet onto
		 * which the pixel center projects to  dummy values                  */
//		if (pos[f]->z[i][j] != -HUGENUMBER) {
		pos[f]->cosi[i][j] = 0.0;
		pos[f]->cose[i][j] = 0.0;
		pos[f]->b[i][j] = 0.0;
		pos[f]->z[i][j] = -HUGENUMBER;
//		}
		if (bistatic/* && pos[f]->zill[i][j] != -HUGENUMBER*/) {
			pos[f]->cosill[i][j] = 0.0;
			pos[f]->zill[i][j] = -HUGENUMBER;
		}
	}
}
__global__ void opt_brightness_krnl(struct dat_t *ddat, int s, int n) {

	if (threadIdx.x == 0) {
		for (int i=1; i<=n; i++) {
			gpu_sum_opt_brightness64 += ddat->set[s].desc.lghtcrv.fit[i] *
					ddat->set[s].desc.lghtcrv.weight;
		}
	}
}

__host__ void vary_params_gpu(
		struct par_t *dpar,
		struct mod_t *dmod,
		struct dat_t *ddat,
		int action,
		double *deldop_zmax,
		double *rad_xsec,
		double *opt_brightness,
		double *cos_subradarlat,
		int *hnframes,
		int *hlc_n,
		int *nviews,
		struct vertices_t **verts,
		unsigned char *htype,
		unsigned char *dtype,
		int nf,
		int nsets,
		cudaStream_t *vp_stream,
		int max_frames,
		double **fit_overflow)
{
	/* This third iteration uses streams that are passed as argument.
	 * It also does not calculate/copy the various parameters but accepts
	 * them as arguments. Many doubles are floats or CUDA internal types like
	 * float3 or int4.
	 * Inputs:
	 * int action 				- dpar->action
	 * double *deldop_zmax 		- &deldop_zmax_save
	 * double *rad_xsec 		- &rad_xsec_save
	 * double *optbrightness	- &opt_brightness_save
	 * double *cos_subradarlat	- &cos_subradarlat_save
	 * int nsets 				- ddat->nsets
	 */

	int c=0, f, s, *compute_brightness, *compute_zmax, *bistatic, bistatic_all,
		*compute_cosdelta, *compute_xsec, nfrm_alloc, nfrm_alloc_max, *posn,
		*ndel, *ndop, blocks;
	nfrm_alloc_max = max_frames + 1;
	int hcomp_xsec[nfrm_alloc_max], npxls_full[nfrm_alloc_max],
		hndop[nfrm_alloc_max], hndel[nfrm_alloc_max], lc_xspan[nfrm_alloc_max],
		*outbndarr,	hposn[nfrm_alloc_max], hbistatic[nfrm_alloc_max],
		nThreadspx1[nfrm_alloc_max], hcomp_cosdelta[nsets], hcomp_zmax[nsets],
		hcomp_brightness[nsets];

	int *dxspan, *hxspan, *ddeldopsize, *hdeldopsize, *dnpxls_bbox, *hnpxls_bbox;
	int2 span[nfrm_alloc_max];
	int4 *xylim, hxylim[nfrm_alloc_max];
	double *u, zmax, *pixels_per_km, xsec[nfrm_alloc_max];
	double3 orbit_offset, *so;
	struct pos_t **pos;
	struct dopfrm_t **dframe;
	struct deldopfrm_t **ddframe;

	dim3 BLKfrm, THD,BLKncalc,THD64, BLKpx1, BLK[nfrm_alloc_max], ddBLK[nfrm_alloc_max];
	THD.x = maxThreadsPerBlock;	THD64.x = 64;

	dim3 THDaf, BLKaf;

	/* Initialize */
	orbit_offset.x = orbit_offset.y = orbit_offset.z = 0.0;

	/* Some arrays are allocated for the maximum number of frames in
	 * any one set.  That way they are allocated once and deallocated once.
	 * They are re-used for each loop.	 */
	gpuErrchk(cudaMalloc((void**)&compute_brightness, sizeof(int)*nsets));
	gpuErrchk(cudaMalloc((void**)&compute_zmax, sizeof(int)*nsets));
	gpuErrchk(cudaMalloc((void**)&compute_cosdelta, sizeof(int)*nsets));
	cudaCalloc1((void**)&pos, sizeof(struct pos_t*), nfrm_alloc_max);
	gpuErrchk(cudaMalloc((void**)&posn, 	   sizeof(int) * nfrm_alloc_max));
	gpuErrchk(cudaMalloc((void**)&ndel, 	   sizeof(int) * nfrm_alloc_max));
	gpuErrchk(cudaMalloc((void**)&ndop, 	   sizeof(int) * nfrm_alloc_max));
	gpuErrchk(cudaMalloc((void**)&outbndarr,   sizeof(int) * nfrm_alloc_max));
	gpuErrchk(cudaMalloc((void**)&compute_xsec,sizeof(int) * nfrm_alloc_max));
	gpuErrchk(cudaMalloc((void**)&xylim, 	   sizeof(int4)* nfrm_alloc_max));
	gpuErrchk(cudaMalloc((void**)&so, sizeof(double3)*(nfrm_alloc_max*3)));
	gpuErrchk(cudaMalloc((void**)&pixels_per_km, sizeof(int)*nfrm_alloc_max));
	gpuErrchk(cudaMalloc((void**)&u, sizeof(double) * nfrm_alloc_max));
	gpuErrchk(cudaMalloc((void**)&dframe, sizeof(struct dopfrm_t*)*nfrm_alloc_max));
	gpuErrchk(cudaMalloc((void**)&ddframe, sizeof(struct deldopfrm_t*)*nfrm_alloc_max));
	gpuErrchk(cudaMalloc((void**)&bistatic, sizeof(int)*nfrm_alloc_max));
	gpuErrchk(cudaMalloc((void**)&dxspan, sizeof(int)*nfrm_alloc_max));
	gpuErrchk(cudaMalloc((void**)&dnpxls_bbox, sizeof(int)*nfrm_alloc_max));
	gpuErrchk(cudaMalloc((void**)&ddeldopsize, sizeof(int)*nfrm_alloc_max));
	hnpxls_bbox = (int *) malloc(nfrm_alloc_max*sizeof(int));
	hxspan = (int *) malloc(nfrm_alloc_max*sizeof(int));
	hdeldopsize = (int *) malloc(nfrm_alloc_max*sizeof(int));

	/* Initialize the device file-scope variables */
	init_krnl<<<1,1>>>(dpar, ddat, compute_zmax, compute_cosdelta,
			compute_brightness, dtype,nsets);
	checkErrorAfterKernelLaunch("init_krnl");
	gpuErrchk(cudaMemcpy(&hcomp_cosdelta, compute_cosdelta,
			sizeof(int)*nsets, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&hcomp_zmax, compute_zmax,
			sizeof(int)*nsets, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&hcomp_brightness, compute_brightness,
			sizeof(int)*nsets, cudaMemcpyDeviceToHost));

	/* Process each dataset in turn */
	for (s=0; s<nsets; s++) {
		/* Get the allocation right as the indices for lghtcrv start with 1
		 * instead of 0 everywhere else. To not run into problems at the end
		 * or start of the array, we allocate one more than strictly needed */
		if (htype[s]==LGHTCRV)	nfrm_alloc = hnframes[s] + 1;
		else					nfrm_alloc = hnframes[s];

		/* Set up set-specific kernel launch parameters */
		BLKfrm = floor((THD64.x - 1 + nfrm_alloc) / THD64.x);
		THDaf.x = 1024;
		blocks = 3;
		BLKaf.x = nfrm_alloc * blocks;

		switch (htype[s]) {
		case DELAY:

			/* Launch nframes-threaded kernel to get all relevant parameters */
			delay_params_krnl<<<BLKfrm,THD64>>>(dpar, ddat, pos, ddframe,
					compute_xsec, posn, ndel, ndop, s, nfrm_alloc, xylim);
			checkErrorAfterKernelLaunch("delay_params_krnl");
			gpuErrchk(cudaMemcpy(&hposn, posn, sizeof(int)*nfrm_alloc,
					cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(&hcomp_xsec, compute_xsec,
					sizeof(int)*nfrm_alloc, cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(&hndop, ndop, sizeof(int)*nfrm_alloc,
					cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(&hndel, ndel, sizeof(int)*nfrm_alloc,
					cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(&hxylim, xylim, sizeof(int4)*nfrm_alloc,
					cudaMemcpyDeviceToHost));

			/* Calculate launch parameters for kernels */
			for (f=0; f<nfrm_alloc; f++) {
				npxls_full[f] = (2*hposn[f] + 1)*(2*hposn[f] + 1);
				hxspan[f] = hxylim[f].x - hxylim[f].w + 1;
				hnpxls_bbox[f] = (hxylim[f].z - hxylim[f].y + 1) * hxspan[f];
				BLK[f].x = floor((THD.x - 1 + npxls_full[f]) / THD.x);
				hdeldopsize[f]= hndel[f] * hndop[f];
				ddBLK[f] = floor((THD.x -1 + hdeldopsize[f]) / THD.x);
			}

			gpuErrchk(cudaMemcpy(dnpxls_bbox, hnpxls_bbox, sizeof(int)*nfrm_alloc,cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(dxspan, hxspan, sizeof(int)*nfrm_alloc,cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(ddeldopsize, hdeldopsize, sizeof(int)*nfrm_alloc, cudaMemcpyHostToDevice));

			/* Assign ae and oe matrices (all frames at once) */
			if (hcomp_zmax[s])
				set_ae_oe_krnl<<<BLKfrm,THD64>>>(ddat, pos, htype[s], s, nfrm_alloc);
			checkErrorAfterKernelLaunch("set_ae_oe_krnl");

			/* Launch posclr_streams_krnl to initialize POS view */
			posclr_radar_krnl_af<<<BLKaf,THDaf>>>(pos, posn, dxspan, xylim, dnpxls_bbox,
					blocks);
			checkErrorAfterKernelLaunch("posclr_radar_krnl_af ");

			/* Determine which POS pixels cover the target, and get distance
			 * toward Earth of each POS pixel. Pass the frame streams, too. */
			posvis_gpu(dpar, dmod, pos, verts, orbit_offset,
					hposn, outbndarr, s, nfrm_alloc, 0, nf, 0, c, htype[s],
					vp_stream, 0);

			clrvect_krnl_af<<<BLKaf,THDaf>>>(ddat, ddeldopsize, s, blocks);
			checkErrorAfterKernelLaunch("clrvect_krnl_af");

			/* Call the CUDA pos2deldop function */
			pos2deldop_gpu(dpar, dmod, ddat, pos, ddframe, xylim, ndel, ndop,
					0.0, 0.0, 0.0, 0, s, nfrm_alloc, 0, outbndarr, vp_stream, fit_overflow);

			/* Calculate zmax for all frames (assumption: all pos in this set
			 * have the same pixel dimensions) */
			if (hcomp_zmax[s]) {
				zmax = compute_zmax_gpu(ddat, pos, nfrm_alloc, npxls_full[0], s, vp_stream);
				zmax_final_krnl<<<1,1>>>(zmax);
				checkErrorAfterKernelLaunch("zmax_final_krnl");
			}

			/* Calculate radar cross section for each frame in set */
			xsec[0] = compute_deldop_xsec_gpu(ddat, hnframes[s], hdeldopsize[0], s, vp_stream);
			xsec_deldop_krnl<<<1,1>>>(xsec[0]);
			checkErrorAfterKernelLaunch("xsec_deldop_krnl");

			if (hcomp_cosdelta[s])
				cosdelta_krnl<<<BLKfrm,THD64>>>(ddat, s, f);
			checkErrorAfterKernelLaunch("cosdelta_krnl");

			break;
		case DOPPLER:
			/* Launch nframes-threaded kernel to get all relevant parameters */
			dop_params_krnl<<<BLK[0],THD>>>(dpar, ddat, pos, dframe,
					compute_xsec, posn, ndop, s, nfrm_alloc, xylim);
			checkErrorAfterKernelLaunch("dop_params_krnl");
			gpuErrchk(cudaMemcpy(&hposn, posn, sizeof(int)*nfrm_alloc,
					cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(&hcomp_xsec, compute_xsec,
					sizeof(int)*nfrm_alloc, cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(&hndop, ndop, sizeof(int)*nfrm_alloc,
					cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(&hxylim, xylim, sizeof(int4)*nfrm_alloc,
					cudaMemcpyDeviceToHost));

			/* Calculate launch parameters*/
			for (f=0; f<nfrm_alloc; f++) {
				npxls_full[f] = (2*hposn[f] + 1)*(2*hposn[f] + 1);
				hxspan[f] = hxylim[f].x - hxylim[f].w + 1;
				hnpxls_bbox[f] = (hxylim[f].z - hxylim[f].y + 1) * hxspan[f];
				BLK[f].x = floor((THD.x - 1 + npxls_full[f]) / THD.x);
				ddBLK[f] = floor((THD.x -1 + hndop[f]) / THD.x);
			}
			gpuErrchk(cudaMemcpy(dnpxls_bbox, hnpxls_bbox, sizeof(int)*nfrm_alloc,cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy(dxspan, hxspan, sizeof(int)*nfrm_alloc,cudaMemcpyHostToDevice));

			/* Assign ae and oe matrices (all frames at once) */
			set_ae_oe_krnl<<<BLKfrm,THD64>>>(ddat, pos, htype[s], s, nfrm_alloc);
			checkErrorAfterKernelLaunch("set_ae_oe_krnl");

			/* Initialize POS view by resetting all relevant pos arrays for
			 * every frame in this set */
			posclr_radar_krnl_af<<<BLKaf,THDaf>>>(pos, posn, dxspan, xylim, dnpxls_bbox,
					blocks);
			checkErrorAfterKernelLaunch("posclr_radar_krnl_af ");

			/* Determine which POS pixels cover the target, and get distance
			 * toward Earth of each POS pixel. Pass the frame streams, too. */
			posvis_gpu(dpar, dmod, pos, verts, orbit_offset, hposn,
					outbndarr, s, nfrm_alloc, 0, nf, 0, c, htype[s], vp_stream, 0);

			for (f=0; f<nfrm_alloc; f++) {
				if (hcomp_xsec[f]) {
					/* Zero out the fit delay-Doppler image and call pos2deldop
					 * to create the fit image by mapping power from the plane
					 * of sky to delay-Doppler space.    				      */
					clrvect_krnl<<<ddBLK[f],THD, 0, vp_stream[f]>>>(ddat,
							hndop[f], s, f);
					/* End frames loop again to call pos2deldop streams version */
				}
			} checkErrorAfterKernelLaunch("clrvect_krnl");

			/* Synchronize streams to default stream */
			for (f=0; f<nfrm_alloc; f++)
				cudaStreamSynchronize(vp_stream[f]);

			pos2doppler_gpu(dpar, dmod, ddat, pos, dframe, xylim, 0.0,
					0.0, 0.0, ndop, 0, s, hnframes[s], 0, outbndarr, vp_stream);

			/* Calculate the Doppler cross-section if applicable */
			for (f=0; f<nfrm_alloc; f++) {
				if (hcomp_xsec[f]) {
					/* Compute cross section */
					xsec[f]=0.0;
					xsec[f] = compute_doppler_xsec(ddat, hndop[f], s, f);
				}
			}
			/* Finalize the xsec calculations and calculate cosdelta if specified */
			for (f=0; f<nfrm_alloc; f++) {
				if (hcomp_xsec[f])
					xsec_doppler_krnl<<<1,1,0,vp_stream[f]>>>(ddat, xsec[f], s, f);
			}

			/* Synchronize streams to default stream */
			for (f=0; f<nfrm_alloc; f++)
				cudaStreamSynchronize(vp_stream[f]);

			if (compute_cosdelta)
				cosdelta_krnl<<<BLKfrm,THD64>>>(ddat, s, nfrm_alloc);

			break;
		case POS:
			break;
		case LGHTCRV:

			if (hcomp_brightness[s]) {
				/* Launch nframes-threaded kernel to get all relevant parameters */
				lghtcrv_params_krnl<<<BLK[0],THD>>>(dpar, ddat, pos,
						posn, bistatic, s, nfrm_alloc, xylim);
				checkErrorAfterKernelLaunch("lghtcrv_params_krnl");
				gpuErrchk(cudaMemcpy(&hposn, posn, sizeof(int)*nfrm_alloc,
						cudaMemcpyDeviceToHost));
				gpuErrchk(cudaMemcpy(&hbistatic, bistatic, sizeof(int)*nfrm_alloc,
						cudaMemcpyDeviceToHost));
				gpuErrchk(cudaMemcpy(&hxylim, xylim, sizeof(int4)*nfrm_alloc,
						cudaMemcpyDeviceToHost));

				/* Calculate launch parameters and create streams */
				for (f=1; f<nfrm_alloc; f++) {
					lc_xspan[f] = 2*posn[f] + 1;
					npxls_full[f] = (2*posn[f]+1)*(2*posn[f]+1);
					BLK[f].x = floor((THD.x - 1 + npxls_full[f]) / THD.x);
				}

				/* Assign ae and oe matrices (all frames at once) */
				set_ae_oe_krnl<<<BLKfrm,THD64>>>(ddat, pos, htype[s], s, nfrm_alloc);
				checkErrorAfterKernelLaunch("set_ae_oe_krnl");

				/* Launch posclr_streams_krnl to initialize POS view */
				for (f=1; f<nfrm_alloc; f++) {
					/* Start the if block for computing zmax and/or cross-section */
					if (hcomp_xsec[f])
						posclr_lc_krnl<<<BLK[f],THD,0,vp_stream[f-1]>>>(pos, posn, f);
				} checkErrorAfterKernelLaunch("posclr_lc_krnl");

				/* Synchronize streams to default stream */
				for (f=1; f<=nfrm_alloc; f++)
					cudaStreamSynchronize(vp_stream[f-1]);

				/* Determine which POS pixels cover the target */
				posvis_gpu(dpar, dmod, pos, verts, orbit_offset,
						hposn, outbndarr, s, hnframes[s], 0, nf, 0, c, htype[s],
						vp_stream, 0);

				/* Now view the model from the source (sun) and get the facet
				 * number and distance toward the source of each pixel in this
				 * projected view; use this information to determine which POS
				 * pixels are shadowed */
				for (f=1; f<nfrm_alloc; f++)
					if (hbistatic[f])	bistatic_all = 1;

				if (bistatic_all)
					posvis_gpu(dpar, dmod, pos, verts,
							orbit_offset, hposn, outbndarr, s, hnframes[s], 1,
							nf, 0, c, htype[s],	vp_stream, 1);

				if (bistatic_all) {
					posmask_init_krnl<<<BLKfrm,THD64>>>(pos, so, pixels_per_km, nfrm_alloc);
					checkErrorAfterKernelLaunch("posmask_init_krnl");

					for (f=1; f<nfrm_alloc; f++) {
						/* Now call posmask kernel for this stream, then loop
						 * to next stream and repeat 	 */
						posmask_krnl<<<BLK[f],THD,0,vp_stream[f-1]>>>(
								dpar, pos, so, pixels_per_km, posn, npxls_full[f],
								lc_xspan[f], f);
					} checkErrorAfterKernelLaunch("posmask_krnl");

					/* Synchronize streams to default stream */
					for (f=1; f<=nfrm_alloc; f++)
						cudaStreamSynchronize(vp_stream[f-1]);
				}

				BLKpx1.x = floor((THD.x - 1 + hnframes[s])/THD.x);
				get_xylim_krnl<<<BLKpx1,THD>>>(dpar, pos, xylim, hnframes[s]);
				checkErrorAfterKernelLaunch("get_xylim_krnl");
				gpuErrchk(cudaMemcpy(&hxylim, xylim, sizeof(int4)*(hnframes[s]+1),
						cudaMemcpyDeviceToHost));

				/* Calculate launch parameters for all frames */
				int xspan_max = 0, yspan_max = 0, maxthds = 0;
				int4 maxxylim;
				maxxylim.w = maxxylim.y = 1e3;
				maxxylim.x = maxxylim.z = -1e3;
				for (f=1; f<=hnframes[s]; f++) {
					span[f].x = hxylim[f].x - hxylim[f].w + 1;
					span[f].y = hxylim[f].z - hxylim[f].y + 1;
					nThreadspx1[f] = span[f].x * span[f].y;
					BLK[f].x = floor ((THD.x -1 + nThreadspx1[f]) / THD.x);
					maxxylim.w = min(maxxylim.w, hxylim[f].w);
					maxxylim.x = max(maxxylim.x, hxylim[f].x);
					maxxylim.y = min(maxxylim.y, hxylim[f].y);
					maxxylim.z = max(maxxylim.z, hxylim[f].z);
				}
				xspan_max = maxxylim.x - maxxylim.w + 1;
				yspan_max = maxxylim.z - maxxylim.y + 1;
				maxthds = xspan_max * yspan_max;

				/* Compute model brightness for this lightcurve point */
				/* lghtcrv->y[ncalc]: calculated points for interpolation,
				 * ncalc-points total 					 */
				apply_photo_gpu(dmod, ddat, pos, xylim, span, BLK, nThreadspx1,
							0, s, hnframes[s], maxthds, maxxylim, vp_stream);

				/* Now that we have calculated the model lightcurve brightnesses
				 * y at each of the epochs x, we use cubic spline interpolation
				 * (Numerical Recipes routines spline and splint) to get model
				 * lightcurve brightness fit[i] at each OBSERVATION epoch t[i],
				 * with i=1,2,...,n.  This will allow us (in routine chi2) to
				 * compare model to data (fit[i] to obs[i]) to get chi squared.
				 * Note that vector y2 contains the second derivatives of the
				 * interpolating function at the calculation epochs x. */

				/* First make a pointer for u and cudaMalloc device memory for it */
				gpuErrchk(cudaMemset(u, 0, nfrm_alloc_max*sizeof(double)));

				lghtcrv_spline_krnl<<<BLKncalc,THD>>>(ddat, s, 2.0e30,
						2.0e30, u, hnframes[s]);
				checkErrorAfterKernelLaunch("lghtcrv_spline_krnl");

				/* Change launch parameters from ncalc threads to n threads */
				//BLKncalc.x = floor((THD.x - 1 + hlc_n[s]) / THD.x);
				lghtcrv_splint_krnl<<<1,1>>>(ddat, s, hlc_n[s], hnframes[s]);
				checkErrorAfterKernelLaunch("lghtcrv_splint_krnl");

				/* Finalize the optical brightness calculation */
				opt_brightness_krnl<<<1,1>>>(ddat, s, hlc_n[s]);
				checkErrorAfterKernelLaunch("opt_brightness_krnl");
			}
			break;
		default:
			bailout("vary_params.c: can't handle this dataset type yet\n");
		}
	}

	/* Calculate the zmax, radar cross-section, optical brightness, and cosine
	 * subradar latitude */
	double dd_zmax, rd_xsec, opt_brtns, cs_sb_rdr_lat;
	finalize_krnl<<<1,1>>>(ddat);
	checkErrorAfterKernelLaunch("vps_set_four_parameters, line ");
	gpuErrchk(cudaMemcpyFromSymbol(&dd_zmax, gpu_deldop_zmax,
			sizeof(double), 0, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpyFromSymbol(&rd_xsec, gpu_rad_xsec,
			sizeof(double), 0, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpyFromSymbol(&opt_brtns, gpu_opt_brightness,
			sizeof(double), 0, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpyFromSymbol(&cs_sb_rdr_lat, gpu_cos_subradarlat,
			sizeof(double), 0, cudaMemcpyDeviceToHost));

	*deldop_zmax = dd_zmax;
	*rad_xsec = rd_xsec;
	*opt_brightness = opt_brtns;
	*cos_subradarlat = cs_sb_rdr_lat;

	cudaFree(u);
	cudaFree(so);
	cudaFree(pixels_per_km);
	cudaFree(pos);
	cudaFree(posn);
	cudaFree(ndel);
	cudaFree(ndop);
	cudaFree(outbndarr);
	cudaFree(compute_xsec);
	cudaFree(xylim);
	cudaFree(compute_brightness);
	cudaFree(compute_zmax);
	cudaFree(compute_cosdelta);
	cudaFree(dframe);
	cudaFree(ddframe);

	cudaFree(dnpxls_bbox);
	cudaFree(ddeldopsize);
	cudaFree(dxspan);
	free(hnpxls_bbox);
	free(hdeldopsize);
	free(hxspan);
}
