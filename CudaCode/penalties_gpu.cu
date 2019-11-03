
/*****************************************************************************************
                                                                              penalties.c
Compute the penalty functions for all penalties being applied to this model
Modified 2016 December 8 by ME:
	Converted to "FIT" action-only, CUDA code
Modified 2014 February 15 by CM:
    Adjust several penalty functions to accommodate multiple radar and optical
        scattering laws within a mod file
Modified 2013 July 6 by CM:
    Add the "pa3tilt" penalty
Modified 2011 September 2 by CM:
    Add the "harmlambert" and "inholambert" cases for the "optalbdel" and "optalbvar"
        penalties
Modified 2011 August 22 by CM:
    Add the "impulse" penalty
Modified 2010 August 31 by CM:
    Fix bug in the "noncosine" penalty: forgot to assign value to n
Modified 2010 June 1 by CM:
    Revise the "rdev" and "maxrdev" penalties now that the "scalefactor"
        parameter is a 3-component vector rather than a scalar
Modified 2010 May 12 by CM:
    Revise the "bifurcation" penalty so that it's not sensitive to the
        exact positions of vertices that lie near zone boundaries
Modified 2010 April 27 by CM:
    Add the "noncosine" and "bifurcation" penalties
Modified 2009 November 15 by CM:
    Remove unused variable
Modified 2009 August 2 by CM:
    Adjust the nonsmooth and concavity penalties to pay attention to the
        "act" (active) flags of model sides rather than of model facets
Modified 2009 July 2 by CM:
    For various penalties, only sum over active facets/vertices/sides,
        thus excluding interior regions for multiple-component models
    For the "rdev" "maxrdev" and "maxellipdev" penalties, scale to the
        overall model's effective radius, not to the effective radius for
        each component
    For the "maxellipdev" penalty, compute deviations from the overall
        model's DEEVE, not from each component's DEEVE
Modified 2007 February 21 by CM:
    Add the "maxrdev" and "maxellipdev" penalties
    Fix bugs in "rdev" penalty, and change this penalty so that each
        vertex deviation is normalized to the effective radius of that
        component
    Change the "comdev" penalty so that the COM displacement is normalized
        to the model's effective radius
Modified 2005 September 7 by CM:
    Add the "harmlommel" "harmhapke" and "harmkaas" cases for the
        "optalbdel" and "optalbvar" penalties
    Add the "harmhapke" case for the "thetadel" and "thetavar" penalties
    Add the "harmcosine" case for the "radalbdel" "radalbvar" "rad_c_del"
        and "rad_c_var" penalties
Modified 2005 August 10 by CM:
    Add the "inhokaas" case for the "optalbdel" and "optalbvar" penalties
Modified 2005 July 20 by CM:
    Add the "thetadel" penalty for the "inhohapke" optical scattering law
    Add four penalties for the "inhocosine" radar scattering law:
        "radalbdel" "radalbvar" "rad_c_del" "rad_c_var"
    Don't display "changed negative penalty to 0.0" messages at all, since
        these situations are always due to slight roundoff error
Modified 2005 July 7 by CM:
    Don't display "changed negative penalty to 0.0" messages to the screen
        unless the par->showstate flag is turned on
Modified 2005 July 4 by CM:
    Adjust the structure for the "inholommel" optical scattering law
    Enable the "optalbdel" penalty for the "inhohapke" optical scattering
        law
    Protect against division by zero for "rdev" penalty
Modified 2005 March 8 by CM:
    Fix bug with negative penalty weights
Modified 2005 February 28 by CM:
    Eliminate checks that photometric parameters are valid, since these
        checks are now performed in routine realize_photo
Modified 2005 January 25 by CM:
    Initialize variable to avoid compilation warning
Modified 2004 May 4 by CM:
    Added "flattening" penalty
Modified 2004 April 25 by CM:
    Added "inertiadev_uni" and "nonpa_uni" penalties for PA rotators
Modified 2003 April 21 by CM:
    Added comments
    Protected against division by zero (for negative penalty weights)
Modified 2003 April 17 by CM:
    Removed the large code block for computing 0th, 1st, and 2nd-order
        moments, partly because it wasn't executed if none of the three
        penalties VOLUME and COMDEV and INERTIADEV were used, partly
        because it would be nice to know volumes and COM displacements
        and inertia tensors even when there's no need to evaluate
        penalties.  (Added this same code to function realize_mod.)
 *****************************************************************************************/
extern "C" {
#include "../shape/head.h"
}

#define RHOCUTOFF 1.0e-20
#define NBIFURCATION 20
#define NZONES (NBIFURCATION - 1)
#define TINYPEN 1.0e-10

__device__ int p_pen_n, p_pen_type, p_ns, p_nf, p_nv, p_ntot, p_got_pa, p_j1,
			   p_j2, p_jmax;
__device__ double p_a, p_b, p_pen, p_av, p_av2, p_min_extent, p_max_extent,
			   p_sumrho2[NBIFURCATION], p_nrho2[NBIFURCATION], p_meanrho2[NBIFURCATION],
			   p_deepest_minimum;
__device__ double p_volume, p_r_eff, p_ap[3][3], p_DEEVE_radius[3],
			   p_axis_increment, p_sum=0.0, pmoment[3];
__device__ unsigned char p_shape_type;

/* Note that the following two custom atomic functions are declared in each
 * file they are needed .  As static __device__ functions, this is the only
 * way to handle them. */

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
__device__ void dev_diag_inertia(double inertia[3][3]){
	int i, j, nrot;
	double jaca[3][3], jacv[3][3], jacd[3];

	/* Use Numerical Recipes "jacobi" routine to diagonalize the inertia tensor:
	 *
	 *  jaca = inertia tensor in body coordinates
	 *  	    (jacobi destroys the part of jaca above the diagonal)
	 *  jacd = eigenvalues of inertia tensor = principal moments of inertia
	 *  jacv = matrix whose columns are the eigenvectors of the inertia tensor:
	 *
	 * Each column of jacv is the direction cosines, in body coordinates, of the
	 * corresponding principal axis; jacv as a whole is the transformation matrix
	 * that takes us from principal-axis coordinates to body coordinates.
	 * (Vector pmoment and matrix ap are the same as jacd and jacv, respectively,
	 * but with rows and columns numbered 0-2 instead of 1-3.)  */

	for (i=0; i<3; i++)
		for (j=0; j<3; j++)
			jaca[i][j] = inertia[i][j];
	dev_jacobi(jaca, 3, jacd, jacv, &nrot);    /*  nrot = # of rotations required  */

	for (i=0; i<3; i++) {
		pmoment[i] = jacd[i];
		for (j=0; j<3; j++)
			p_ap[i][j] = jacv[i][j];
	}
}

__global__ void p_get_pen_n_krnl(struct par_t *dpar) {
	/* Single-threaded kernel */
	if (threadIdx.x == 0) {
		p_pen_n = dpar->pen.n;

		/* Now some initializations */
		p_a = p_b = 0.0;
		p_ntot = 0;
		p_av = p_av2 = 0.0;
		p_got_pa = 0;
		p_min_extent = HUGENUMBER;
		p_max_extent = -HUGENUMBER;
		p_deepest_minimum = HUGENUMBER;
		p_sum = 0.0;
		p_pen = 0.0;
	}
}
__global__ void p_get_pen_type_krnl(struct par_t *dpar, struct mod_t
		*dmod, int i) {
	/* Single-threaded kernel */
	int c=0;
	int k, j;
	if (threadIdx.x == 0) {
		p_pen_type = dpar->pen.type[i];
		p_shape_type = dmod->shape.comp[c].type;
		for (k=0; k<3; k++)
			for (j=0; j<3; j++)
				p_ap[k][j] = 0.0;
		p_ntot = 0;
		p_pen = 0.0;
	}
}
__global__ void p_get_real_info_krnl(struct mod_t *dmod) {
	/* Single-threaded kernel */
	int c = 0;
	if (threadIdx.x == 0) {
		p_ns = dmod->shape.comp[c].real.ns;
		p_nf = dmod->shape.comp[c].real.nf;
		p_nv = dmod->shape.comp[c].real.nv;
	}
}
__global__ void p_optalbdel_krnl(struct mod_t *dmod, double *a, double *b) {
	/* ns-threaded kernel */
	int s = blockIdx.x * blockDim.x + threadIdx.x;
	int c = 0, ilaw, f1, f2, v1, v2;
	double len, x, y;

	if (s < p_ns) {

		f1 = dmod->shape.comp[c].real.s[s].f[0];
		f2 = dmod->shape.comp[c].real.s[s].f[1];
		if (dmod->shape.comp[c].real.f[f1].act &&
			dmod->shape.comp[c].real.f[f2].act    ) {
			v1 = dmod->shape.comp[c].real.s[s].v[0];
			v2 = dmod->shape.comp[c].real.s[s].v[1];
			len = dev_distance(dmod->shape.comp[c].real.v[v1].x,
					       dmod->shape.comp[c].real.v[v2].x);

			for (ilaw=0; ilaw<dmod->photo.noptlaws; ilaw++) {
				switch (dmod->photo.opttype[ilaw]) {
				case HARMLAMBERT:
				case HARMLOMMEL:
					x = dmod->photo.optical[ilaw].harmR.local[c][f1].R.val;
					y = dmod->photo.optical[ilaw].harmR.local[c][f2].R.val;
					a[s] = len * fabs(x+y);
					b[s] = len * fabs(x-y);
					break;
				case INHOLAMBERT:
				case INHOLOMMEL:
					x = dmod->photo.optical[ilaw].inhoR.local[c][f1].R.val;
					y = dmod->photo.optical[ilaw].inhoR.local[c][f2].R.val;
					a[s] = len * fabs(x+y);
					b[s] = len * fabs(x-y);
					break;
				case HARMHAPKE:
					x = dmod->photo.optical[ilaw].harmhapke.local[c][f1].w.val;
					y = dmod->photo.optical[ilaw].harmhapke.local[c][f2].w.val;
					a[s] = len * fabs(x+y);
					b[s] = len * fabs(x-y);
					break;
				case INHOHAPKE:
					x = dmod->photo.optical[ilaw].inhohapke.local[c][f1].w.val;
					y = dmod->photo.optical[ilaw].inhohapke.local[c][f2].w.val;
					a[s] = len * fabs(x+y);
					b[s] = len * fabs(x-y);
					break;
				case HARMKAAS:
					x = dmod->photo.optical[ilaw].harmkaas.local[c][f1].R.val;
					y = dmod->photo.optical[ilaw].harmkaas.local[c][f2].R.val;
					a[s] = len * fabs(x+y);
					b[s] = len * fabs(x-y);
					break;
				case INHOKAAS:
					x = dmod->photo.optical[ilaw].inhokaas.local[c][f1].R.val;
					y = dmod->photo.optical[ilaw].inhokaas.local[c][f2].R.val;
					a[s] = len * fabs(x+y);
					b[s] = len * fabs(x-y);
					break;
				}
			}
		}
	}
}
__global__ void p_optalbdel_finish_krnl(double asum, double bsum) {
	/* Single-threaded kernel to set p_pen */

	if (threadIdx.x == 0) {
		if (p_a == 0.0)
			printf("penalties_cuda.cu: 'optalbdel' can't be used with this radar scattering law\n");
		p_pen = bsum/asum; /* penalty = b/a */
	}
}
__global__ void p_optalbvar_krnl(struct mod_t *dmod, double *av, double *av2) {
	/* nf-threaded kernel */
	int f = blockIdx.x * blockDim.x + threadIdx.x;
	int c=0, ilaw;
	double x;

	if (f < p_nf) {

		if (dmod->shape.comp[c].real.f[f].act) {
			for (ilaw=0; ilaw<dmod->photo.noptlaws; ilaw++) {
				switch (dmod->photo.opttype[ilaw]) {
				case HARMLAMBERT:
				case HARMLOMMEL:
					x = dmod->photo.optical[ilaw].harmR.local[c][f].R.val;
					av[f] = x;
					av2[f] = x*x;
					atomicAdd(&p_ntot, 1);
					break;
				case INHOLAMBERT:
				case INHOLOMMEL:
					x = dmod->photo.optical[ilaw].inhoR.local[c][f].R.val;
					av[f] = x;
					av2[f] = x*x;
					atomicAdd(&p_ntot, 1);
					break;
				case HARMHAPKE:
					x = dmod->photo.optical[ilaw].harmhapke.local[c][f].w.val;
					av[f] = x;
					av2[f] = x*x;
					break;
				case INHOHAPKE:
					x = dmod->photo.optical[ilaw].inhohapke.local[c][f].w.val;
					av[f] = x;
					av2[f] = x*x;
					atomicAdd(&p_ntot, 1);
					break;
				case HARMKAAS:
					x = dmod->photo.optical[ilaw].harmkaas.local[c][f].R.val;
					av[f] = x;
					av2[f] = x*x;
					atomicAdd(&p_ntot, 1);
					break;
				case INHOKAAS:
					x = dmod->photo.optical[ilaw].inhokaas.local[c][f].R.val;
					av[f] = x;
					av2[f] = x*x;
					atomicAdd(&p_ntot, 1);
					break;
				}
			}
		}
	}
}
__global__ void p_optalbvar_finish_krnl(double avsum, double av2sum) {
	/* Single-threaded kernel to finish the penalty calculation for optalbvar */
	if (threadIdx.x == 0) {
		if (p_ntot == 0)
			printf("penalties_cuda.cu: 'optalbvar' can't be used with this radar scattering law\n");
		p_pen = p_ntot*(av2sum/(avsum*avsum)) - 1.0; /* fractional variance */
		if (p_pen < 0.0)	p_pen = 0.0;    /* roundoff error */
	}
}
__global__ void p_volume_krnl(struct mod_t *dmod) {
	/* Single-threaded kernel */
	if (threadIdx.x == 0) {
		p_pen = dmod->shape.volume;
	}
}
__global__ void p_thetadel_krnl(struct mod_t *dmod, double *a, double *b) {
	/* ns-threaded kernel */
	int s = blockIdx.x * blockDim.x + threadIdx.x;
	int c=0, ilaw, f1, f2, v1, v2;
	double len, x, y;
	if (s < p_ns) {
		f1 = dmod->shape.comp[c].real.s[s].f[0];
		f2 = dmod->shape.comp[c].real.s[s].f[1];
		if  (dmod->shape.comp[c].real.f[f1].act &&
			 dmod->shape.comp[c].real.f[f2].act    ) {
			v1 = dmod->shape.comp[c].real.s[s].v[0];
			v2 = dmod->shape.comp[c].real.s[s].v[1];
			len = dev_distance(dmod->shape.comp[c].real.v[v1].x,
					dmod->shape.comp[c].real.v[v2].x);

			for (ilaw=0; ilaw<dmod->photo.noptlaws; ilaw++) {
				switch (dmod->photo.opttype[ilaw]) {
				case HARMHAPKE:
					x = dmod->photo.optical[ilaw].harmhapke.local[c][f1].theta.val;
					y = dmod->photo.optical[ilaw].harmhapke.local[c][f2].theta.val;
					a[s] = len * fabs(x+y);
					b[s] = len * fabs(x-y);
					break;
				case INHOHAPKE:
					x = dmod->photo.optical[ilaw].inhohapke.local[c][f1].theta.val;
					y = dmod->photo.optical[ilaw].inhohapke.local[c][f2].theta.val;
					a[s] = len * fabs(x+y);
					b[s] = len * fabs(x-y);
					break;
				}
			}
		}
	}
}
__global__ void p_thetadel_finish_krnl(double asum, double bsum) {
	/* What follows is a single-thread task */
	if (threadIdx.x == 0) {
		if (asum == 0.0)
			printf("penalties_cuda.cu: 'thetadel' can't be used with this optical scattering law\n");
		p_pen = bsum/asum;
	}
}
__global__ void p_thetavar_krnl(struct mod_t *dmod, double *av, double *av2) {
	/* nf-threaded kernel */
	int f = blockIdx.x * blockDim.x + threadIdx.x;
	int c=0, ilaw;
	double x;

	if (f < p_nf){
		if (dmod->shape.comp[c].real.f[f].act) {
			for (ilaw=0; ilaw<dmod->photo.noptlaws; ilaw++) {
				switch (dmod->photo.opttype[ilaw]) {
				case HARMHAPKE:
					x = dmod->photo.optical[ilaw].harmhapke.local[c][f].theta.val;
					av[f] = x;
					av2[f] = x*x;
					atomicAdd(&p_ntot, 1);
					break;
				case INHOHAPKE:
					x = dmod->photo.optical[ilaw].inhohapke.local[c][f].theta.val;
					av[f] = x;
					av2[f] = x*x;
					atomicAdd(&p_ntot, 1);
					break;
				}
			}
		}
	}
}
__global__ void p_thetavar_finish_krnl(double avsum, double av2sum) {

	/* Single-threaded kernel */
	if (threadIdx.x == 0) {
		if (p_ntot == 0)
			printf("penalties_cuda.cu: 'thetavar' can't be used with this optical scattering law\n");
		p_pen = p_ntot * (av2sum/(avsum*avsum)) - 1.0; /* fractional variance */
		if (p_pen < 0.0)		p_pen = 0.0;    /* roundoff error */
	}
}
__global__ void p_radalbdel_krnl(struct mod_t *dmod, double *a, double *b) {
	/* ns-threaded kernel */
	int s = blockIdx.x * blockDim.x + threadIdx.x;
	int c=0, f1, f2, v1, v2, ilaw;
	double len, x, y;

	if (s < p_ns) {
		f1 = dmod->shape.comp[c].real.s[s].f[0];
		f2 = dmod->shape.comp[c].real.s[s].f[1];
		if (dmod->shape.comp[c].real.f[f1].act &&
				dmod->shape.comp[c].real.f[f2].act    ) {
			v1 = dmod->shape.comp[c].real.s[s].v[0];
			v2 = dmod->shape.comp[c].real.s[s].v[1];
			len = dev_distance(dmod->shape.comp[c].real.v[v1].x,
							   dmod->shape.comp[c].real.v[v2].x);
			for (ilaw=0; ilaw<dmod->photo.nradlaws; ilaw++) {
				switch (dmod->photo.radtype[ilaw]) {
				case HARMCOSINE_DIFF:
					x = dmod->photo.radar[ilaw].harmcosine.local[c][f1].R.val;
					y = dmod->photo.radar[ilaw].harmcosine.local[c][f2].R.val;
					a[s] = len * fabs(x+y);
					b[s] = len * fabs(x-y);
					break;
				case INHOCOSINE_DIFF:
					x = dmod->photo.radar[ilaw].inhocosine.local[c][f1].R.val;
					y = dmod->photo.radar[ilaw].inhocosine.local[c][f2].R.val;
					a[s] = len * fabs(x+y);
					b[s] = len * fabs(x-y);
					break;
				}
			}
		}
	}
}
__global__ void p_radalbdel_finish_krnl(double asum, double bsum) {

	/* Single-threaded kernel */
	if (threadIdx.x == 0) {
		if (asum == 0.0)
			printf("penalties_cuda.cu: 'radalbdel' can't be used with this "
					"radar scattering law\n");
		p_pen = bsum/asum;
	}
}
__global__ void p_radalbvar_krnl(struct mod_t *dmod, double *av, double *av2) {
	/* nf-threaded kernel */
	int f = blockIdx.x * blockDim.x + threadIdx.x;
	int c=0, ilaw;
	double x;

	if (f < p_nf) {
		if (dmod->shape.comp[c].real.f[f].act) {
			for (ilaw=0; ilaw<dmod->photo.nradlaws; ilaw++) {
				switch (dmod->photo.radtype[ilaw]) {
				case HARMCOSINE_DIFF:
					x = dmod->photo.radar[ilaw].harmcosine.local[c][f].R.val;
					av[f] = x;
					av2[f] = x*x;
					atomicAdd(&p_ntot, 1);
					break;
				case INHOCOSINE_DIFF:
					x = dmod->photo.radar[ilaw].inhocosine.local[c][f].R.val;
					av[f] = x;
					av2[f] = x*x;
					atomicAdd(&p_ntot, 1);
					break;
				}
			}
		}
	}
}
__global__ void p_radalbvar_finish_krnl(double avsum, double av2sum){
	/* Single-threaded task */
	if (threadIdx.x == 0) {
		if (p_ntot == 0)
			printf("penalties_cuda.cu: 'radalbvar' can't be used with this "
					"radar scattering law\n");
		p_pen = p_ntot * (av2sum/(avsum*avsum)) - 1.0; /* fractional variance */
		if (p_pen < 0.0)		p_pen = 0.0;    /* roundoff error */
	}
}
__global__ void p_radcdel_krnl(struct mod_t *dmod, double *a, double *b) {
	/* ns-threaded kernel */
	int s = blockIdx.x * blockDim.x + threadIdx.x;
	int c=0, ilaw, f1, f2, v1, v2;
	double len, x, y;

	if (s < p_ns) {
		f1 = dmod->shape.comp[c].real.s[s].f[0];
		f2 = dmod->shape.comp[c].real.s[s].f[1];
		if (dmod->shape.comp[c].real.f[f1].act &&
			dmod->shape.comp[c].real.f[f2].act    ) {
			v1 = dmod->shape.comp[c].real.s[s].v[0];
			v2 = dmod->shape.comp[c].real.s[s].v[1];
			len = dev_distance(dmod->shape.comp[c].real.v[v1].x,
							   dmod->shape.comp[c].real.v[v2].x);
			for (ilaw=0; ilaw<dmod->photo.nradlaws; ilaw++) {
				switch (dmod->photo.radtype[ilaw]) {
				case HARMCOSINE_DIFF:
					x = dmod->photo.radar[ilaw].harmcosine.local[c][f1].C.val;
					y = dmod->photo.radar[ilaw].harmcosine.local[c][f2].C.val;
					a[s] = len * fabs(x+y);
					b[s] = len * fabs(x-y);
					break;
				case INHOCOSINE_DIFF:
					x = dmod->photo.radar[ilaw].inhocosine.local[c][f1].C.val;
					y = dmod->photo.radar[ilaw].inhocosine.local[c][f2].C.val;
					a[s] = len * fabs(x+y);
					b[s] = len * fabs(x-y);
					break;
				}
			}
		}
	}
}
__global__ void p_radcdel_finish_krnl(double asum, double bsum){
	/* Single-thread task */
	if (threadIdx.x == 0) {
		if (asum == 0.0)
			printf("penalties_cuda.cu: 'rad_c_del' can't be used with this radar "
					"scattering law\n");
		p_pen = bsum/asum;
	}
}
__global__ void p_radcvar_krnl(struct mod_t *dmod, double *av, double *av2) {
	/* nf-threaded kernel */
	int f = blockIdx.x * blockDim.x + threadIdx.x;
	int c=0, ilaw;
	double x;

	if (f < p_nf) {
		if (dmod->shape.comp[c].real.f[f].act) {
			for (ilaw=0; ilaw<dmod->photo.nradlaws; ilaw++) {
				switch (dmod->photo.radtype[ilaw]) {
				case HARMCOSINE_DIFF:
					x = dmod->photo.radar[ilaw].harmcosine.local[c][f].C.val;
					av[f] = x;
					av2[f] = x*x;
					atomicAdd(&p_ntot, 1);
					break;
				case INHOCOSINE_DIFF:
					x = dmod->photo.radar[ilaw].inhocosine.local[c][f].C.val;
					av[f] = x;
					av2[f] = x*x;
					atomicAdd(&p_ntot, 1);
					break;
				}
			}
		}
	}
}
__global__ void p_radcvar_finish_krnl(double avsum, double av2sum){
	/* Single-threaded task */
	if (threadIdx.x == 0) {
		if (p_ntot == 0)
			printf("penalties_cuda.cu: 'rad_c_var' can't be used with this "
					"radar scattering law\n");
		p_pen = p_ntot * (av2sum/(avsum*avsum)) - 1.0; /* fractional variance */
		if (p_pen < 0.0)		p_pen = 0.0;    /* roundoff error */
	}
}
__global__ void p_noncosine_krnl(struct par_t *dpar, struct mod_t *dmod){
	/* Single-threaded kernel */
	int ilaw, j, n;
	double x, y, ymean, rho, rho_fit, x2sum, y2sum, xysum, slope, intercept,
		   resid2sum;

	if (threadIdx.x == 0) {
		resid2sum = 0.0;
		for (ilaw=0; ilaw<dmod->photo.nradlaws; ilaw++) {
			switch (dmod->photo.radtype[ilaw]) {
			case TABULARLAW:
				/* Using the points with theta < 90 deg, compute the best-fit
				 * slope and intercept and the sum of squared residuals. The
				 * x-values and their mean were computed by the read_mod_cuda
				 * routine, since they never change. For trial rho values that
				 * are less than a tiny positive cutoff value, avoid NaN errors
				 * by continuing the log(rho) curve toward negative rho values
				 * as a straight line that matches the slope at the cutoff point   */
				n = dmod->photo.radar[ilaw].tabular.n;
				atomicAdd(&p_ntot, n);
				ymean = 0.0;
				for (j=0; j<(n-1); j++) {
					rho = dmod->photo.radar[ilaw].tabular.rho[j].val;
					if (rho > RHOCUTOFF)
						dpar->pen.y_noncosine[j] = log10(rho);
					else
						dpar->pen.y_noncosine[j] = log10(RHOCUTOFF) + (rho/RHOCUTOFF - 1)/LN10;
					ymean += dpar->pen.y_noncosine[j];
				}
				ymean /= (n - 1);
				x2sum = y2sum = xysum = 0.0;
				for (j=0; j<(n-1); j++) {
					x = dpar->pen.x_noncosine[j] - dpar->pen.xmean_noncosine;
					y = dpar->pen.y_noncosine[j] - ymean;
					x2sum += x*x;
					y2sum += y*y;
					xysum += x*y;
				}
				slope = xysum/x2sum;
				intercept = ymean - slope * dpar->pen.xmean_noncosine;
				resid2sum += y2sum - xysum*xysum/x2sum;

				/* If the rho value for 90 deg is greater than for the next
				 * smallest angle, add in its squared residual about the fit
				 * value for that smaller angle; otherwise treat this final
				 * point as having zero residual                  */
				rho = dmod->photo.radar[ilaw].tabular.rho[n-1].val;
				if (rho > dmod->photo.radar[ilaw].tabular.rho[n-2].val) {
					rho_fit = slope * dpar->pen.x_noncosine[n-2] + intercept;
					if (rho > RHOCUTOFF)
						y = log10(rho) - rho_fit;
					else
						y = log10(RHOCUTOFF) + (rho/RHOCUTOFF - 1)/LN10 - rho_fit;
					resid2sum += y*y;
				}
				break;
			}
		}

		/* Set the penalty function equal to the mean squared residual about the fit  */
		if (p_ntot == 0)
			printf("penalties_cuda.cu: 'noncosine' can't be used with this radar "
					"scattering law\n");
		p_pen = MAX(resid2sum/p_ntot, 0.0);
	}
}
__global__ void p_nonsmooth_krnl(struct mod_t *dmod, double *a) {
	/* ns-threaded kernel */
	int s = blockIdx.x * blockDim.x + threadIdx.x;
	int c=0;
	double x;

	if (s < p_ns) {
		if (dmod->shape.comp[c].real.s[s].act) {
			x = 1 - dev_dot(dmod->shape.comp[c].real.f
					[dmod->shape.comp[c].real.s[s].f[0] ].n,
					dmod->shape.comp[c].real.f
					[dmod->shape.comp[c].real.s[s].f[1] ].n );

			a[s] = powf(x,4);//*x*x*x;
			atomicAdd(&p_ntot, 1);
		}
	}
}
__global__ void p_nonsmooth_finish_krnl(double sum){
	/* Single-threaded task */
	if (threadIdx.x == 0){
		p_pen = sum/p_ntot;
	}
}
__global__ void p_concavity_krnl(struct mod_t *dmod, double *a) {
	/* ns-threaded kernel */
	int s = blockIdx.x * blockDim.x + threadIdx.x;
	int c=0, f1, f2, v1, v2, v3, j;
	double disp[3], x;

	if (s < p_ns) {
		a[s] = 0.0;
		if (dmod->shape.comp[c].real.s[s].act) {
			f1 = dmod->shape.comp[c].real.s[s].f[0];
			f2 = dmod->shape.comp[c].real.s[s].f[1];
			v1 = dmod->shape.comp[c].real.s[s].v[0];
			v2 = dmod->shape.comp[c].real.s[s].v[1];
			for (j=0; j<=2; j++)
				if ((dmod->shape.comp[c].real.f[f1].v[j] != v1) &&
					(dmod->shape.comp[c].real.f[f1].v[j] != v2)    )
					v3 = dmod->shape.comp[c].real.f[f1].v[j];
			for (j=0; j<=2; j++)
				disp[j] = dmod->shape.comp[c].real.v[v3].x[j] -
						  dmod->shape.comp[c].real.v[v1].x[j];
			if (dev_dot(disp, dmod->shape.comp[c].real.f[f2].n) > 0.0) {
				x = 1 - dev_dot(dmod->shape.comp[c].real.f[f1].n,
						dmod->shape.comp[c].real.f[f2].n);
				a[s] = x*x;
			}
			atomicAdd(&p_ntot, 1);
		}
	}
}
__global__ void p_concavity_finish_krnl(double sum){
	/* Single-thread task */
	if (threadIdx.x == 0) {
		p_pen = sum/p_ntot;
	}
}
__global__ void p_rdev_get_radius_krnl(struct mod_t *dmod) {
	/* Single-threaded kernel */
	if (threadIdx.x == 0) {
		p_volume = dmod->shape.volume;
		p_r_eff = pow( 3*p_volume/(4*PIE), 1.0/3.0);
	}
}
__global__ void p_rdev_vertex_krnl(struct mod_t *dmod, double *varr) {
	/* nv-threaded kernel */
	int v = blockIdx.x * blockDim.x + threadIdx.x;
	int c=0, j;
	double scale, x, y;
	if (v < p_nv) {
		if (dmod->shape.comp[c].real.v[v].act) {
			scale = 0.0;
			for (j=0; j<=2; j++) {
				y = dmod->shape.comp[c].real.v[v].u[j] *
					dmod->shape.comp[c].real.scalefactor[j].val;
				scale += y*y;
			}
			scale = sqrt(scale);
			x = scale * dmod->shape.comp[c].real.v[v].r.val / p_r_eff;
			varr[v] = x*x;
			atomicAdd(&p_ntot, 1);
		}
	}
	__syncthreads();

	if (threadIdx.x == 0 && p_ntot == 0)
		printf("penalties_cuda.cu: need at least one vertex component for 'rdev'\n");
}
__global__ void p_rdev_pen_krnl(double sum) {
	/* Single-threaded kernel */
	if (threadIdx.x == 0)
		p_pen = sum/p_ntot;
}
__global__ void p_maxrdev_vertex_krnl(struct mod_t *dmod, double *varr) {
	/* nv-threaded kernel */
	int v = blockIdx.x * blockDim.x + threadIdx.x;
	int c=0, j;
	double scale, x, y;

	if (v < p_nv) {
		if (dmod->shape.comp[c].real.v[v].act) {
			scale = 0.0;
			for (j=0; j<=2; j++) {
				y = dmod->shape.comp[c].real.v[v].u[j] *
					dmod->shape.comp[c].real.scalefactor[j].val;
				scale += y*y;
			}
			scale = sqrt(scale);
			x = scale * dmod->shape.comp[c].real.v[v].r.val / p_r_eff;
			x *= x;
			varr[v] = x;
		}
	}
}
__global__ void p_maxrdev_finish_krnl(double sum) {
	/* Single-threaded kernel */
	if (threadIdx.x == 0)
		p_pen = sum;
}
__global__ void p_maxellipdev_inertia_krnl(struct mod_t *dmod) {
	/* Single-threaded kernel */
	double radius[3], /*pmoment[3],*/ scale;
	int j;

	if (threadIdx.x == 0) {
		if (!p_got_pa) {
			dev_diag_inertia(dmod->shape.inertia);//, pmoment);
			p_got_pa = 1;
		}
		/* Given a unit-density ellipsoid with volume V_ell and axis radii a,
		 * b, c along x, y, and z, respectively, the moment of inertia about
		 * the x-axis is (V_ell/5)*(b^2 + c^2), and similarly for the other two
		 * axes. Hence the 3 parameters below are the 3 radii of an ellipsoid
		 * whose principal moments of inertia are the same as our model's. */
		radius[0] = sqrt( (5/p_volume) * (-pmoment[0] + pmoment[1] + pmoment[2]) / 2 );
		radius[1] = sqrt( (5/p_volume) * ( pmoment[0] - pmoment[1] + pmoment[2]) / 2 );
		radius[2] = sqrt( (5/p_volume) * ( pmoment[0] + pmoment[1] - pmoment[2]) / 2 );

		/* Take those inertia ellipsoid radii and multiply them by the cube
		 * root of ( V_model / V_ell ): These are the DEEVE radii.  */
		scale = pow( 3*p_volume/(4*PIE*radius[0]*radius[1]*radius[2]), 1.0/3.0);
		for (j=0; j<=2; j++)
			p_DEEVE_radius[j] = scale*radius[j];
	}
}
__global__ void p_maxellipsdev_vertex_krnl(struct mod_t *dmod, double *varr) {
	/* nv-threaded kernel */
	int v = blockIdx.x * blockDim.x + threadIdx.x;
	int c=0, j;
	double vcoord[3], r, theta, phi, xtemp, ytemp, ztemp, x, r_DEEVE;

	if (v < p_nv) {
		/*  Loop through this component's vertices to get the maximum deviation  */
		if (dmod->shape.comp[c].real.v[v].act) {
			/* Transform this vertex's Cartesian body-fixed coordinates to
			 * spherical principal-axis coordinates                   */
			for (j=0; j<=2; j++)
				vcoord[j] = dmod->shape.comp[c].real.v[v].x[j];
			dev_cotrans2(vcoord, p_ap, vcoord, -1);
			r = dev_vecnorm(vcoord);
			theta = atan2(sqrt(vcoord[0]*vcoord[0] + vcoord[1]*vcoord[1]), vcoord[2]);
			phi = atan2(vcoord[1], vcoord[0]);

			/* For these angular coordinates theta and phi, compute the radial
			 * coordinate r_DEEVE of the point that lies on the DEEVE  */
			xtemp = sin(theta)*cos(phi)/p_DEEVE_radius[0];    /*  (x/a)/r  */
			ytemp = sin(theta)*sin(phi)/p_DEEVE_radius[1];    /*  (y/b)/r  */
			ztemp = cos(theta)/p_DEEVE_radius[2];             /*  (z/c)/r  */
			r_DEEVE = 1/sqrt(xtemp*xtemp + ytemp*ytemp + ztemp*ztemp);

			/* Compute the difference between r and r_DEEVE, expressed as a
			 * fraction of the model's effective radius, and then square it */
			x = (r - r_DEEVE)/p_r_eff;
			x *= x;
			varr[v] = x;
		}
	}
}
__global__ void p_comdev_krnl(struct mod_t *dmod) {
	/* Single-threaded kernel */
	int k;
	if (threadIdx.x == 0) {
		for (k=0; k<=2; k++){
			p_pen += dmod->shape.com[k]*dmod->shape.com[k];
//			printf("dmod->shape.com[%i]=%3.6g\n", k, dmod->shape.com[k]);
		}
		p_pen /= (p_r_eff*p_r_eff);
	}
}
__global__ void p_inertiadev_krnl(struct mod_t *dmod) {
	/* Single-threaded kernel */
	double a, b, adotb;
	int j, k;

	if (threadIdx.x == 0) {
		a = b = adotb = 0.0;
		for (j=0; j<=2; j++) {
			a += dmod->spin.inertia[j].val*dmod->spin.inertia[j].val;
			adotb += dmod->spin.inertia[j].val*dmod->shape.inertia[j][j];
			for (k=0; k<=2; k++) {
				b += dmod->shape.inertia[j][k]*dmod->shape.inertia[j][k];
			}
		}
		p_pen = 1 - adotb/sqrt(a*b);
	}
}
__global__ void p_inertiadev_uni_krnl(struct mod_t *dmod) {
	/* Single-threaded kernel */
	int j, k;
	double /*pmoment[3], */a, b, adotb;
	if (threadIdx.x == 0) {
		if (!p_got_pa) {
			/* Diagonalize inertia tensor to get pmoments, the principal
			 * moments of inertia, and ap, the transformation matrix taking us
			 * from principal-axis to body-fixed coordinates              */
			dev_diag_inertia(dmod->shape.inertia);//, pmoment);
			p_got_pa = 1;
		}
		a = b = adotb = 0.0;
		for (j=0; j<=2; j++) {
			a += pmoment[j]*pmoment[j];
			adotb += pmoment[j]*dmod->shape.inertia[j][j];
			for (k=0; k<=2; k++) {
				b += dmod->shape.inertia[j][k]*dmod->shape.inertia[j][k];
			}
		}
		p_pen = 1 - adotb/sqrt(a*b);
	}
}
__global__ void p_pa3tilt_krnl(struct mod_t *dmod) {
	/* Single-threaded kernel */

	if (threadIdx.x == 0) {
		if (!p_got_pa) {
			/* Diagonalize the inertia tensor to get pmoments, the principal
			 * moments of inertia, and ap, the transformation matrix taking us
			 * from principal-axis to body-fixed coordinates                  */
			dev_diag_inertia(dmod->shape.inertia);//, pmoment);
			p_got_pa = 1;
		}
		/*  ap[2][2] = cos(angle between PA3 and the body-fixed z-axis)  */
		p_pen = max(0.0, (1 - p_ap[2][2]*p_ap[2][2]));

	}
}
__global__ void p_nonpa_krnl(struct mod_t *dmod) {
	/* Single-threaded kernel */

	if (threadIdx.x == 0) {
		/* The 0.01 term below avoids biaxial inertia ellipsoids  */
		p_pen =  max((dmod->spin.inertia[0].val/dmod->spin.inertia[2].val) - 1,
				(dmod->spin.inertia[1].val/dmod->spin.inertia[2].val) - 1 )  + 0.01;
		if (p_pen < 0.0)		p_pen = 0.0;
	}
}
__global__ void p_nonpa_uni_krnl(struct mod_t *dmod) {
	/* Single-threaded kernel */

	if (threadIdx.x == 0) {
		if (!p_got_pa) {
			/* Diagonalize inertia tensor to get pmoments, the principal
			 * moments of inertia, and ap, the transformation matrix taking us
			 * from principal-axis to body-fixed coordinates                */
			dev_diag_inertia(dmod->shape.inertia);//, pmoment);
			p_got_pa = 1;
		}
		/* The 0.01 term below avoids biaxial inertia ellipsoids */
		p_pen = max((pmoment[0]/pmoment[2]) - 1, (pmoment[1]/pmoment[2]) - 1 )
												  + 0.01;
		if (p_pen < 0.0)	p_pen = 0.0;
	}
}
__global__ void p_euleroffs_krnl(struct dat_t *ddat) {
	/* Single-threaded kernel */
	int k, j;

	if (threadIdx.x == 0) {
		p_pen = 0.0;
		p_ntot = 0;
		for (k=0; k<ddat->nsets; k++)
			for (j=0; j<=2; j++) {
				p_pen += ddat->set[k].angleoff[j].val*ddat->set[k].angleoff[j].val;
				p_pen += ddat->set[k].omegaoff[j].val*ddat->set[k].omegaoff[j].val;
				p_ntot += 2;
			}
		p_pen *= R2D*R2D/p_ntot;
	}
}
__global__ void p_flattening_krnl(struct mod_t *dmod) {
	/* Single-threaded kernel */
	double b_over_c, x;

	if (threadIdx.x == 0) {
		if (!p_got_pa) {
			/* Diagonalize the inertia tensor to get pmoments, the principal
			 * moments of inertia, and ap, the transformation matrix taking us
			 * from principal-axis to body-fixed coordinates               */
			dev_diag_inertia(dmod->shape.inertia);//, pmoment);
			p_got_pa = 1;
		}
		b_over_c = sqrt(MIN((-pmoment[0] + pmoment[1] + pmoment[2]),
							 (pmoment[0] - pmoment[1] + pmoment[2]))
						   / (pmoment[0] + pmoment[1] - pmoment[2]));
		if (b_over_c > 1.0) {
			x = b_over_c - 1;
			p_pen = x*x*x*x;
		} else
			p_pen = 0.0;
	}
}
__global__ void p_bifur_gotpa_krnl(struct mod_t *dmod) {
	/* Single-threaded kernel */

	if (threadIdx.x == 0) {
		if (!p_got_pa) {
			/* Diagonalize the inertia tensor to get pmoments, the principal
			 * moments of inertia, and ap, the transformation matrix taking us
			 * from principal-axis to body-fixed coordinates                   */
			dev_diag_inertia(dmod->shape.inertia);//, pmoment);
			p_got_pa = 1;
		}

		/* Figure out which axis corresponds to the smallest principal moment  */
		if (pmoment[0] <= pmoment[1] && pmoment[0] <= pmoment[2])
			p_jmax = 0;
		else if (pmoment[1] <= pmoment[0] && pmoment[1] <= pmoment[2])
			p_jmax = 1;
		else
			p_jmax = 2;
		p_j1 = (p_jmax + 1) % 3;
		p_j2 = (p_jmax + 2) % 3;
	}
}
__global__ void p_bifur_1st_vertex_krnl(struct mod_t *dmod, double *varr) {
	/* nv-threaded kernel that assembles an array of values. After completion,
	 * a parallel reduction is run to find min and max */
	int v = blockIdx.x * blockDim.x + threadIdx.x;
	int c=0;

	if (v < p_nv) {
		varr[v] = dmod->shape.comp[c].real.v[v].x[p_jmax];
//		atomicMinf(&p_min_extent, temp);
//		atomicMaxf(&p_max_extent, temp);
	}
}
__global__ void p_bifur_init_krnl(double min_extent, double max_extent) {
	/* Single-threaded kernel */
	int k;
	if (threadIdx.x == 0) {
		p_axis_increment = (max_extent - min_extent + SMALLVAL) / NBIFURCATION;
		for (k=0; k<NZONES; k++)
			p_sumrho2[k] = p_nrho2[k] = 0.0;
	}
}
__global__ void p_bifur_2nd_vertex_krnl(struct mod_t *dmod, double min_extent,
		double max_extent) {
	/* nv-threaded kernel */
	int v = blockIdx.x * blockDim.x + threadIdx.x;
	int c=0, k, k1, k2;
	double rho2, x, y, w1, w2, temp;

	if (v < p_nv) {
		if (dmod->shape.comp[c].real.v[v].act) {
			rho2 = dmod->shape.comp[c].real.v[v].x[p_j1] * dmod->shape.comp[c].real.v[v].x[p_j1] +
				   dmod->shape.comp[c].real.v[v].x[p_j2] * dmod->shape.comp[c].real.v[v].x[p_j2];
			x = (dmod->shape.comp[c].real.v[v].x[p_jmax] - p_min_extent)/p_axis_increment;
			k = (int) floor(x);
			y = x - k;
			if (k == 0) {
				k1 = k2 = 0;
				w1 = 0.0;
				w2 = 1.0;
			} else if (k == NZONES) {
				k1 = k2 = NZONES - 1;
				w1 = 1.0;
				w2 = 0.0;
			} else {
				k1 = k - 1;
				k2 = k;
				w1 = 1 - pow(y, 6);
				w2 = 1 - pow(1-y, 6);
			}
			temp = w1*rho2;
			atomicAdd(&p_sumrho2[k1], __double2float_rn(temp));
			temp = w2*rho2;
			atomicAdd(&p_sumrho2[k2], __double2float_rn(temp));
			temp = w1;
			atomicAdd(&p_nrho2[k1], __double2float_rn(temp));
			temp = w2;
			atomicAdd(&p_nrho2[k2], __double2float_rn(temp));
		}
	}
}
__global__ void p_bifur_meanrho2_krnl() {
	/* NZONES-threaded kernel */
	int k = blockIdx.x * blockDim.x + threadIdx.x;

	if (k < NZONES)
		p_meanrho2[k] = p_sumrho2[k]/p_nrho2[k];
}
__global__ void p_bifur_deepest_krnl() {
	/* NZONES-1-threaded kernel */
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	int k1, k2;
	float temp;

	if (k < (NZONES-1)) {
		for (k1=0; k1<k; k1++)
			if (p_meanrho2[k1] > p_meanrho2[k])
				for (k2=k+1; k2<NZONES; k2++)
					if (p_meanrho2[k2] > p_meanrho2[k])
						temp = min(p_deepest_minimum,
								p_meanrho2[k]/((p_meanrho2[k1] + p_meanrho2[k2])/2));
		atomicMin64(&p_deepest_minimum, temp);
	}
	__syncthreads();

	/* Single-thread task */
	if (k == 0) {
		/* Set penalty equal to (the reciprocal of this fractional minimum, minus 1)^2  */
		if (p_deepest_minimum < 1.0)
			p_pen = (1/p_deepest_minimum - 1)*(1/p_deepest_minimum - 1);
		else
			p_pen = 0.0;
	}
}
__global__ void p_impulse_krnl(struct mod_t *dmod) {
	/* Single-threaded kernel */
	int n, j;

	if (threadIdx.x == 0) {
		p_pen = 0.0;
		for (n=0; n<dmod->spin.n_impulse; n++)
			for (j=0; j<=2; j++)
				p_pen += dmod->spin.impulse[n][j].val * dmod->spin.impulse[n][j].val;
		p_pen *= R2D*R2D/(3 * dmod->spin.n_impulse);
	}
}
__global__ void p_final_krnl(struct par_t *dpar, int i, unsigned char showstate) {
	/* Single-threaded kernel */
	if (threadIdx.x == 0) {
		if (dpar->pen.weight[i] >= 0.0)
			dpar->pen.base[i] = p_pen;
		else
			dpar->pen.base[i] = 1.0 / max(p_pen, TINYPEN);

		/* Add this penalty to the penalty sum; if desired, display the penalty
		 * weight and value to the user.  */
		p_sum += fabs(dpar->pen.weight[i])*dpar->pen.base[i];
		if (showstate) {
			printf("# %e = fabs(%13.6e) * %e\n",
					fabs(dpar->pen.weight[i])*dpar->pen.base[i],
					dpar->pen.weight[i], dpar->pen.base[i]);
		}
	}
}

__host__ double penalties_gpu(struct par_t *dpar, struct mod_t *dmod,
		struct dat_t *ddat, unsigned char spar_showstate)
{
	int i, ntot;
	double sum=0.0, *a, *b, *absum, *av, *av2, *varr, out, min, max;
	char name[80];

	int pen_n, pen_type, ns, nf, nv;
	unsigned char shape_type;
	dim3 BLKs, BLKf, BLKv,THD;
	THD.x = maxThreadsPerBlock;

	gpuErrchk(cudaSetDevice(GPU0));
	/* Get # of penalties from dpar */
	p_get_pen_n_krnl<<<1,1>>>(dpar);
	checkErrorAfterKernelLaunch("p_get_pen_n_krnl (penalties_cuda)");
	gpuErrchk(cudaMemcpyFromSymbol(&pen_n, p_pen_n, sizeof(int),
			0, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpyFromSymbol(&shape_type, p_shape_type,
			sizeof(unsigned char), 0, cudaMemcpyDeviceToHost));

	/* Launch single-thread kernel to find # of sides of real */
	p_get_real_info_krnl<<<1,1>>>(dmod);
	checkErrorAfterKernelLaunch("p_get_real_info_krnl (penalties_cuda)");
	gpuErrchk(cudaMemcpyFromSymbol(&ns, p_ns, sizeof(int),
			0, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpyFromSymbol(&nf, p_nf, sizeof(int),
			0, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpyFromSymbol(&nv, p_nv, sizeof(int),
			0, cudaMemcpyDeviceToHost));

	/* Calculate launch parameters */
	BLKs.x = floor((THD.x - 1 + ns)/THD.x);
	BLKf.x = floor((THD.x - 1 + nf)/THD.x);
	BLKv.x = floor((THD.x - 1 + nv)/THD.x);

	gpuErrchk(cudaMalloc((void**)&a, 	sizeof(double) * ns));
	gpuErrchk(cudaMalloc((void**)&b, 	sizeof(double) * ns));
	gpuErrchk(cudaMalloc((void**)&av, 	sizeof(double) * nf));
	gpuErrchk(cudaMalloc((void**)&av2, 	sizeof(double) * nf));
	gpuErrchk(cudaMalloc((void**)&varr,	sizeof(double) * nv));
	absum = (double *) malloc(2*sizeof(double));

	/* G thru penalties & calculate each contribution to penalty-function sum */
	for (i=1; i<=pen_n; i++) {
		cudaMemset(a, 0, sizeof(double)*ns);

		/* Single-threaded kernel to get penalty type */
		p_get_pen_type_krnl<<<1,1>>>(dpar, dmod, i);
		checkErrorAfterKernelLaunch("p_get_pen_type_krnl (penalties_cuda)");
		gpuErrchk(cudaMemcpyFromSymbol(&pen_type, p_pen_type, sizeof(int),
				0, cudaMemcpyDeviceToHost));

		/* Note that each of these switch cases had originally a component loop.
		 * The first version of shape-cuda-v1.0 is capable of only single-
		 * component models	 */
		switch (pen_type) {
		case OPTALBDEL:
			/* pen = (weighted mean over model "sides" [edges] of |albedo difference|
			 * for the two facets sharing that side)/(weighted mean over model
			 *  "sides" of albedo sum for those facets) where the weighting
			 *  factor is the length of the side  */
			/* Penalty is calculated by summing up a and b over all sides, then
			 * pen = b/a.  So the first kernel calculates a and b for each side
			 * (as doubles) then a parallel reduction gets the sum of all. Then
			 * we calculate b/a.			 */
			strcpy( name, "optalbdel");

			/* Launch the ns-threaded optalbdel kernel */
			p_optalbdel_krnl<<<BLKs,THD>>>(dmod, a, b);
			checkErrorAfterKernelLaunch("p_optalbdel_krnl");

			/* Now do parallel reduction on arrays a and b */
			sum_2_double_arrays(a, b, absum, ns);

			/* Now finish the pen(alty) calculation pen = b/a */
			p_optalbdel_finish_krnl<<<1,1>>>(absum[0], absum[1]);
			checkErrorAfterKernelLaunch("p_optalbdel_finish_krnl");
			break;
		case OPTALBVAR:
			/*  pen = (facet albedo variance) / (mean facet albedo)^2  */
			strcpy( name, "optalbvar");

			/* Launch the nf-threaded optalbvar kernel */
			p_optalbvar_krnl<<<BLKf,THD>>>(dmod, av, av2);
			checkErrorAfterKernelLaunch("p_optalbvar_krnl (penalties_cuda)");

			/* Now do parallel reduction on arrays av and av2 */
			sum_2_double_arrays(av, av2, absum, nf);

			/* Now finish the pen(alty) calculation */
			p_optalbvar_finish_krnl<<<1,1>>>(absum[0], absum[1]);
			checkErrorAfterKernelLaunch("p_optalbdel_finish_krnl");
			break;
		case THETADEL:
			/* pen = (weighted mean over model "sides" [edges] of
			 *  |slope angle difference| for the two facets sharing that side)
			 *  / (weighted mean over model "sides" of slope angle sum for
			 *  those facets) where the weighting factor is the length of the
			 *  side. More precisely, theta is the mean slope angle for
			 *  intrafacet topographic roughness, so "mean facet slope angle"
			 *  is actually "mean over all model facets of the mean slope angle
			 *  for roughness within each facet," and similarly for the
			 *  variance.                                               */
			strcpy( name, "thetadel");

			/* Launch the ns-threaded thetadel kernel */
			p_thetadel_krnl<<<BLKs,THD>>>(dmod, a, b);
			checkErrorAfterKernelLaunch("p_thetadel_krnl (penalties_cuda)");

			/* Now do parallel reduction on arrays a and b */
			sum_2_double_arrays(a, b, absum, ns);

			/* Now finish penalty calculation */
			p_thetadel_finish_krnl<<<1,1>>>(absum[0], absum[1]);
			checkErrorAfterKernelLaunch("p_thetadel_finish_krnl");
			break;
		case THETAVAR:
			/* pen = (facet slope angle variance) / (mean facet slope angle)^2
			 * More precisely, theta is the mean slope angle for intrafacet
			 * topographic roughness, so "mean facet slope angle" is actually
			 * "mean over all model facets of the mean slope angle for
			 * roughness within each facet," and similarly for the variance */
			strcpy( name, "thetavar");

			/* Launch the nf-threaded thetavar kernel */
			p_thetavar_krnl<<<BLKf,THD>>>(dmod, av, av2);
			checkErrorAfterKernelLaunch("p_thetadel_krnl (penalties_cuda)");

			/* Parallel reduction on av and av2 to get sums */
			sum_2_double_arrays(av, av2, absum, nf);

			/* Now finish penalty calculation */
			p_thetavar_finish_krnl<<<1,1>>>(absum[0], absum[1]);
			checkErrorAfterKernelLaunch("p_thetavar_finish_krnl");
			break;
		case RADALBDEL:
			/* pen = (weighted mean over model "sides" [edges] of |albedo
			 * difference | for the two facets sharing that side)/(weighted
			 * mean over model "sides" of albedo sum for those facets) where
			 * the weighting factor is the length of the side  */
			strcpy( name, "radalbdel");

			/* Launch the ns-threaded radalbdel kernel */
			p_radalbdel_krnl<<<BLKs,THD>>>(dmod, a, b);
			checkErrorAfterKernelLaunch("p_radalbdel_krnl (penalties_cuda)");

			/* Parallel reduction on a and b to get sums */
			sum_2_double_arrays(a, b, absum, ns);

			/* Finish penalty calculations */
			p_radalbdel_finish_krnl<<<1,1>>>(absum[0], absum[1]);
			checkErrorAfterKernelLaunch("p_radalbdel_finish_krnl");
			break;
		case RADALBVAR:
			/* pen = (facet albedo variance) / (mean facet albedo)^2  */
			strcpy( name, "radalbvar");

			/* Launch the nf-threaded radalbvar kernel */
			p_radalbvar_krnl<<<BLKf,THD>>>(dmod, av, av2);
			checkErrorAfterKernelLaunch("p_radalbvar_krnl (penalties_cuda)");

			/* Paralle reduction on av and av2 to get sums */
			sum_2_double_arrays(av, av2, absum, nf);

			/* Finish penalty calculations */
			p_radalbvar_finish_krnl<<<1,1>>>(absum[0], absum[1]);
			checkErrorAfterKernelLaunch("p_radalbvar_finish_krnl");
			break;
		case RAD_C_DEL:
			/* pen = (weighted mean over model "sides" [edges] of |C difference|
			 * for the two facets sharing that side)/(weighted mean over model
			 * "sides" of C sum for those facets) where the weighting factor is
			 * the length of the side  */
			strcpy( name, "rad_c_del");

			/* Launch the ns-threaded rad_c_del kernel */
			p_radcdel_krnl<<<BLKs,THD>>>(dmod, a, b);
			checkErrorAfterKernelLaunch("p_radcdel_krnl (penalties_cuda)");

			/* Parallel reduction to get sums */
			sum_2_double_arrays(a, b, absum, ns);

			/* Finish penalty calculations */
			p_radcdel_finish_krnl<<<1,1>>>(absum[0], absum[1]);
			checkErrorAfterKernelLaunch("p_radcdel_finish_krnl");
			break;
		case RAD_C_VAR:
			/* pen = (facet C variance) / (mean facet C)^2  */
			strcpy( name, "rad_c_var");

			/* Launch the nf-threaded radalbvar kernel */
			p_radcvar_krnl<<<BLKf,THD>>>(dmod, av, av2);
			checkErrorAfterKernelLaunch("p_radcvar_krnl (penalties_cuda)");

			/* Parallel reduction to get sums */
			sum_2_double_arrays(av, av2, absum, nf);

			/* Finish penalty calculations */
			p_radcvar_finish_krnl<<<1,1>>>(absum[0], absum[1]);
			checkErrorAfterKernelLaunch("p_radcvar_finish_krnl");
			break;
		case NONCOSINE:
			/* pen = mean squared residual about fit to cosine scattering law
			 * (unweighted linear fit of log rho vs. log cos theta)
			 * The fit is done over the range 0 <= theta < 90 deg. The point at
			 * 90 deg must be treated separately, since log(cos(90)) is
			 * undefined. If it is less than the preceding rho value then the
			 * final point contributes nothing to the penalty function; if it
			 * is greater than the preceding value then the final point's
			 * squared residual is computed about the value predicted by the
			 * fit for the preceding point.                                   */
			strcpy( name, "noncosine");

			/* This is a single-thread kernel */
			p_noncosine_krnl<<<1,1>>>(dpar, dmod);
			checkErrorAfterKernelLaunch("p_noncosine_krnl (penalties_cuda)");

			break;
		case NONSMOOTH:
			/* pen = mean over all "sides" (edges) of [1 - cos(theta)]^4 where
			 * theta = angle between the normals to the two facets adjoining a
			 * given side.
			 * Even an ellipsoid model has a nonzero "nonsmooth" penalty, but
			 * the high power (4) ensures the penalty will be much greater if
			 * facet-scale topography is present. Since this penalty depends on
			 * facet-scale roughness, it is smaller (for a given shape) when
			 * the model is subdivided into a larger number of smaller facets.
			 * To be precise, when realizing an ellipsoid or harmonic model as
			 * a vertex model, the nonsmooth penalty is roughly proportional to
			 * 1/(number of vertices)^4. Must adjust the "nonsmooth" penalty
			 * weight accordingly.                    */
			strcpy( name, "nonsmooth");

			/* Launch the ns-threaded nonsmooth kernel */
			p_nonsmooth_krnl<<<BLKs,THD>>>(dmod, a);
			checkErrorAfterKernelLaunch("p_nonsmooth_krnl (penalties_cuda)");

			/* Parallel reduction on a to get sum */
			out = sum_double_array(a, ns);

			/* Finish penalty calculations */
			p_nonsmooth_finish_krnl<<<1,1>>>(out);
			checkErrorAfterKernelLaunch("p_noncosine_finish_krnl");
			break;
		case CONCAVITY:
			/* pen = mean over all "sides" (edges) of the following quantity:
			 *               [1 - cos(theta)]^2  if side is "concave"
			 *                0           if side is "convex"
			 *
			 * where theta = angle between the normals to the two facets
			 * adjoining a given side. To determine whether or not a given side
			 * represents a concavity: Look at the two facets adjoining that
			 * side; construct a vector from one end of the side to the far
			 * vertex of facet 1; and take its dot product with the normal to
			 * facet 2. If the dot product is positive, these two facets are
			 * tilted relative to each other in the concave sense. Note that
			 * while ellipsoids are convex-definite, the vertex realization of
			 * an ellipsoid model CAN have some shallow concavities. Since this
			 * penalty depends on facet-scale concavities, it is smaller (for a
			 * given shape) when the model is subdivided into a larger number
			 * of smaller facets. To be precise, when realizing an ellipsoid or
			 * harmonic model as a vertex model, the concavity penalty is
			 * roughly proportional to 1/(number of vertices)^2. Must adjust
			 * the "concavity" penalty weight accordingly.               */
			strcpy( name, "concavity");
			/* Launch the ns-threaded concavity kernel */
			p_concavity_krnl<<<BLKs,THD>>>(dmod, a);
			checkErrorAfterKernelLaunch("p_concavity_krnl (penalties_cuda)");

			/* Parallel reduction */
			out = sum_double_array(a, ns);

			p_concavity_finish_krnl<<<1,1>>>(out);
			checkErrorAfterKernelLaunch("p_concavity_finish_krnl");
			break;
		case RDEV:
			/* pen = mean squared vertex deviation length (where each vertex
			 * deviation is expressed as a fraction of the model's effective
			 * radius)  */
			strcpy( name, "rdev");

			/* Single-thread kernel to get the model's effective radius  */
			p_rdev_get_radius_krnl<<<1,1>>>(dmod);
			checkErrorAfterKernelLaunch("p_rdev_get_radius_krnl (penalties_cuda)");

			if (shape_type == VERTEX) {
				/* Launch nv-threaded kernel oop through vertices to build up mean squared deviation  */
				p_rdev_vertex_krnl<<<BLKv,THD>>>(dmod, varr);
				checkErrorAfterKernelLaunch("p_rdev_vertex_krnl (penalties_cuda)");
				gpuErrchk(cudaMemcpyFromSymbol(&ntot, p_ntot, sizeof(int),
						0, cudaMemcpyDeviceToHost));

				/* Parallel reduction */
				out = sum_double_array(varr, nv);

				/* Single-thread kernel to calculate pen */
				p_rdev_pen_krnl<<<1,1>>>(out);
				checkErrorAfterKernelLaunch("p_rdev_pen_krnl (penalties_cuda)");
			}
			break;
		case MAXRDEV:
			/* pen = maximum squared vertex deviation length (where each vertex
			 * deviation is expressed as a fraction of the model's effective
			 * radius)  */
			strcpy( name, "maxrdev");

			/* Single-thread kernel to get the model's effective radius  */
			p_rdev_get_radius_krnl<<<1,1>>>(dmod);
			checkErrorAfterKernelLaunch("p_maxrdev_get_radius_krnl (penalties_cuda)");

			/* Launch nv-threaded kernel to find maximum deviation for each
			 * vertex.		 */
			if (shape_type == VERTEX) {
				/*  Loop through vertices to find maximum deviation (nv-threaded) */
				p_maxrdev_vertex_krnl<<<BLKv,THD>>>(dmod, varr);
				checkErrorAfterKernelLaunch("p_maxrdev_vertex_krnl (penalties_cuda)");

				/* Do parallel reduction to find the maximum */
				out = find_max_in_double_array(varr, nv);

				/* Finish up penalty calculation */
				p_maxrdev_finish_krnl<<<1,1>>>(out);
				checkErrorAfterKernelLaunch("p_maxrdev_finish_krnl");
			}
			break;
		case MAXELLIPDEV:
			/* pen = maximum squared deviation from the model's DEEVE (where
			 * each deviation is expressed as a fraction of the model's effective radius)*/
			strcpy( name, "maxellipdev");

			/* Single-thread kernel to get the model's effective radius  */
			p_rdev_get_radius_krnl<<<1,1>>>(dmod);
			checkErrorAfterKernelLaunch("p_maxrdev_get_radius_krnl (penalties_cuda)");

			/* Diagonalize the inertia tensor to get pmoments, the principal
			 * moments of inertia, and ap, the transformation matrix taking us
			 * from principal-axis to body-fixed coordinates                    */
			p_maxellipdev_inertia_krnl<<<1,1>>>(dmod);
			checkErrorAfterKernelLaunch("p_maxellipdev_inertia_krnl (penalties_cuda)");

			/* Loop thru components & compute each vertex's deviation from DEEVE,
			 * expressed as a fraction of the model's effective radius */
			p_maxellipsdev_vertex_krnl<<<BLKv,THD>>>(dmod, varr);
			checkErrorAfterKernelLaunch("p_maxellipsdev_vertex_krnl (penalties_cuda)");

			/* Do parallel reduction to find the maximum */
			out = find_max_in_double_array(varr, nv);
			/* Finish up penalty calculation (yes, this is the right kernel!) */
			p_maxrdev_finish_krnl<<<1,1>>>(out);
			checkErrorAfterKernelLaunch("p_maxrdev_finish_krnl");
			break;
		case VOLUME:
			/* pen = model's total volume (km^3)  */
			strcpy( name, "volume");
			p_volume_krnl<<<1,1>>>(dmod);
			checkErrorAfterKernelLaunch("p_volume_krnl (penalties_cuda)");

			break;
		case COMDEV:
			/* pen = squared length of the center-of-mass displacement divided
			 * by the square of the model's effective radius  */
			strcpy( name, "comdev");

			/* Single-thread kernel to get the model's effective radius  */
			p_rdev_get_radius_krnl<<<1,1>>>(dmod);
			checkErrorAfterKernelLaunch("p_maxrdev_get_radius_krnl (penalties_cuda)");

			/* Compute the squared magnitude of the model's COM displacement  */
			p_comdev_krnl<<<1,1>>>(dmod);
			checkErrorAfterKernelLaunch("p_comdev_krnl (penalties_cuda)");

			break;
		case INERTIADEV:
			/* pen = 1 - (dot product of two vectors described below):
			 * Create a vector using the diagonal elements of inertia tensor
			 * (in body coordinates) as the three components; then divide
			 * this vector by the square root of the sum of squares of all nine
			 * tensor elements. (The latter quantity appears as sqrt(b) below,
			 * and is invariant under rotation - in particular, under
			 * transformation to principal-axis coordinates.) The resulting
			 * vector has unit length if the inertia tensor is diagonal, but is
			 * shorter otherwise.
			 * Create a second vector - this one certainly a unit vector - by
			 * carrying out the same procedure with the 3 principal moments of
			 * inertia,treating them as the diagonal elements of a 3x3 diagonal
			 * matrix. Here we use the 3 spin.inertia parameters listed in the
			 * "spin" section of the model file, NOT the principal moments we
			 * would obtain by diagonalizing the inertia tensor. Since the
			 * inertia tensor was computed assuming uniform density, these two
			 * sets of principal moments will differ if the density is nonuniform.
			 * The resulting penalty function is zero iff these two vectors are
			 * identical - that is, iff the inertia tensor is diagonal AND the
			 * model density is uniform. It is positive for any other case.
			 * Hence this penalty pushes the model towards uniform density with
			 * principal axes remaining close to the body-coordinate axes.
			 *
			 * Note that this penalty is confined to the range [0, 2].
			 * Note also that the three "spin.inertia" moments are the parameters
			 * used in Euler's equations for evolving the spin state of NPA
			 * rotators. The "inertiadev" penalty, then, is the only link between
			 * the model's principal moments and the model's *shape*.	 */
			strcpy( name, "inertiadev");

			/* Launch single-threaded kernel for this */
			p_inertiadev_krnl<<<1,1>>>(dmod);
			checkErrorAfterKernelLaunch("p_inertiadev_krnl (penalties_cuda)");

			break;
		case INERTIADEV_UNI:
			/* Same as the inertiadev penalty (see above), except that instead of
			 * comparing the inertia tensor (computed assuming uniform density)
			 * to the three "spin.inertia" principal moments, we compare the
			 * inertia tensor to the three diagonal elements of the diagonalized
			 * inertia tensor; this is appropriate for a principal-axis rotator,
			 * since the data can't constrain the spin.inertia principal moments
			 * (i.e., we have no choice but to assume uniform density).	 */
			strcpy( name, "inertiadev_uni");

			/* Launch single-threaded kernel for inertiadev_uni */
			p_inertiadev_uni_krnl<<<1,1>>>(dmod);
			checkErrorAfterKernelLaunch("p_inertiadev_uni_krnl (penalties_cuda)");

			break;
		case PA3TILT:
			/* pen = sin^2 (angle between third principal axis and body-fixed z-axis)
			 * pa3tilt is to be used for a principal-axis rotator instead of the
			 * inertiadev_uni penalty (see above) if we don't care how the first two
			 * principal axes are oriented relative to the body-fixed x and y axes
			 * but we still want to enforce the physical requirement that the third
			 * principal axis be parallel to the spin vector (i.e., to the body-fixed
			 * z-axis). The principal axes are determined assuming uniform density.*/
			strcpy( name, "pa3tilt");

			/* Launch single-threaded kernel for pa3tilt */
			p_pa3tilt_krnl<<<1,1>>>(dmod);
			checkErrorAfterKernelLaunch("p_pa3tilt_krnl (penalties_cuda)");

			break;
		case NONPA:
			/* pen = MAX( 0, (0.01 + fraction by which the largest of the 1st 2
			 * principal moments of inertia exceeds the 3rd moment))
			 * This penalty drives the first two principal moments to be at
			 * least 1% smaller than the third.                     */
			strcpy( name, "nonpa");

			/* Launch single-threaded kernel for nonpa */
			p_nonpa_krnl<<<1,1>>>(dmod);
			checkErrorAfterKernelLaunch("p_nonpa_krnl (penalties_cuda)");

			break;
		case NONPA_UNI:
			/* Same as the nonpa penalty (see above), except that instead of
			 * comparing the 3 spin.inertia principal moments, we take the
			 * inertia tensor (computed assuming uniform density), diagonalize
			 * it, and compare the 3 diagonal elements; this is appropriate for
			 * a principal-axis rotator, since the data can't constrain the
			 * spin.inertia principal moments (i.e., we have no choice but to
			 * assume uniform density).			 */
			strcpy( name, "nonpa_uni");

			/* Launch single-threaded kernel for nonpa_unit */
			p_nonpa_uni_krnl<<<1,1>>>(dmod);
			checkErrorAfterKernelLaunch("p_nonpa_uni_krnl (penalties_cuda)");

			break;
		case EULEROFFS:
			/* pen = mean squared *component* of angle and spin Euler offsets
			 * lumped together, with component values in degrees (angle) and
			 * degrees/day (spin). That is, for nsets datasets, sum 6*nsets
			 * squared component values and then divide by 6*nsets.      */
			strcpy( name, "euleroffs");

			/* Launch single-threaded kernel for pa3tilt */
			p_euleroffs_krnl<<<1,1>>>(ddat);
			checkErrorAfterKernelLaunch("p_euleroffs_krnl (penalties_cuda)");

			break;
		case FLATTENING:
			/* Discourage flattening by setting penalty to (b/c - 1)^4, where b
			 * is the smaller of the 1st 2 DEEVE diameters and c is the 3rd
			 * DEEVE diameter; both diameters are estimated via the inertia
			 * tensor computed under the assumption of uniform density.		 */
			strcpy( name, "flattening");

			/* Launch single-threaded kernel for flattening */
			p_flattening_krnl<<<1,1>>>(dmod);
			checkErrorAfterKernelLaunch("p_flattening_krnl (penalties_cuda)");

			break;
		case BIFURCATION:
			/* Discourage bifurcation along the longest DEEVE diameter - that is,
			 * along the axis corresponding to the smallest principal moment of
			 * inertia, computed under the assumption of uniform density.
			 * Divide this axis into NBIFURCATION equal-width zones. For each active
			 * vertex, compute the squared distance from the axis, then add this
			 * contribution to TWO adjacent zones (except at the model's ends)
			 * according to a 1 - x^6 "response function" whose "base" is two zones
			 * wide. Then compute the mean squared distance S for each zone. This
			 * procedure produces correlated S values for adjacent zones, but it
			 * prevents the penalty function from being sensitive to the exact
			 * positions of vertices that lie near zone boundaries.
			 * Now consider each possible set of three zones k1 < k < k2 for which the
			 * mean squared distance S for zone k is less than that for both k1 and k2.
			 * Find the deepest fractional minimum, that is, the minimum value of ratio
			 * r = S(k) / <mean of S(k1) and S(k2)>.  Set the penalty function equal to
			 * (1/r - 1)^2 (unless r > 1, in which case set it to zero).			 */
			strcpy( name, "bifurcation");

			/* Launch single-threaded kernel to set things up with diagonalizing the
			 * inertia tensor and calculating j1 and j2		 */
			p_bifur_gotpa_krnl<<<1,1>>>(dmod);
			checkErrorAfterKernelLaunch("p_bifur_gotpa_krnl (penalties_cuda)");

			/* Loop through vertices to assemble an array of values, then launch
			 * a parallel reduction to find min and max values (nv-threaded)  */
			p_bifur_1st_vertex_krnl<<<BLKv,THD>>>(dmod, varr);
			checkErrorAfterKernelLaunch("p_bifur_1st_vertex_krnl (penalties_cuda)");
			max = find_max_in_double_array(varr, nv);
			min = find_min_in_double_array(varr, nv);
			/* Launch single-thread kernel to initialize sumrho2[] and to set
			 * axis_increment  */
			p_bifur_init_krnl<<<1,1>>>(min, max);
			checkErrorAfterKernelLaunch("p_bifur_init_vertex_krnl (penalties_cuda)");

			/* Loop over all "active" (exterior) vertices of all model components,
			 * computing the mean value of the squared distance from the chosen axis
			 * in each of NZONES zones along that axis. Except for vertices near
			 * the model's ends, each vertex contributes to TWO adjacent zones
			 * according to a 1 - x^6 "response function" whose "base" is two zones
			 * wide. This implies fractional contributions and hence the "nrho2"
			 * vector is floating-point rather than integer.              */

			/* Launch nv-threaded kernel */
			p_bifur_2nd_vertex_krnl<<<BLKv,THD>>>(dmod, min, max);
			checkErrorAfterKernelLaunch("p_bifur_2nd_vertex_krnl (penalties_cuda)");

			/* Launch an NZONES-threaded kernel to calculate meanrho2[] */
			THD.x = NZONES;
			p_bifur_meanrho2_krnl<<<1,THD>>>();
			checkErrorAfterKernelLaunch("p_bifur_meanrho2_krnl (penalties_cuda)");

			/* Look for the deepest fractional minimum in the mean squared distance
			 * from the longest axis: compare the value for a given zone k to the
			 * mean value for zones k1 and k2 on opposite sides of zone k, where the
			 * values for both zone k1 and zone k2 are greater than for zone k   */

			/* Another NZONES-threaded kernel */
			THD.x = NZONES-1;
			p_bifur_deepest_krnl<<<1,THD>>>();
			checkErrorAfterKernelLaunch("p_bifur_deepest_krnl (penalties_cuda)");
			/* Reset THD.x */
			THD.x = maxThreadsPerBlock;
			break;
		case IMPULSE:
			/* pen = mean squared spin impulse *component* in degrees/day  */
			strcpy( name, "impulse");

			/* Launch single-threaded kernel for impulse */
			p_impulse_krnl<<<1,1>>>(dmod);
			checkErrorAfterKernelLaunch("p_impulse_krnl (penalties_cuda)");

			break;
		default:
			bailout("penalties_cuda.cu: haven't heard for that penalty yet!\n");
		}


		/* Print the penalty names if par->showstate==1.  The penalty weights and
		 * numbers will be printed by the p_final_krnl.  I print the name here
		 * to avoid a memcopy of name[80] to the GPU		 */
		//printf("penalties_gpu showstate: %i\n", showstate);
		if (spar_showstate) {
			printf("# %15s ", name);
		}
		/* A negative penalty weight yields the reciprocal of the usual penalty
		 * function, leading the corresponding model property to be pushed
		 * toward large rather than small values.           */
		/* Single-threaded kernel */
		p_final_krnl<<<1,1>>>(dpar, i, spar_showstate);
		checkErrorAfterKernelLaunch("p_final_krnl (penalties_gpu)");
	}
	gpuErrchk(cudaMemcpyFromSymbol(&sum, p_sum, sizeof(double),
				0, cudaMemcpyDeviceToHost));
	cudaFree(a);
	cudaFree(b);
	cudaFree(av);
	cudaFree(av2);
	cudaFree(varr);
	free(absum);
	return sum;
}

#undef RHOCUTOFF
#undef NBIFURCATION
#undef NZONES
#undef TINYPEN
