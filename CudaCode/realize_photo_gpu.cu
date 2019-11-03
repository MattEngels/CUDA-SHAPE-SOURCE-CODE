/*****************************************************************************************
                                                                          realize_photo.c

Realize the model's radar and optical photometric scattering parameters, including those
set to the '=' state.

Setting radar and optical albedos is more complicated than setting other photometric
parameters, since the "vary_radalb" and "vary_optalb" fitting parameters, which permit
radar and optical albedos to be varied jointly with shape/spin parameters that are being
fit, might be turned on.  The "albedo_mode" parameter to realize_photo determines how
albedos are handled:

    albedo_mode = 0:
        Albedos are not being varied jointly with shape/spin parameters, or else we aren't
        fitting a shape/spin parameter right now; just update the "save" parameters (e.g.,
        R_save) by setting them equal to the corresponding albedo-related parameter values
        (e.g., R) in case joint variation is needed later in the fit

    albedo_mode = 1:
        Albedos are being varied jointly with shape/spin parameters, and we're in the
        process of fitting some shape/spin parameter p (i.e., we're realizing the model
        for a trial value of p); set each radar-albedo-related parameter equal to the
        corresponding "save" value multiplied by "radalb_factor" and set each
        optical-albedo-related parameter equal to the corresponding "save" value
        multiplied by "optalb_factor"

    albedo_mode = 2:
        Albedos are being varied jointly with shape/spin parameters, we've just obtained
        the best-fit value for shape/spin parameter p, and now we need to set the albedos
        to their best-fit values (i.e., to the values which "go with" the best-fit value
        of p); set each radar-albedo-related parameter equal to the corresponding "save"
        value multiplied by "radalb_factor" and set each optical-albedo-related parameter
        equal to the corresponding "save" value multiplied by "optalb_factor," then update
        the "save" parameters by setting them equal to these same products

Modified 2014 February 12 by CM:
    Implement multiple radar and optical scattering laws

Modified 2013 July 15 by CM:
    Bug fix: delete repeated statement in INHOCOSINE_DIFF block (dealing with R)

Modified 2011 September 7 by CM:
    Bug fix: don't vary any parameters with state = 'c'
    Add "harmlambert" and "inholambert" optical scattering laws

Modified 2010 April 27 by CM:
    Added "tabular" radar scattering law

Modified 2009 April 21 by CM:
    Fix a bug in the "checkphotopar" routine

Modified 2009 April 3 by CM:
    Initialize the "badphoto_logfactor" parameter and set its value if
        there are illegal values for photometric parameters
    Add the "checkphotopar" routine

Modified 2007 August 10 by CM:
    Initialize uninitialized variable

Modified 2006 October 1 by CM:
    Add "radalb_factor" "optalb_factor" and "albedo_mode" parameters in
        order to implement the "vary_radalb" and "vary_optalb" parameters

Modified 2006 June 18 by CM:
    Implement user-specified upper and lower limits to photometric
        parameters (rad_R_min, opt_w_max, etc.)

Modified 2005 September 8 by CM:
    Added "harmlommel" "harmhapke" and "harmkaas" optical scattering laws
    Added "harmcosine" radar scattering law

Modified 2005 August 8 by CM:
    Added "inhokaas" optical scattering law

Modified 2005 July 20 by CM:
    Added "gaussian" and "hagfors" and "cosine_qs" and "gauss+cosine" and
        "hagfors+cosine" and "cosine+cosine" and inhomogeneous "inhocosine"
        radar scattering laws

Modified 2005 July 4 by CM:
    Adjusted structure for the "inholommel" optical scattering law
    Enabled "inholommel" and "inhohapke" laws to be used for ellipsoid
        and harmonic model components, not just for vertex components

Modified 2005 February 28 by CM:
    Added checks for radar scattering law parameters
    Added NOLAW case for both optical and radar scattering laws
    Initialize the "badphoto" parameter (flag indicating illegal values
        for photometric parameters) to 0 here instead of in bestfit.c
        so that it can be used for actions other than "fit"
    Added a default case to switch statement, just to be safe

Modified 2004 May 8 by CM:
    Fixed treatment of inhomogeneous Lommel-Seeliger and Hapke laws

Modified 2004 April 29 by CM:
    For Kaasalainen scattering law, switch from weighting factor "c"
        (ranging from 0 to infinity) to "wt" (ranging from 0 to 1)

Modified 2004 March 25 by CM:
    Check for valid parameter values for Hapke optical scattering law.
        Indicate illegal values by setting the badphoto parameter to 1;
        this later will result in a doubled objective function for the
        model.

Modified 2004 February 26 by CM:
    Check for valid parameter values for Lambert, Lommel-Seeliger,
        geometric, and Kaasalainen "Lambert + Lommel-Seeliger"
        optical scattering laws.  Indicate illegal values by setting
        the badphoto parameter to 1; this later will result in a
        doubled objective function for the model.
    As a result, must now pass the model's parameter structure
        to realize_photo
 *****************************************************************************************/

extern "C" {
#include "../shape/head.h"
}

__device__ int dL, dLmax, dn, dnoptlaws, dnradlaws;
__device__ unsigned char dradtype, dopttype;
__device__ void dev_checkphotopar( double parval, double parmin, double parmax, int mode,
		unsigned char *badphoto, double *badphoto_logfactor);

__device__ void dev_checkphotopar( double parval, double parmin, double parmax, int mode,
		unsigned char *badphoto, double *badphoto_logfactor){

	/*  Flag photometric parameter as bad if
           mode = 0:  parval <  parmin  or  parval >  parmax
           mode = 1:  parval <= parmin  or  parval >  parmax
           mode = 2:  parval <  parmin  or  parval >= parmax
           mode = 3:  parval <= parmin  or  parval >= parmax  */

	if (mode < 0 || mode > 3)
		printf("realize_photo.c: checkphotopar mode must be between 0 and 3\n");

	if (mode == 0 || mode == 2) {
		if (parval < parmin) {
			*badphoto = 1;
			*badphoto_logfactor += log(1 + parmin - parval);
		}
	} else {
		if (parval <= parmin) {
			*badphoto = 1;
			*badphoto_logfactor += log(1 + parmin - parval);
		}
	}

	if (mode == 0 || mode == 1) {
		if (parval > parmax) {
			*badphoto = 1;
			*badphoto_logfactor += log(1 + parval - parmax);
		}
	} else {
		if (parval >= parmax) {
			*badphoto = 1;
			*badphoto_logfactor += log(1 + parval - parmax);
		}
	}
}

__global__ void get_photo_types_krnl(struct par_t *dpar, struct mod_t *dmod){
	/* Single-threaded kernel */
	if (threadIdx.x == 0) {
		dpar->badphoto = 0;
		dpar->badphoto_logfactor = 0.0;
		dopttype = dmod->photo.opttype[0];
		dradtype = dmod->photo.radtype[0];
		dnoptlaws = dmod->photo.noptlaws;
		dnradlaws = dmod->photo.nradlaws;
	}
}
__global__ void opt_lommel_krnl(struct par_t *dpar, struct mod_t *dmod,
		double optalb_factor, int albedo_mode, int ilaw){
	/* Single-threaded kernel */
	if (threadIdx.x == 0) {
		if (dmod->photo.optical[ilaw].R.R.state == 'f') {
			if (albedo_mode != 0)
				dmod->photo.optical[ilaw].R.R.val = dmod->photo.optical[ilaw].R.R_save * optalb_factor;
			if (albedo_mode != 1)
				dmod->photo.optical[ilaw].R.R_save = dmod->photo.optical[ilaw].R.R.val;
		}
		dev_checkphotopar( dmod->photo.optical[ilaw].R.R.val,
				dpar->opt_R_min, dpar->opt_R_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
	}
}
__global__ void opt_harmlommel_getL_krnl(struct mod_t *dmod, int ilaw){
	/* Single-treaded kernel */
	if (threadIdx.x == 0)
		dL = dmod->photo.optical[ilaw].harmR.R.nhar;
}
__global__ void opt_harmlommel_set_ab_krnl(struct mod_t *dmod, double optalb_factor,
		int albedo_mode, int ilaw){
	/* L-threaded kernel */
	int m, l = threadIdx.x;
	if (threadIdx.x < dmod->photo.optical[ilaw].harmR.R.nhar)
	{
		if (dmod->photo.optical[ilaw].harmR.R.a[l][0].state == 'f') {
			if (albedo_mode != 0)
				dmod->photo.optical[ilaw].harmR.R.a[l][0].val
				= dmod->photo.optical[ilaw].harmR.R.a_save[l][0] * optalb_factor;
			if (albedo_mode != 1)
				dmod->photo.optical[ilaw].harmR.R.a_save[l][0]
				= dmod->photo.optical[ilaw].harmR.R.a[l][0].val;
		}
		for (m=1; m<=l; m++) {
			if (dmod->photo.optical[ilaw].harmR.R.a[l][m].state == 'f') {
				if (albedo_mode != 0)
					dmod->photo.optical[ilaw].harmR.R.a[l][m].val
					= dmod->photo.optical[ilaw].harmR.R.a_save[l][m] * optalb_factor;
				if (albedo_mode != 1)
					dmod->photo.optical[ilaw].harmR.R.a_save[l][m]
				    = dmod->photo.optical[ilaw].harmR.R.a[l][m].val;
			}
			if (dmod->photo.optical[ilaw].harmR.R.b[l][m].state == 'f') {
				if (albedo_mode != 0)
					dmod->photo.optical[ilaw].harmR.R.b[l][m].val
					= dmod->photo.optical[ilaw].harmR.R.b_save[l][m] * optalb_factor;
				if (albedo_mode != 1)
					dmod->photo.optical[ilaw].harmR.R.b_save[l][m]
				    = dmod->photo.optical[ilaw].harmR.R.b[l][m].val;
			}
		}
	}
}
__global__ void opt_harmlommel_set_nlm_krnl(struct mod_t *dmod, double **nlm, int L){
	/* Multi-threaded kernel (L+1)^2 */
	int l = blockIdx.x * blockDim.x + threadIdx.x;
	int m;
	if (l < (L+1)*(L+1))
	{
		for (m=0; m<=l; m++)
			nlm[l][m] = sqrt( (2*l+1) * exp(dev_gammln(l-m+1.0) - dev_gammln(l+m+1.0)) );
	}
}
__global__ void opt_harmlommel_facet_krnl(struct par_t *dpar, struct mod_t *dmod,
		double **afactor, double **bfactor, double **nlm, int ilaw){
	/* Multi-threaded kernel (nf) */
	int m, l, c = 0;
	int f = blockIdx.x * blockDim.x + threadIdx.x;
	int L = dmod->photo.optical[ilaw].harmR.R.nhar;
	double costheta, phi, plm;

	if (f < dmod->shape.comp[c].real.nf)
	{
		costheta = cos( dmod->shape.comp[c].real.f[f].theta);
		phi = dmod->shape.comp[c].real.f[f].phi;
		for (l=0; l<=L; l++)
			for (m=0; m<=l; m++) {
				plm = nlm[l][m]*dev_plgndr( l, m, costheta);
				afactor[l][m] = cos(m*phi)*plm;
				if (m > 0)
					bfactor[l][m] = sin(m*phi)*plm;
			}

		L = dmod->photo.optical[ilaw].harmR.R.nhar;
		dmod->photo.optical[ilaw].harmR.local[c][f].R.val = 0.0;
		for (l=0; l<=L; l++) {
			dmod->photo.optical[ilaw].harmR.local[c][f].R.val
			+= dmod->photo.optical[ilaw].harmR.R.a[l][0].val
			* afactor[l][0];
			for (m=1; m<=l; m++)
				dmod->photo.optical[ilaw].harmR.local[c][f].R.val
				+= dmod->photo.optical[ilaw].harmR.R.a[l][m].val
				* afactor[l][m]
				+ dmod->photo.optical[ilaw].harmR.R.b[l][m].val
				* bfactor[l][m];
		}

		dev_checkphotopar( dmod->photo.optical[ilaw].harmR.local[c][f].R.val,
				dpar->opt_R_min, dpar->opt_R_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
	}
}
__global__ void opt_inholommel_A_krnl(struct par_t *dpar, struct mod_t *dmod,
		int albedo_mode, double optalb_factor, int ilaw){
	/* Single-threaded kernel */
	if(threadIdx.x == 0) {
		if (dmod->photo.optical[ilaw].inhoR.global.R.state == 'f') {
			if (albedo_mode != 0)
				dmod->photo.optical[ilaw].inhoR.global.R.val
				= dmod->photo.optical[ilaw].inhoR.global.R_save * optalb_factor;
			if (albedo_mode != 1)
				dmod->photo.optical[ilaw].inhoR.global.R_save
				= dmod->photo.optical[ilaw].inhoR.global.R.val;
		}
		dev_checkphotopar( dmod->photo.optical[ilaw].inhoR.global.R.val,
				dpar->opt_R_min, dpar->opt_R_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
	}
}
__global__ void opt_inholommel_facet_krnl(struct par_t *dpar, struct mod_t
		*dmod, int albedo_mode, double optalb_factor, int ilaw){
	int f = blockIdx.x * blockDim.x + threadIdx.x;
	int c = 0;

	if (f < dmod->shape.comp[c].real.nf)
		if (dmod->photo.optical[ilaw].inhoR.local[c][f].R.state == '=') {
			dmod->photo.optical[ilaw].inhoR.local[c][f].R.val
			= dmod->photo.optical[ilaw].inhoR.global.R.val;
		} else if (dmod->photo.optical[ilaw].inhoR.local[c][f].R.state == 'f') {
			if (albedo_mode != 0)
				dmod->photo.optical[ilaw].inhoR.local[c][f].R.val
				= dmod->photo.optical[ilaw].inhoR.local[c][f].R_save
				* optalb_factor;
			if (albedo_mode != 1)
				dmod->photo.optical[ilaw].inhoR.local[c][f].R_save
				= dmod->photo.optical[ilaw].inhoR.local[c][f].R.val;
		}
	dev_checkphotopar( dmod->photo.optical[ilaw].inhoR.local[c][f].R.val,
			dpar->opt_R_min, dpar->opt_R_max, 3,
			&dpar->badphoto, &dpar->badphoto_logfactor);
}
__global__ void opt_hapke_krnl(struct par_t *dpar, struct mod_t *dmod,
		int albedo_mode, double optalb_factor, int ilaw){
	/* Single-threaded kernel */
	if (threadIdx.x == 0) {
		if (dmod->photo.optical[ilaw].hapke.w.state == 'f') {
			if (albedo_mode != 0)
				dmod->photo.optical[ilaw].hapke.w.val = dmod->photo.optical[ilaw].hapke.w_save
				* optalb_factor;
			if (albedo_mode != 1)
				dmod->photo.optical[ilaw].hapke.w_save = dmod->photo.optical[ilaw].hapke.w.val;
		}
		dev_checkphotopar( dmod->photo.optical[ilaw].hapke.w.val,
				dpar->opt_w_min, dpar->opt_w_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].hapke.h.val,
				dpar->opt_h_min, dpar->opt_h_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].hapke.B0.val,
				dpar->opt_B0_min, dpar->opt_B0_max, 2,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].hapke.g.val,
				dpar->opt_g_min, dpar->opt_g_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].hapke.theta.val,
				dpar->opt_theta_min, dpar->opt_theta_max, 2,
				&dpar->badphoto, &dpar->badphoto_logfactor);
	}
}
__global__ void opt_harmhapke_A_krnl(struct mod_t *dmod, int albedo_mode,
		double optalb_factor, int ilaw)
{
	/* Single-threaded kernel */
	int l, m, L;

	if(threadIdx.x == 0) {
		L = dmod->photo.optical[ilaw].harmhapke.w.nhar;
		for (l=0; l<=L; l++) {
			if (dmod->photo.optical[ilaw].harmhapke.w.a[l][0].state == 'f') {
				if (albedo_mode != 0)
					dmod->photo.optical[ilaw].harmhapke.w.a[l][0].val
					= dmod->photo.optical[ilaw].harmhapke.w.a_save[l][0] * optalb_factor;
				if (albedo_mode != 1)
					dmod->photo.optical[ilaw].harmhapke.w.a_save[l][0]
					                                                   = dmod->photo.optical[ilaw].harmhapke.w.a[l][0].val;
			}
			for (m=1; m<=l; m++) {
				if (dmod->photo.optical[ilaw].harmhapke.w.a[l][m].state == 'f') {
					if (albedo_mode != 0)
						dmod->photo.optical[ilaw].harmhapke.w.a[l][m].val
						= dmod->photo.optical[ilaw].harmhapke.w.a_save[l][m] * optalb_factor;
					if (albedo_mode != 1)
						dmod->photo.optical[ilaw].harmhapke.w.a_save[l][m]
						                                                   = dmod->photo.optical[ilaw].harmhapke.w.a[l][m].val;
				}
				if (dmod->photo.optical[ilaw].harmhapke.w.b[l][m].state == 'f') {
					if (albedo_mode != 0)
						dmod->photo.optical[ilaw].harmhapke.w.b[l][m].val
						= dmod->photo.optical[ilaw].harmhapke.w.b_save[l][m] * optalb_factor;
					if (albedo_mode != 1)
						dmod->photo.optical[ilaw].harmhapke.w.b_save[l][m]
						                                                   = dmod->photo.optical[ilaw].harmhapke.w.b[l][m].val;
				}
			}
		}
	}
}
__global__ void opt_harmhapke_Lmax_krnl(struct mod_t *dmod, int ilaw)
{
	/* Single-threaded kernel */
	if (threadIdx.x == 0) {
		dLmax = MAX( dmod->photo.optical[ilaw].harmhapke.w.nhar,
				dmod->photo.optical[ilaw].harmhapke.h.nhar);
		dLmax = MAX( dLmax, dmod->photo.optical[ilaw].harmhapke.B0.nhar);
		dLmax = MAX( dLmax, dmod->photo.optical[ilaw].harmhapke.g.nhar);
		dLmax = MAX( dLmax, dmod->photo.optical[ilaw].harmhapke.theta.nhar);
	}
}
__global__ void opt_harmhapke_kaas_nlm_krnl(struct mod_t *dmod, double **nlm, int Lmax) {
	/* Lmax-threaded kernel */
	int l = blockIdx.x * blockDim.x + threadIdx.x;
	int m;
	if (l < Lmax) {
		for (m=0; m<=l; m++)
			nlm[l][m] = sqrt( (2*l+1) * exp(dev_gammln(l-m+1.0) -
					dev_gammln(l+m+1.0)) );
	}
}
__global__ void opt_harmhapke_facet_krnl(struct par_t *dpar, struct mod_t
		*dmod, double **nlm,	double **afactor, double **bfactor, int nf,
		int Lmax, int ilaw)
{
	/* nf-threaded kernel */
	int f = blockIdx.x * blockDim.x + threadIdx.x;
	int c = 0, L, l, m;
	double costheta, phi, plm;

	if (f < nf) {
		costheta = cos( dmod->shape.comp[c].real.f[f].theta);
		phi = dmod->shape.comp[c].real.f[f].phi;
		for (l=0; l<=Lmax; l++)
			for (m=0; m<=l; m++) {
				plm = nlm[l][m]*dev_plgndr( l, m, costheta);
				afactor[l][m] = cos(m*phi)*plm;
				if (m > 0)
					bfactor[l][m] = sin(m*phi)*plm;
			}

		L = dmod->photo.optical[ilaw].harmhapke.w.nhar;
		dmod->photo.optical[ilaw].harmhapke.local[c][f].w.val = 0.0;
		for (l=0; l<=L; l++) {
			dmod->photo.optical[ilaw].harmhapke.local[c][f].w.val
			+= dmod->photo.optical[ilaw].harmhapke.w.a[l][0].val
			* afactor[l][0];
			for (m=1; m<=l; m++)
				dmod->photo.optical[ilaw].harmhapke.local[c][f].w.val
				+= dmod->photo.optical[ilaw].harmhapke.w.a[l][m].val
				* afactor[l][m]
				             + dmod->photo.optical[ilaw].harmhapke.w.b[l][m].val
				             * bfactor[l][m];
		}

		L = dmod->photo.optical[ilaw].harmhapke.h.nhar;
		dmod->photo.optical[ilaw].harmhapke.local[c][f].h.val = 0.0;
		for (l=0; l<=L; l++) {
			dmod->photo.optical[ilaw].harmhapke.local[c][f].h.val
			+= dmod->photo.optical[ilaw].harmhapke.h.a[l][0].val
			* afactor[l][0];
			for (m=1; m<=l; m++)
				dmod->photo.optical[ilaw].harmhapke.local[c][f].h.val
				+= dmod->photo.optical[ilaw].harmhapke.h.a[l][m].val
				* afactor[l][m]
				             + dmod->photo.optical[ilaw].harmhapke.h.b[l][m].val
				             * bfactor[l][m];
		}

		L = dmod->photo.optical[ilaw].harmhapke.B0.nhar;
		dmod->photo.optical[ilaw].harmhapke.local[c][f].B0.val = 0.0;
		for (l=0; l<=L; l++) {
			dmod->photo.optical[ilaw].harmhapke.local[c][f].B0.val
			+= dmod->photo.optical[ilaw].harmhapke.B0.a[l][0].val
			* afactor[l][0];
			for (m=1; m<=l; m++)
				dmod->photo.optical[ilaw].harmhapke.local[c][f].B0.val
				+= dmod->photo.optical[ilaw].harmhapke.B0.a[l][m].val
				* afactor[l][m]
				             + dmod->photo.optical[ilaw].harmhapke.B0.b[l][m].val
				             * bfactor[l][m];
		}

		L = dmod->photo.optical[ilaw].harmhapke.g.nhar;
		dmod->photo.optical[ilaw].harmhapke.local[c][f].g.val = 0.0;
		for (l=0; l<=L; l++) {
			dmod->photo.optical[ilaw].harmhapke.local[c][f].g.val
			+= dmod->photo.optical[ilaw].harmhapke.g.a[l][0].val
			* afactor[l][0];
			for (m=1; m<=l; m++)
				dmod->photo.optical[ilaw].harmhapke.local[c][f].g.val
				+= dmod->photo.optical[ilaw].harmhapke.g.a[l][m].val
				* afactor[l][m]
				             + dmod->photo.optical[ilaw].harmhapke.g.b[l][m].val
				             * bfactor[l][m];
		}

		L = dmod->photo.optical[ilaw].harmhapke.theta.nhar;
		dmod->photo.optical[ilaw].harmhapke.local[c][f].theta.val = 0.0;
		for (l=0; l<=L; l++) {
			dmod->photo.optical[ilaw].harmhapke.local[c][f].theta.val
			+= dmod->photo.optical[ilaw].harmhapke.theta.a[l][0].val
			* afactor[l][0];
			for (m=1; m<=l; m++)
				dmod->photo.optical[ilaw].harmhapke.local[c][f].theta.val
				+= dmod->photo.optical[ilaw].harmhapke.theta.a[l][m].val
				* afactor[l][m]
				             + dmod->photo.optical[ilaw].harmhapke.theta.b[l][m].val
				             * bfactor[l][m];
		}

		dev_checkphotopar( dmod->photo.optical[ilaw].harmhapke.local[c][f].w.val,
				dpar->opt_w_min, dpar->opt_w_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].harmhapke.local[c][f].h.val,
				dpar->opt_h_min, dpar->opt_h_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].harmhapke.local[c][f].B0.val,
				dpar->opt_B0_min, dpar->opt_B0_max, 2,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].harmhapke.local[c][f].g.val,
				dpar->opt_g_min, dpar->opt_g_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].harmhapke.local[c][f].theta.val,
				dpar->opt_theta_min, dpar->opt_theta_max, 2,
				&dpar->badphoto, &dpar->badphoto_logfactor);
	}
}
__global__ void opt_inhohapke_A_krnl(struct par_t *dpar, struct mod_t *dmod,
		int albedo_mode, double optalb_factor, int ilaw) {
	/* Single-threaded kernel */
	if (threadIdx.x == 0) {
		if (dmod->photo.optical[ilaw].inhohapke.global.w.state == 'f') {
			if (albedo_mode != 0)
				dmod->photo.optical[ilaw].inhohapke.global.w.val
				= dmod->photo.optical[ilaw].inhohapke.global.w_save * optalb_factor;
			if (albedo_mode != 1)
				dmod->photo.optical[ilaw].inhohapke.global.w_save
				= dmod->photo.optical[ilaw].inhohapke.global.w.val;
		}
		dev_checkphotopar( dmod->photo.optical[ilaw].inhohapke.global.w.val,
				dpar->opt_w_min, dpar->opt_w_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].inhohapke.global.h.val,
				dpar->opt_h_min, dpar->opt_h_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].inhohapke.global.B0.val,
				dpar->opt_B0_min, dpar->opt_B0_max, 2,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].inhohapke.global.g.val,
				dpar->opt_g_min, dpar->opt_g_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].inhohapke.global.theta.val,
				dpar->opt_theta_min, dpar->opt_theta_max, 2,
				&dpar->badphoto, &dpar->badphoto_logfactor);
	}
}
__global__ void opt_inhohapke_facet_krnl(struct par_t *dpar, struct mod_t
		*dmod, int nf, int albedo_mode, double optalb_factor, int ilaw)
{
	/* nf-threaded kernel */
	int f = blockIdx.x * blockDim.x + threadIdx.x;
	int c = 0;

	if (f < nf) {
		if (dmod->photo.optical[ilaw].inhohapke.local[c][f].w.state == '=') {
			dmod->photo.optical[ilaw].inhohapke.local[c][f].w.val
			= dmod->photo.optical[ilaw].inhohapke.global.w.val;
		} else if (dmod->photo.optical[ilaw].inhohapke.local[c][f].w.state == 'f') {
			if (albedo_mode != 0)
				dmod->photo.optical[ilaw].inhohapke.local[c][f].w.val
				= dmod->photo.optical[ilaw].inhohapke.local[c][f].w_save
				* optalb_factor;
			if (albedo_mode != 1)
				dmod->photo.optical[ilaw].inhohapke.local[c][f].w_save
				= dmod->photo.optical[ilaw].inhohapke.local[c][f].w.val;
		}
		if (dmod->photo.optical[ilaw].inhohapke.local[c][f].h.state == '=')
			dmod->photo.optical[ilaw].inhohapke.local[c][f].h.val
			= dmod->photo.optical[ilaw].inhohapke.global.h.val;
		if (dmod->photo.optical[ilaw].inhohapke.local[c][f].B0.state == '=')
			dmod->photo.optical[ilaw].inhohapke.local[c][f].B0.val
			= dmod->photo.optical[ilaw].inhohapke.global.B0.val;
		if (dmod->photo.optical[ilaw].inhohapke.local[c][f].g.state == '=')
			dmod->photo.optical[ilaw].inhohapke.local[c][f].g.val
			= dmod->photo.optical[ilaw].inhohapke.global.g.val;
		if (dmod->photo.optical[ilaw].inhohapke.local[c][f].theta.state == '=')
			dmod->photo.optical[ilaw].inhohapke.local[c][f].theta.val
			= dmod->photo.optical[ilaw].inhohapke.global.theta.val;

		dev_checkphotopar( dmod->photo.optical[ilaw].inhohapke.local[c][f].w.val,
				dpar->opt_w_min, dpar->opt_w_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].inhohapke.local[c][f].h.val,
				dpar->opt_h_min, dpar->opt_h_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].inhohapke.local[c][f].B0.val,
				dpar->opt_B0_min, dpar->opt_B0_max, 2,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].inhohapke.local[c][f].g.val,
				dpar->opt_g_min, dpar->opt_g_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].inhohapke.local[c][f].theta.val,
				dpar->opt_theta_min, dpar->opt_theta_max, 2,
				&dpar->badphoto, &dpar->badphoto_logfactor);
	}
}
__global__ void opt_kaas_krnl(struct par_t *dpar, struct mod_t *dmod,
		int albedo_mode, double optalb_factor, int ilaw)
{
	/* Single-treaded kernel */
	if (threadIdx.x ==0) {
		if (dmod->photo.optical[ilaw].kaas.R.state == 'f') {
			if (albedo_mode != 0)
				dmod->photo.optical[ilaw].kaas.R.val = dmod->photo.optical[ilaw].kaas.R_save
				* optalb_factor;
			if (albedo_mode != 1)
				dmod->photo.optical[ilaw].kaas.R_save = dmod->photo.optical[ilaw].kaas.R.val;
		}
		dev_checkphotopar( dmod->photo.optical[ilaw].kaas.R.val,
				dpar->opt_R_min, dpar->opt_R_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].kaas.wt.val,
				dpar->opt_wt_min, dpar->opt_wt_max, 0,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].kaas.A0.val,
				dpar->opt_A0_min, dpar->opt_A0_max, 2,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].kaas.D.val,
				dpar->opt_D_min, dpar->opt_D_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].kaas.k.val,
				dpar->opt_k_min, dpar->opt_k_max, 1,
				&dpar->badphoto, &dpar->badphoto_logfactor);
	}
}
__global__ void opt_harmkaas_getL_krnl(struct mod_t *dmod, int ilaw) {
	/* Single-threaded kernel */
	if (threadIdx.x == 0) {
		dL = dmod->photo.optical[ilaw].harmkaas.R.nhar;
	}
}
__global__ void opt_harmkaas_set_ab_and_absave_krnl(struct mod_t *dmod,
		int albedo_mode, double optalb_factor, int L, int ilaw) {
	/* L-threaded kernel */
	int m, l = blockIdx.x * blockDim.x + threadIdx.x;

	if (l < L) {
		for (l=0; l<=L; l++) {
			if (dmod->photo.optical[ilaw].harmkaas.R.a[l][0].state == 'f') {
				if (albedo_mode != 0)
					dmod->photo.optical[ilaw].harmkaas.R.a[l][0].val
					= dmod->photo.optical[ilaw].harmkaas.R.a_save[l][0] * optalb_factor;
				if (albedo_mode != 1)
					dmod->photo.optical[ilaw].harmkaas.R.a_save[l][0]
					= dmod->photo.optical[ilaw].harmkaas.R.a[l][0].val;
			}
			for (m=1; m<=l; m++) {
				if (dmod->photo.optical[ilaw].harmkaas.R.a[l][m].state == 'f') {
					if (albedo_mode != 0)
						dmod->photo.optical[ilaw].harmkaas.R.a[l][m].val
						= dmod->photo.optical[ilaw].harmkaas.R.a_save[l][m] * optalb_factor;
					if (albedo_mode != 1)
						dmod->photo.optical[ilaw].harmkaas.R.a_save[l][m]
					    = dmod->photo.optical[ilaw].harmkaas.R.a[l][m].val;
				}
				if (dmod->photo.optical[ilaw].harmkaas.R.b[l][m].state == 'f') {
					if (albedo_mode != 0)
						dmod->photo.optical[ilaw].harmkaas.R.b[l][m].val
						= dmod->photo.optical[ilaw].harmkaas.R.b_save[l][m] * optalb_factor;
					if (albedo_mode != 1)
						dmod->photo.optical[ilaw].harmkaas.R.b_save[l][m]
					    = dmod->photo.optical[ilaw].harmkaas.R.b[l][m].val;
				}
			}
		}
	}
}
__global__ void opt_harmkaas_Lmax_krnl(struct mod_t *dmod, int ilaw) {
	/* Single-threaded kernel */
	if (threadIdx.x == 0) {
		dLmax = MAX( dmod->photo.optical[ilaw].harmkaas.R.nhar,
				dmod->photo.optical[ilaw].harmkaas.wt.nhar);
		dLmax = MAX( dLmax, dmod->photo.optical[ilaw].harmkaas.A0.nhar);
		dLmax = MAX( dLmax, dmod->photo.optical[ilaw].harmkaas.D.nhar);
		dLmax = MAX( dLmax, dmod->photo.optical[ilaw].harmkaas.k.nhar);
	}
}
__global__ void opt_harmkaas_facet_krnl(struct par_t *dpar, struct mod_t
		*dmod, double **nlm, double **afactor, double **bfactor, int nf,
		int Lmax, int ilaw) {
	/* nf-threaded kernel */
	int f = blockIdx.x * blockDim.x + threadIdx.x;
	int c = 0, L, l, m;
	double costheta, phi, plm;

	if (f > nf) {
		costheta = cos( dmod->shape.comp[c].real.f[f].theta);
		phi = dmod->shape.comp[c].real.f[f].phi;
		for (l=0; l<=Lmax; l++)
			for (m=0; m<=l; m++) {
				plm = nlm[l][m]*dev_plgndr( l, m, costheta);
				afactor[l][m] = cos(m*phi)*plm;
				if (m > 0)
					bfactor[l][m] = sin(m*phi)*plm;
			}

		L = dmod->photo.optical[ilaw].harmkaas.R.nhar;
		dmod->photo.optical[ilaw].harmkaas.local[c][f].R.val = 0.0;
		for (l=0; l<=L; l++) {
			dmod->photo.optical[ilaw].harmkaas.local[c][f].R.val
			+= dmod->photo.optical[ilaw].harmkaas.R.a[l][0].val
			* afactor[l][0];
			for (m=1; m<=l; m++)
				dmod->photo.optical[ilaw].harmkaas.local[c][f].R.val
				+= dmod->photo.optical[ilaw].harmkaas.R.a[l][m].val
				* afactor[l][m]
				+ dmod->photo.optical[ilaw].harmkaas.R.b[l][m].val
				* bfactor[l][m];
		}

		L = dmod->photo.optical[ilaw].harmkaas.wt.nhar;
		dmod->photo.optical[ilaw].harmkaas.local[c][f].wt.val = 0.0;
		for (l=0; l<=L; l++) {
			dmod->photo.optical[ilaw].harmkaas.local[c][f].wt.val
			+= dmod->photo.optical[ilaw].harmkaas.wt.a[l][0].val
			* afactor[l][0];
			for (m=1; m<=l; m++)
				dmod->photo.optical[ilaw].harmkaas.local[c][f].wt.val
				+= dmod->photo.optical[ilaw].harmkaas.wt.a[l][m].val
				* afactor[l][m]
				+ dmod->photo.optical[ilaw].harmkaas.wt.b[l][m].val
				* bfactor[l][m];
		}

		L = dmod->photo.optical[ilaw].harmkaas.A0.nhar;
		dmod->photo.optical[ilaw].harmkaas.local[c][f].A0.val = 0.0;
		for (l=0; l<=L; l++) {
			dmod->photo.optical[ilaw].harmkaas.local[c][f].A0.val
			+= dmod->photo.optical[ilaw].harmkaas.A0.a[l][0].val
			* afactor[l][0];
			for (m=1; m<=l; m++)
				dmod->photo.optical[ilaw].harmkaas.local[c][f].A0.val
				+= dmod->photo.optical[ilaw].harmkaas.A0.a[l][m].val
				* afactor[l][m]
				+ dmod->photo.optical[ilaw].harmkaas.A0.b[l][m].val
				* bfactor[l][m];
		}

		L = dmod->photo.optical[ilaw].harmkaas.D.nhar;
		dmod->photo.optical[ilaw].harmkaas.local[c][f].D.val = 0.0;
		for (l=0; l<=L; l++) {
			dmod->photo.optical[ilaw].harmkaas.local[c][f].D.val
			+= dmod->photo.optical[ilaw].harmkaas.D.a[l][0].val
			* afactor[l][0];
			for (m=1; m<=l; m++)
				dmod->photo.optical[ilaw].harmkaas.local[c][f].D.val
				+= dmod->photo.optical[ilaw].harmkaas.D.a[l][m].val
				* afactor[l][m]
				+ dmod->photo.optical[ilaw].harmkaas.D.b[l][m].val
				* bfactor[l][m];
		}

		L = dmod->photo.optical[ilaw].harmkaas.k.nhar;
		dmod->photo.optical[ilaw].harmkaas.local[c][f].k.val = 0.0;
		for (l=0; l<=L; l++) {
			dmod->photo.optical[ilaw].harmkaas.local[c][f].k.val
			+= dmod->photo.optical[ilaw].harmkaas.k.a[l][0].val
			* afactor[l][0];
			for (m=1; m<=l; m++)
				dmod->photo.optical[ilaw].harmkaas.local[c][f].k.val
				+= dmod->photo.optical[ilaw].harmkaas.k.a[l][m].val
				* afactor[l][m]
				+ dmod->photo.optical[ilaw].harmkaas.k.b[l][m].val
				* bfactor[l][m];
		}

		dev_checkphotopar( dmod->photo.optical[ilaw].harmkaas.local[c][f].R.val,
				dpar->opt_R_min, dpar->opt_R_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].harmkaas.local[c][f].wt.val,
				dpar->opt_wt_min, dpar->opt_wt_max, 0,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].harmkaas.local[c][f].A0.val,
				dpar->opt_A0_min, dpar->opt_A0_max, 2,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].harmkaas.local[c][f].D.val,
				dpar->opt_D_min, dpar->opt_D_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].harmkaas.local[c][f].k.val,
				dpar->opt_k_min, dpar->opt_k_max, 1,
				&dpar->badphoto, &dpar->badphoto_logfactor);
	}
}
__global__ void opt_inhokaas_A_krnl(struct par_t *dpar, struct mod_t *dmod,
		int albedo_mode, double optalb_factor, int ilaw) {
	/* Single-threaded kernel */
	if(threadIdx.x == 0) {
		if (dmod->photo.optical[ilaw].inhokaas.global.R.state == 'f') {
			if (albedo_mode != 0)
				dmod->photo.optical[ilaw].inhokaas.global.R.val
				= dmod->photo.optical[ilaw].inhokaas.global.R_save * optalb_factor;
			if (albedo_mode != 1)
				dmod->photo.optical[ilaw].inhokaas.global.R_save
				= dmod->photo.optical[ilaw].inhokaas.global.R.val;
		}
		dev_checkphotopar( dmod->photo.optical[ilaw].inhokaas.global.R.val,
				dpar->opt_R_min, dpar->opt_R_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].inhokaas.global.wt.val,
				dpar->opt_wt_min, dpar->opt_wt_max, 0,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].inhokaas.global.A0.val,
				dpar->opt_A0_min, dpar->opt_A0_max, 2,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].inhokaas.global.D.val,
				dpar->opt_D_min, dpar->opt_D_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].inhokaas.global.k.val,
				dpar->opt_k_min, dpar->opt_k_max, 1,
				&dpar->badphoto, &dpar->badphoto_logfactor);
	}
}
__global__ void opt_inhokaas_facet_krnl(struct par_t *dpar, struct mod_t
		*dmod, int nf, int albedo_mode, double optalb_factor, int ilaw) {
	/* nf-threaded kernel */
	int f = blockIdx.x * blockDim.x + threadIdx.x;
	int c = 0;

	if (f < nf) {
		if (dmod->photo.optical[ilaw].inhokaas.local[c][f].R.state == '=') {
			dmod->photo.optical[ilaw].inhokaas.local[c][f].R.val
			= dmod->photo.optical[ilaw].inhokaas.global.R.val;
		} else if (dmod->photo.optical[ilaw].inhokaas.local[c][f].R.state == 'f') {
			if (albedo_mode != 0)
				dmod->photo.optical[ilaw].inhokaas.local[c][f].R.val
				= dmod->photo.optical[ilaw].inhokaas.local[c][f].R_save
				* optalb_factor;
			if (albedo_mode != 1)
				dmod->photo.optical[ilaw].inhokaas.local[c][f].R_save
				= dmod->photo.optical[ilaw].inhokaas.local[c][f].R.val;
		}
		if (dmod->photo.optical[ilaw].inhokaas.local[c][f].wt.state == '=')
			dmod->photo.optical[ilaw].inhokaas.local[c][f].wt.val
			= dmod->photo.optical[ilaw].inhokaas.global.wt.val;
		if (dmod->photo.optical[ilaw].inhokaas.local[c][f].A0.state == '=')
			dmod->photo.optical[ilaw].inhokaas.local[c][f].A0.val
			= dmod->photo.optical[ilaw].inhokaas.global.A0.val;
		if (dmod->photo.optical[ilaw].inhokaas.local[c][f].D.state == '=')
			dmod->photo.optical[ilaw].inhokaas.local[c][f].D.val
			= dmod->photo.optical[ilaw].inhokaas.global.D.val;
		if (dmod->photo.optical[ilaw].inhokaas.local[c][f].k.state == '=')
			dmod->photo.optical[ilaw].inhokaas.local[c][f].k.val
			= dmod->photo.optical[ilaw].inhokaas.global.k.val;
		dev_checkphotopar( dmod->photo.optical[ilaw].inhokaas.local[c][f].R.val,
				dpar->opt_R_min, dpar->opt_R_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].inhokaas.local[c][f].wt.val,
				dpar->opt_wt_min, dpar->opt_wt_max, 0,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].inhokaas.local[c][f].A0.val,
				dpar->opt_A0_min, dpar->opt_A0_max, 2,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].inhokaas.local[c][f].D.val,
				dpar->opt_D_min, dpar->opt_D_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.optical[ilaw].inhokaas.local[c][f].k.val,
				dpar->opt_k_min, dpar->opt_k_max, 1,
				&dpar->badphoto, &dpar->badphoto_logfactor);
	}
}
__global__ void rad_cosinelaw_krnl(struct par_t *dpar, struct mod_t *dmod,
		int albedo_mode, double radalb_factor, int ilaw) {
	/* Single-threaded kernel */
	if (threadIdx.x == 0) {

		if (dmod->photo.radar[ilaw].RC.R.state == 'f') {
			if (albedo_mode != 0)
				dmod->photo.radar[ilaw].RC.R.val = dmod->photo.radar[ilaw].RC.R_save * radalb_factor;
			if (albedo_mode != 1)
				dmod->photo.radar[ilaw].RC.R_save = dmod->photo.radar[ilaw].RC.R.val;
		}
		dev_checkphotopar( dmod->photo.radar[ilaw].RC.R.val,dpar->rad_R_min,
				dpar->rad_R_max, 3,&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.radar[ilaw].RC.C.val, dpar->rad_C_min,
				dpar->rad_C_max, 3,&dpar->badphoto, &dpar->badphoto_logfactor);


	}
}
__global__ void rad_tabularlaw_get_n_krnl(struct mod_t *dmod, int ilaw) {
	/* Single-threaded kernel just to get n */
	if (threadIdx.x ==0)
		dn = dmod->photo.radar[ilaw].tabular.n;
}
__global__ void rad_tabularlaw_val_krnl(struct par_t *dpar, struct mod_t
		*dmod, int albedo_mode, double radalb_factor, int ilaw) {
	/* n-threaded kernel */
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < dn) {
		if (dmod->photo.radar[ilaw].tabular.rho[i].state == '=') {
			dmod->photo.radar[ilaw].tabular.rho[i].val = dmod->photo.radar[ilaw].tabular.rho[i-1].val;
		} else if (dmod->photo.radar[ilaw].tabular.rho[i].state == 'f') {
			if (albedo_mode != 0)
				dmod->photo.radar[ilaw].tabular.rho[i].val = dmod->photo.radar[ilaw].tabular.rho_save[i]
				* radalb_factor;
			if (albedo_mode != 1)
				dmod->photo.radar[ilaw].tabular.rho_save[i] = dmod->photo.radar[ilaw].tabular.rho[i].val;
		}
		dev_checkphotopar( dmod->photo.radar[ilaw].tabular.rho[i].val,
				dpar->rad_rho_min, dpar->rad_rho_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
	}
}
__global__ void rad_cosinelaw_qs_krnl(struct par_t *dpar, struct mod_t
		*dmod, int albedo_mode, double radalb_factor, int ilaw) {
	/* Single-threaded wrapper kernel */
	if (threadIdx.x ==0) {
		if (dmod->photo.radar[ilaw].quasispec.R.state == 'f') {
			if (albedo_mode != 0)
				dmod->photo.radar[ilaw].quasispec.R.val = dmod->photo.radar[ilaw].quasispec.R_save
				* radalb_factor;
			if (albedo_mode != 1)
				dmod->photo.radar[ilaw].quasispec.R_save = dmod->photo.radar[ilaw].quasispec.R.val;
		}
		dev_checkphotopar( dmod->photo.radar[ilaw].quasispec.R.val,
				dpar->rad_R_min, dpar->rad_R_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.radar[ilaw].quasispec.C.val,
				dpar->rad_C_min, dpar->rad_C_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
	}
}
__global__ void rad_cosine_cosine_krnl(struct par_t *dpar, struct mod_t
		*dmod, int albedo_mode, double radalb_factor, int ilaw) {
	/* Single-threaded wrapper kernel */
	if (threadIdx.x == 0) {
		if (dmod->photo.radar[ilaw].hybrid.qs.R.state == 'f') {
			if (albedo_mode != 0)
				dmod->photo.radar[ilaw].hybrid.qs.R.val = dmod->photo.radar[ilaw].hybrid.qs.R_save
				* radalb_factor;
			if (albedo_mode != 1)
				dmod->photo.radar[ilaw].hybrid.qs.R_save = dmod->photo.radar[ilaw].hybrid.qs.R.val;
		}
		if (dmod->photo.radar[ilaw].hybrid.diff.R.state == 'f') {
			if (albedo_mode != 0)
				dmod->photo.radar[ilaw].hybrid.diff.R.val = dmod->photo.radar[ilaw].hybrid.diff.R_save
				* radalb_factor;
			if (albedo_mode != 1)
				dmod->photo.radar[ilaw].hybrid.diff.R_save = dmod->photo.radar[ilaw].hybrid.diff.R.val;
		}
		dev_checkphotopar( dmod->photo.radar[ilaw].hybrid.qs.R.val,
				dpar->rad_R_min, dpar->rad_R_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.radar[ilaw].hybrid.qs.C.val,
				dpar->rad_C_min, dpar->rad_C_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.radar[ilaw].hybrid.diff.R.val,
				dpar->rad_R_min, dpar->rad_R_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.radar[ilaw].hybrid.diff.C.val,
				dpar->rad_C_min, dpar->rad_C_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
	}
}
__global__ void rad_harmcosine_diff_getL_krnl(struct mod_t *dmod, int ilaw) {
	/* Single-threaded kernel to get L */
	if (threadIdx.x == 0) {
		dL = dmod->photo.radar[ilaw].harmcosine.R.nhar;
		dLmax = MAX( dmod->photo.radar[ilaw].harmcosine.R.nhar,
		             dmod->photo.radar[ilaw].harmcosine.C.nhar);
	}
}
__global__ void rad_harmcosine_diff_set_ab_krnl(struct mod_t *dmod,
		int albedo_mode, double radalb_factor, int L, int ilaw) {
	/* L-threaded kernel */
	int l = blockIdx.x * blockDim.x + threadIdx.x;
	int m;
	if (l < L) {
		if (dmod->photo.radar[ilaw].harmcosine.R.a[l][0].state == 'f') {
			if (albedo_mode != 0)
				dmod->photo.radar[ilaw].harmcosine.R.a[l][0].val
				= dmod->photo.radar[ilaw].harmcosine.R.a_save[l][0] * radalb_factor;
			if (albedo_mode != 1)
				dmod->photo.radar[ilaw].harmcosine.R.a_save[l][0]
			    = dmod->photo.radar[ilaw].harmcosine.R.a[l][0].val;
		}
		for (m=1; m<=l; m++) {
			if (dmod->photo.radar[ilaw].harmcosine.R.a[l][m].state == 'f') {
				if (albedo_mode != 0)
					dmod->photo.radar[ilaw].harmcosine.R.a[l][m].val
					= dmod->photo.radar[ilaw].harmcosine.R.a_save[l][m] * radalb_factor;
				if (albedo_mode != 1)
					dmod->photo.radar[ilaw].harmcosine.R.a_save[l][m]
		            = dmod->photo.radar[ilaw].harmcosine.R.a[l][m].val;
			}
			if (dmod->photo.radar[ilaw].harmcosine.R.b[l][m].state == 'f') {
				if (albedo_mode != 0)
					dmod->photo.radar[ilaw].harmcosine.R.b[l][m].val
					= dmod->photo.radar[ilaw].harmcosine.R.b_save[l][m] * radalb_factor;
				if (albedo_mode != 1)
					dmod->photo.radar[ilaw].harmcosine.R.b_save[l][m]
		            = dmod->photo.radar[ilaw].harmcosine.R.b[l][m].val;
			}
		}
	}
}
__global__ void rad_harmcosine_facet_krnl(struct par_t *dpar, struct
		mod_t *dmod, double **nlm, double **afactor, double **bfactor,
		int nf, int Lmax, int ilaw) {
	/* nf-threaded kernel */
	int f = blockIdx.x * blockDim.x + threadIdx.x;
	int l, L, m, c = 0;
	double costheta, phi, plm;

	if (f < nf) {
		costheta = cos( dmod->shape.comp[c].real.f[f].theta);
		phi = dmod->shape.comp[c].real.f[f].phi;
		for (l=0; l<=Lmax; l++)
			for (m=0; m<=l; m++) {
				plm = nlm[l][m]*dev_plgndr( l, m, costheta);
				afactor[l][m] = cos(m*phi)*plm;
				if (m > 0)
					bfactor[l][m] = sin(m*phi)*plm;
			}

		L = dmod->photo.radar[ilaw].harmcosine.R.nhar;
		dmod->photo.radar[ilaw].harmcosine.local[c][f].R.val = 0.0;
		for (l=0; l<=L; l++) {
			dmod->photo.radar[ilaw].harmcosine.local[c][f].R.val
			+= dmod->photo.radar[ilaw].harmcosine.R.a[l][0].val
			* afactor[l][0];
			for (m=1; m<=l; m++)
				dmod->photo.radar[ilaw].harmcosine.local[c][f].R.val
				+= dmod->photo.radar[ilaw].harmcosine.R.a[l][m].val
				* afactor[l][m]
			    + dmod->photo.radar[ilaw].harmcosine.R.b[l][m].val
			    * bfactor[l][m];
		}

		L = dmod->photo.radar[ilaw].harmcosine.C.nhar;
		dmod->photo.radar[ilaw].harmcosine.local[c][f].C.val = 0.0;
		for (l=0; l<=L; l++) {
			dmod->photo.radar[ilaw].harmcosine.local[c][f].C.val
			+= dmod->photo.radar[ilaw].harmcosine.C.a[l][0].val
			* afactor[l][0];
			for (m=1; m<=l; m++)
				dmod->photo.radar[ilaw].harmcosine.local[c][f].C.val
				+= dmod->photo.radar[ilaw].harmcosine.C.a[l][m].val
				* afactor[l][m]
			    + dmod->photo.radar[ilaw].harmcosine.C.b[l][m].val
			    * bfactor[l][m];
		}

		dev_checkphotopar( dmod->photo.radar[ilaw].harmcosine.local[c][f].R.val,
				dpar->rad_R_min, dpar->rad_R_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.radar[ilaw].harmcosine.local[c][f].C.val,
				dpar->rad_C_min, dpar->rad_C_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
	}
}
__global__ void rad_inhocosine_set_ab_krnl(struct par_t *dpar, struct mod_t
		*dmod, int albedo_mode, double radalb_factor, int ilaw) {
	/* Single-threaded kernel */
	if (threadIdx.x == 0) {

		if (dmod->photo.radar[ilaw].inhocosine.global.R.state == 'f') {
			if (albedo_mode != 0)
				dmod->photo.radar[ilaw].inhocosine.global.R.val
				= dmod->photo.radar[ilaw].inhocosine.global.R_save * radalb_factor;
			if (albedo_mode != 1)
				dmod->photo.radar[ilaw].inhocosine.global.R_save
				= dmod->photo.radar[ilaw].inhocosine.global.R.val;
		}
		dev_checkphotopar( dmod->photo.radar[ilaw].inhocosine.global.R.val,
				dpar->rad_R_min, dpar->rad_R_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.radar[ilaw].inhocosine.global.C.val,
				dpar->rad_C_min, dpar->rad_C_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
	}
}
__global__ void rad_inhocosine_facet_krnl(struct par_t *dpar, struct mod_t
		*dmod, int nf, int albedo_mode, double radalb_factor, int ilaw) {
	/* nf-threaded kernel */
	int f = blockIdx.x * blockDim.x + threadIdx.x;
	int c = 0;

	if (f < nf) {
		if (dmod->photo.radar[ilaw].inhocosine.local[c][f].R.state == '=') {
			dmod->photo.radar[ilaw].inhocosine.local[c][f].R.val
			= dmod->photo.radar[ilaw].inhocosine.global.R.val;
		} else if (dmod->photo.radar[ilaw].inhocosine.local[c][f].R.state == 'f') {
			if (albedo_mode != 0)
				dmod->photo.radar[ilaw].inhocosine.local[c][f].R.val
				= dmod->photo.radar[ilaw].inhocosine.local[c][f].R_save
				* radalb_factor;
			if (albedo_mode != 1)
				dmod->photo.radar[ilaw].inhocosine.local[c][f].R_save
				= dmod->photo.radar[ilaw].inhocosine.local[c][f].R.val;
		}
		if (dmod->photo.radar[ilaw].inhocosine.local[c][f].C.state == '=')
			dmod->photo.radar[ilaw].inhocosine.local[c][f].C.val
			= dmod->photo.radar[ilaw].inhocosine.global.C.val;
		dev_checkphotopar( dmod->photo.radar[ilaw].inhocosine.local[c][f].R.val,
				dpar->rad_R_min, dpar->rad_R_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
		dev_checkphotopar( dmod->photo.radar[ilaw].inhocosine.local[c][f].C.val,
				dpar->rad_C_min, dpar->rad_C_max, 3,
				&dpar->badphoto, &dpar->badphoto_logfactor);
	}
}
/* Note that realize_photo_cuda currently cannot handle more than one optlaw
 * and one radlaw at a time. The capability to handle multiple laws may be
 * added at a later date.   (November 1, 2016) */
__host__ void realize_photo_gpu( struct par_t *dpar, struct mod_t *dmod,
		double radalb_factor, double optalb_factor, int albedo_mode, int nf)
{
	int Lmax, L, i, n, noptlaws, nradlaws, ilaw;
 	double **nlm, **afactor, **bfactor=NULL;
	unsigned char opttype, radtype;
	dim3 BLK, THD, THD1;
	THD.x = maxThreadsPerBlock;
	BLK.x = floor((THD.x + nf - 1)/THD.x);

	/* Initialize illegal photometric parameters flag & get opttype and radtype */
	get_photo_types_krnl<<<1,1>>>(dpar, dmod);//, dopttype, radtype);
	checkErrorAfterKernelLaunch("get_photo_types_krnl, line ");
	gpuErrchk(cudaMemcpyFromSymbol(&opttype, dopttype, sizeof(dopttype), 0, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpyFromSymbol(&radtype, dradtype, sizeof(dopttype), 0, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpyFromSymbol(&noptlaws, dnoptlaws, sizeof(int), 0, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpyFromSymbol(&nradlaws, dnradlaws, sizeof(int), 0, cudaMemcpyDeviceToHost));

	/* Check that all optical scattering law parameters have legal values  */
	for (ilaw=0; ilaw<noptlaws; ilaw++) {
		THD.x = maxThreadsPerBlock;
		switch (opttype) {
		case NOLAW:
			break;
		case GEOMETRICAL:
		case LAMBERTLAW:
		case LOMMEL:
			/* Call single-threaded kernel for Lommel */
			opt_lommel_krnl<<<1,1>>>(dpar, dmod, optalb_factor, albedo_mode, ilaw);
			checkErrorAfterKernelLaunch("opt_lommel_krnl");
			break;
		case HARMLAMBERT:
		case HARMLOMMEL:
			int L;
			/* Call single-threaded kernel for Lommel */
			opt_harmlommel_getL_krnl<<<1,1>>>(dmod, ilaw);
			checkErrorAfterKernelLaunch("opt_harmlommel_getL_krnl");
			gpuErrchk(cudaMemcpyFromSymbol(&L, dL, sizeof(dL), 0,
					cudaMemcpyDeviceToHost));

			/* Launch an L-threaded kernel (# of harmonics) */
			THD1.x = L;
			opt_harmlommel_set_ab_krnl<<<1,THD>>>(dmod, optalb_factor, albedo_mode, ilaw);
			gpuErrchk(cudaMemcpyFromSymbol(&L, dL, sizeof(dL), 0,
					cudaMemcpyDeviceToHost));

			/* Set up matrices first */
			gpuErrchk(cudaMalloc((void**)&nlm, 	   sizeof(double*)*(L+1)));
			gpuErrchk(cudaMalloc((void**)&afactor, sizeof(double*)*(L+1)));
			gpuErrchk(cudaMalloc((void**)&bfactor, sizeof(double*)* L));

			for (i=0; i<=L; i++) {
				gpuErrchk(cudaMalloc((void**)&nlm[i], sizeof(double)*(L+1)));
				gpuErrchk(cudaMalloc((void**)&afactor[i], sizeof(double)*(L+1)));
				if (i<L)
					gpuErrchk(cudaMalloc((void**)&bfactor[i], sizeof(double)*L));
			}
			/* End setup of matrices */

			/* Launch the kernel to set up nlm[L+1][L+1] */
			THD1.x = (L+1)*(L+1);
			opt_harmlommel_set_nlm_krnl<<<1,THD>>>(dmod, nlm, L);
			checkErrorAfterKernelLaunch("opt_harmlommel_set_nlm_krnl");

			/* Host-side check whether to execute kernel */
			if (L > 0) {
				opt_harmlommel_facet_krnl<<<BLK,THD>>>(dpar, dmod, afactor, bfactor, nlm, ilaw);
				checkErrorAfterKernelLaunch("opt_harmlommel_vertex_krnl");
			}
			/* Free up the previously allocated double-pointers nlm, afactor, and bfactor */
			cudaFree(nlm);
			cudaFree(afactor);
			if (L > 0)	cudaFree(bfactor);
			break;
		case INHOLAMBERT:
		case INHOLOMMEL:
			/* Call single-threaded kernel for inho-Lommel */
			opt_inholommel_A_krnl<<<1,1>>>(dpar, dmod, albedo_mode, optalb_factor, ilaw);
			checkErrorAfterKernelLaunch("opt_inholommel_A_krnl");

			opt_inholommel_facet_krnl<<<BLK,THD>>>(dpar, dmod,
					albedo_mode, optalb_factor, ilaw);
			checkErrorAfterKernelLaunch("opt_inholommel_vertex_krnl");
			break;
		case HAPKE:
			/* Call single-threaded wrapper kernel for the Hapke case */
			opt_hapke_krnl<<<1,1>>>(dpar, dmod, albedo_mode, optalb_factor, ilaw);
			checkErrorAfterKernelLaunch("opt_hapke_krnl");
			break;
		case HARMHAPKE:
			/* Call single-threaded kernel for Harmhapke part A */
			opt_harmhapke_A_krnl<<<1,1>>>(dmod, albedo_mode, optalb_factor, ilaw);
			checkErrorAfterKernelLaunch("opt_harmhapke_A_krnl");

			/* Find out what Lmax is */
			opt_harmhapke_Lmax_krnl<<<1,1>>>(dmod, ilaw);
			checkErrorAfterKernelLaunch("opt_harmhapke_Lmax_krnl");

			/* Copy Lmax (for sizing arrays) from device to host  then allocate double pntrs */
			gpuErrchk(cudaMemcpyFromSymbol(&Lmax, dLmax, sizeof(dLmax), 0,
					cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMalloc((void**)&nlm, 	   sizeof(double*)*(Lmax+1)));
			gpuErrchk(cudaMalloc((void**)&afactor, sizeof(double*)*(Lmax+1)));
			gpuErrchk(cudaMalloc((void**)&bfactor, sizeof(double*)* Lmax));

			for (i=0; i<=Lmax; i++) {
				gpuErrchk(cudaMalloc((void**)&nlm[i], sizeof(double)*(Lmax+1)));
				gpuErrchk(cudaMalloc((void**)&afactor[i], sizeof(double)*(Lmax+1)));
				if (i<Lmax && Lmax >0)
					gpuErrchk(cudaMalloc((void**)&bfactor[i], sizeof(double)*Lmax));
			}
			/* End setup of matrices */

			/* Launch nlm population kernel */
			THD1.x = Lmax;
			opt_harmhapke_kaas_nlm_krnl<<<1,THD>>>(dmod, nlm, Lmax);
			checkErrorAfterKernelLaunch("opt_harmhapke_nlm_krnl");

			opt_harmhapke_facet_krnl<<<BLK,THD>>>(dpar, dmod, nlm, afactor,
					bfactor, nf, Lmax, ilaw);
			checkErrorAfterKernelLaunch("opt_harmhapke_facet_krnl");

			/* Free the previously allocated double pointers */
			cudaFree(nlm);
			cudaFree(afactor);
			if (Lmax > 0)	cudaFree(bfactor);
			break;
		case INHOHAPKE:
			/* Call single-threaded kernel for Harmhapke part A */
			opt_inhohapke_A_krnl<<<1,1>>>(dpar, dmod, albedo_mode,
					optalb_factor, ilaw);
			checkErrorAfterKernelLaunch("opt_inhohapke_A_krnl");

			opt_inhohapke_facet_krnl<<<BLK,THD>>>(dpar, dmod, nf, albedo_mode,
					optalb_factor, ilaw);
			checkErrorAfterKernelLaunch("opt_inhohapke_facet_krnl");
			break;
		case KAASALAINEN:
			/* Call single-threaded kernel for Harmhapke part A */
			opt_kaas_krnl<<<1,1>>>(dpar, dmod, albedo_mode, optalb_factor, ilaw);
			checkErrorAfterKernelLaunch("opt_kaas_krnl");
			break;
		case HARMKAAS:
			/* Call a single-threaded kernel to fetch L */
			opt_harmkaas_getL_krnl<<<1,1>>>(dmod, ilaw);
			checkErrorAfterKernelLaunch("opt_harmkaas_getL_krnl");
			gpuErrchk(cudaMemcpyFromSymbol(&L, dL, sizeof(dL), 0, cudaMemcpyDeviceToHost));

			/* Configure and launch L-threaded kernel to set a, b, a_save, b_save */
			THD1.x = L;
			opt_harmkaas_set_ab_and_absave_krnl<<<1,THD>>>(dmod, albedo_mode,
					optalb_factor, L, ilaw);

			/* Launch single-threaded kernel to find Lmax */
			opt_harmkaas_Lmax_krnl<<<1,1>>>(dmod, ilaw);
			checkErrorAfterKernelLaunch("opt_harmkaas_get_Lmax_krnl");

			/* Copy Lmax (for sizing arrays) from device to host  then allocate double pntrs */
			gpuErrchk(cudaMemcpyFromSymbol(&Lmax, dLmax, sizeof(dLmax), 0,
					cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMalloc((void**)&nlm, 	   sizeof(double*)*(Lmax+1)));
			gpuErrchk(cudaMalloc((void**)&afactor, sizeof(double*)*(Lmax+1)));
			gpuErrchk(cudaMalloc((void**)&bfactor, sizeof(double*)* Lmax));

			for (i=0; i<=Lmax; i++) {
				gpuErrchk(cudaMalloc((void**)&nlm[i], sizeof(double)*(Lmax+1)));
				gpuErrchk(cudaMalloc((void**)&afactor[i], sizeof(double)*(Lmax+1)));
				if (i<Lmax && Lmax >0)
					gpuErrchk(cudaMalloc((void**)&bfactor[i], sizeof(double)*Lmax));
			}
			/* End setup of matrices */

			/* Launch nlm population kernel */
			THD1.x = Lmax;
			opt_harmhapke_kaas_nlm_krnl<<<BLK,THD>>>(dmod, nlm, Lmax);
			checkErrorAfterKernelLaunch("opt_harmhapke_kaas_nlm_krnl");

			opt_harmkaas_facet_krnl<<<BLK,THD>>>(dpar, dmod, nlm, afactor,
					bfactor, nf, Lmax, ilaw);
			checkErrorAfterKernelLaunch("opt_harmkaas_facet_krnl");

			cudaFree(nlm);
			cudaFree(afactor);
			if (Lmax > 0)	cudaFree(bfactor);
			break;
		case INHOKAAS:
			/* Call single-threaded kernel to start off inhokaas */
			opt_inhokaas_A_krnl<<<1,1>>>(dpar, dmod, albedo_mode,
					optalb_factor, ilaw);
			checkErrorAfterKernelLaunch("opt_inhokaas_A_krnl, line ");

			opt_inhokaas_facet_krnl<<<BLK,THD>>>(dpar, dmod, nf, albedo_mode,
					optalb_factor, ilaw);
			checkErrorAfterKernelLaunch("opt_harmkaas_facet_krnl");
			break;
		default:
			bailout("realize_photo_gpu: can't handle this optical law yet\n");
		}
	}

	/*  Check that all radar scattering law parameters have legal values  */
 	 for (ilaw=0; ilaw<nradlaws; ilaw++) {
	switch (radtype) {
	case NOLAW:
		break;
	case COSINELAW_DIFF:
		/* Launch single-threaded wrapper kernel for Cosine law radar */
		rad_cosinelaw_krnl<<<1,1>>>(dpar, dmod, albedo_mode, radalb_factor, ilaw);
		checkErrorAfterKernelLaunch("rad_cosinelaw_krnl");
		break;
	case TABULARLAW:
		/* Not sure if this works as written due to pointer memory managment (dynamic) */
		/* Call single-threaded kernel to get n */
		rad_tabularlaw_get_n_krnl<<<1,1>>>(dmod, ilaw);
		checkErrorAfterKernelLaunch("rad_tabularlaw_get_n_krnl,");
		gpuErrchk(cudaMemcpyFromSymbol(&n, dn, sizeof(dn), 0, cudaMemcpyDeviceToHost));

		/* Launch n-threaded kernel to finish tabular law realization */
		BLK.x = 1; THD.x = n;
		rad_tabularlaw_val_krnl<<<BLK,THD>>>(dpar, dmod, albedo_mode,
				radalb_factor, ilaw);
		checkErrorAfterKernelLaunch("rad_tabularlaw_val_krnl");

		break;
	case GAUSSIANLAW :
	case HAGFORSLAW  :
	case COSINELAW_QS:
		/* Launch single-threaded wrapper kernel for cosinelaw_qs */
		rad_cosinelaw_qs_krnl<<<1,1>>>(dpar, dmod, albedo_mode,
				radalb_factor, ilaw);
		checkErrorAfterKernelLaunch("rad_cosinelaw_qs_krnl");
		break;
	case GAUSSIAN_COSINE:
	case HAGFORS_COSINE :
	case COSINE_COSINE  :
		/* Launch single-threaded wrapper kernel for cosinelaw_qs */
		rad_cosine_cosine_krnl<<<1,1>>>(dpar, dmod, albedo_mode,
				radalb_factor, ilaw);
		checkErrorAfterKernelLaunch("rad_cosinelaw_qs_krnl");
		break;
	case HARMCOSINE_DIFF:
		/* Launch single-threaded kernel to get L and Lmax */
		rad_harmcosine_diff_getL_krnl<<<1,1>>>(dmod, ilaw);
		checkErrorAfterKernelLaunch("rad_harmcosine_diff_getL_krnl");
		gpuErrchk(cudaMemcpyFromSymbol(&L, dL, sizeof(dL), 0,
				cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpyFromSymbol(&Lmax, dLmax, sizeof(dLmax), 0,
				cudaMemcpyDeviceToHost));

		/* Launch L-threaded kernel to set R a,b, a_save, b_save */
		THD1.x = L;
		rad_harmcosine_diff_set_ab_krnl<<<1,THD>>>(dmod, albedo_mode,
				radalb_factor, L, ilaw);
		checkErrorAfterKernelLaunch("rad_harmcosine_diff_set_ab_krnl");

		/* Copy Lmax (for sizing arrays) from device to host  then allocate double pntrs */
		gpuErrchk(cudaMalloc((void**)&nlm, 	   sizeof(double*)*(Lmax+1)));
		gpuErrchk(cudaMalloc((void**)&afactor, sizeof(double*)*(Lmax+1)));
		gpuErrchk(cudaMalloc((void**)&bfactor, sizeof(double*)* Lmax));

		for (i=0; i<=Lmax; i++) {
			gpuErrchk(cudaMalloc((void**)&nlm[i], sizeof(double)*(Lmax+1)));
			gpuErrchk(cudaMalloc((void**)&afactor[i], sizeof(double)*(Lmax+1)));
			if (i<Lmax && Lmax >0)
				gpuErrchk(cudaMalloc((void**)&bfactor[i], sizeof(double)*Lmax));
		}
		/* End setup of matrices */

		/* Launch nlm population kernel */
		THD1.x = Lmax;
		opt_harmhapke_kaas_nlm_krnl<<<1,THD>>>(dmod, nlm, Lmax);
		checkErrorAfterKernelLaunch("opt_harmhapke_kaas_nlm_krnl");

		rad_harmcosine_facet_krnl<<<BLK,THD>>>(dpar, dmod, nlm, afactor,
				bfactor, nf, Lmax, ilaw);
		checkErrorAfterKernelLaunch("rad_harmcosine_facet_krnl");

		cudaFree(nlm);
		cudaFree(afactor);
		if (Lmax > 0)	cudaFree(bfactor);
		break;
	case INHOCOSINE_DIFF:
		/* Launch single-threaded kernel to set R.a, R.b, R.a_save, R.b_save
		 * and also nf */
		rad_inhocosine_set_ab_krnl<<<1,1>>>(dpar, dmod, albedo_mode,
				radalb_factor, ilaw);
		checkErrorAfterKernelLaunch("rad_inhocosine_set_ab_krnl");

		rad_inhocosine_facet_krnl<<<BLK,THD>>>(dpar, dmod, nf,
				albedo_mode, radalb_factor, ilaw);
		checkErrorAfterKernelLaunch("rad_inhocosine_facet_krnl");
		break;
	default:
		bailout("realize_photo: can't handle this radar law yet\n");
	}
 	 }
}
