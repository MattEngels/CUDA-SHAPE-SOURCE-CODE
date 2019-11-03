/*****************************************************************************************
                                                                              mkparlist.c

Takes mod_t and dat_t files and makes a list of pointers to the free parameters.
Modified 2014 Aug 22 by SN:
    Add spin.lib_amp, spin.lib_freq, spin.lib_phase for taking into account librations.

Modified 2014 February 12 by CM:
    Implement multiple radar and optical scattering laws

Modified 2013 May 20 by CM:
    Implement ovoid shape components

Modified 2012 March 23 by CM:
    Add the "dopscale" parameter for delay-Doppler and Doppler datasets

Modified 2012 March 14 by CM:
    Have separate code blocks for delay-Doppler vs. Doppler datasets when handling the
        delay correction polynomial coefficients

Modified 2011 September 2 by CM:
    Add the "harmlambert" and "inholambert" optical scattering laws

Modified 2011 August 7 by CM:
    Add spin impulses

Modified 2010 June 1 by CM:
    Change "scalefactor" parameter from a scalar to a 3-component vector
        for harmonic and vertex shape structures

Modified 2010 April 27 by CM:
    Add "tabular" radar scattering law

Modified 2008 August 10 by CM:
    Change "delcor_step" and "delcor_abstol" parameters to be vectors
        rather than scalars
    Change parameter type for each component's linear offsets in the
        mod file from SHAPEPAR to SIZEPAR

Modified 2006 October 1 by CM:
    Replace ellipsoid diameters D with two_a, a_over_b, and b_over_c
    Add "scalefactor" to harmonic and vertex shape structures
    Add "ratio_step" "ratio_tol" and "ratio_abstol" fit parameters
    Add SIZEPAR parameters

Modified 2006 March 6 by PT:
    Add "spin.omegadot" parameter for changing spin rate

Modified 2005 September 7 by CM:
    Add "harmlommel" "harmhapke" and "harmkaas" optical scattering laws
    Add "harmcosine" radar scattering law

Modified 2005 August 1 by CM:
    Add "inhokaas" optical scattering law

Modified 2005 July 20 by CM:
    Add "hagfors" and "cosine_qs" and "gauss+cosine" and "hagfors+cosine"
        and "cosine+cosine" and inhomogeneous "inhocosine" radar
        scattering laws
    Eliminate "flat" radar scattering law

Modified 2005 July 4 by CM:
    Add inhomogeneous "inholommel" and "inhohapke" optical scattering laws

Modified 2005 February 24 by CM:
    Add POS case for data parameters so that the horizontal and vertical
        offsets for plane-of-sky datasets can be fitted

Modified 2004 August 13 by CM:
    Assign "fparabstol" pointers to handle absolute fitting tolerances

Modified 2004 April 29 by CM:
    For Kaasalainen scattering law, switch from weighting factor "c"
        (ranging from 0 to infinity) to "wt" (ranging from 0 to 1)

Modified 2004 March 13 by CM:
    Add code to handle harmonic components

Modified 2004 February 25 by CM:
    Add Kaasalainen "Lambert + Lommel-Seeliger" scattering law parameters

Modified 2003 April 3 by CM:
    1) Fit ellipsoid diameters using length_step and length_tol
       rather than angle_step and angle_tol
    2) Fit photometric parameters using photo_step and photo_tol
       rather than length_step and length_tol
    3) Fit moments of inertia using inertia_step and inertia_tol
       rather than spin_step and spin_tol
    4) Fit delay correction polynomial coefficients using
       delcor_step and delcor_tol rather than spin_step and spin_tol
 *****************************************************************************************/

#include "head.h"

void mkparlist( struct par_t *par, struct mod_t *mod, struct dat_t *dat)
{
	int i, j, k, ilaw, p=(-1), s, c, f, l, m;

	/*  Allocate memory for pointers, steps, and tolerances  */

	par->fparstep = vector( 0, par->nfpar-1);
	par->fpartol = vector( 0, par->nfpar-1);
	par->fparabstol = vector( 0, par->nfpar-1);
	par->fpntr = (double **) calloc( par->nfpar, sizeof( double *));
	par->fpartype = ivector( 0, par->nfpar-1);

	/*  Shape parameters  */

	for (i=0; i<mod->shape.ncomp; i++) { /* read each component */
		for (j=0; j<=2; j++) {                                /* check linear offsets */
			if (mod->shape.comp[i].off[j].state == 'f') {
				par->fpntr[++p] = &mod->shape.comp[i].off[j].val;
				par->fparstep[p] = par->length_step;
				par->fpartol[p] = par->length_tol;
				par->fparabstol[p] = par->length_abstol;
				par->fpartype[p] = SIZEPAR;
			}
		}
		for (j=0; j<=2; j++) {                                /* check angular offsets */
			if (mod->shape.comp[i].rot[j].state == 'f') {
				par->fpntr[++p] = &mod->shape.comp[i].rot[j].val;
				par->fparstep[p] = par->angle_step;
				par->fpartol[p] = par->angle_tol;
				par->fparabstol[p] = par->angle_abstol;
				par->fpartype[p] = SHAPEPAR;
			}
		}

		switch (mod->shape.comp[i].type) {
		case ELLIPSE:
			if (mod->shape.comp[i].desc.ell.two_a.state == 'f') {
				par->fpntr[++p] = &mod->shape.comp[i].desc.ell.two_a.val;
				par->fparstep[p] = par->length_step;
				par->fpartol[p] = par->length_tol;
				par->fparabstol[p] = par->length_abstol;
				par->fpartype[p] = SIZEPAR;
			}
			if (mod->shape.comp[i].desc.ell.a_over_b.state == 'f') {
				par->fpntr[++p] = &mod->shape.comp[i].desc.ell.a_over_b.val;
				par->fparstep[p] = par->ratio_step;
				par->fpartol[p] = par->ratio_tol;
				par->fparabstol[p] = par->ratio_abstol;
				par->fpartype[p] = SHAPEPAR;
			}
			if (mod->shape.comp[i].desc.ell.b_over_c.state == 'f') {
				par->fpntr[++p] = &mod->shape.comp[i].desc.ell.b_over_c.val;
				par->fparstep[p] = par->ratio_step;
				par->fpartol[p] = par->ratio_tol;
				par->fparabstol[p] = par->ratio_abstol;
				par->fpartype[p] = SHAPEPAR;
			}
			break;
		case OVOID:
			if (mod->shape.comp[i].desc.ovoid.two_a.state == 'f') {
				par->fpntr[++p] = &mod->shape.comp[i].desc.ovoid.two_a.val;
				par->fparstep[p] = par->length_step;
				par->fpartol[p] = par->length_tol;
				par->fparabstol[p] = par->length_abstol;
				par->fpartype[p] = SIZEPAR;
			}
			if (mod->shape.comp[i].desc.ovoid.a_over_b.state == 'f') {
				par->fpntr[++p] = &mod->shape.comp[i].desc.ovoid.a_over_b.val;
				par->fparstep[p] = par->ratio_step;
				par->fpartol[p] = par->ratio_tol;
				par->fparabstol[p] = par->ratio_abstol;
				par->fpartype[p] = SHAPEPAR;
			}
			if (mod->shape.comp[i].desc.ovoid.b_over_c.state == 'f') {
				par->fpntr[++p] = &mod->shape.comp[i].desc.ovoid.b_over_c.val;
				par->fparstep[p] = par->ratio_step;
				par->fpartol[p] = par->ratio_tol;
				par->fparabstol[p] = par->ratio_abstol;
				par->fpartype[p] = SHAPEPAR;
			}
			if (mod->shape.comp[i].desc.ovoid.k.state == 'f') {
				par->fpntr[++p] = &mod->shape.comp[i].desc.ovoid.k.val;
				par->fparstep[p] = par->ratio_step;
				par->fpartol[p] = par->ratio_tol;
				par->fparabstol[p] = par->ratio_abstol;
				par->fpartype[p] = SHAPEPAR;
			}
			break;
		case HARMONIC:
			for (j=0; j<=2; j++)
				if (mod->shape.comp[i].desc.har.scalefactor[j].state == 'f') {
					par->fpntr[++p] = &mod->shape.comp[i].desc.har.scalefactor[j].val;
					par->fparstep[p] = par->ratio_step;
					par->fpartol[p] = par->ratio_tol;
					par->fparabstol[p] = par->ratio_abstol;
					par->fpartype[p] = SIZEPAR;
				}
			for (j=0; j<=mod->shape.comp[i].desc.har.nhar; j++) {
				if (mod->shape.comp[i].desc.har.a[j][0].state == 'f') {
					par->fpntr[++p] = &mod->shape.comp[i].desc.har.a[j][0].val;
					par->fparstep[p] = par->length_step;
					par->fpartol[p] = par->length_tol;
					par->fparabstol[p] = par->length_abstol;
					par->fpartype[p] = SHAPEPAR;
				}
				for (k=1; k<=j; k++) {
					if (mod->shape.comp[i].desc.har.a[j][k].state == 'f') {
						par->fpntr[++p] = &mod->shape.comp[i].desc.har.a[j][k].val;
						par->fparstep[p] = par->length_step;
						par->fpartol[p] = par->length_tol;
						par->fparabstol[p] = par->length_abstol;
						par->fpartype[p] = SHAPEPAR;
					}
					if (mod->shape.comp[i].desc.har.b[j][k].state == 'f') {
						par->fpntr[++p] = &mod->shape.comp[i].desc.har.b[j][k].val;
						par->fparstep[p] = par->length_step;
						par->fpartol[p] = par->length_tol;
						par->fparabstol[p] = par->length_abstol;
						par->fpartype[p] = SHAPEPAR;
					}
				}
			}
			break;
		case VERTEX:
			for (j=0; j<=2; j++)
				if (mod->shape.comp[i].desc.ver.scalefactor[j].state == 'f') {
					par->fpntr[++p] = &mod->shape.comp[i].desc.ver.scalefactor[j].val;
					par->fparstep[p] = par->ratio_step;
					par->fpartol[p] = par->ratio_tol;
					par->fparabstol[p] = par->ratio_abstol;
					par->fpartype[p] = SIZEPAR;
				}
			for (j=0; j<mod->shape.comp[i].desc.ver.nv; j++) {
				if (mod->shape.comp[i].desc.ver.v[j].r.state == 'f') {
					par->fpntr[++p] = &mod->shape.comp[i].desc.ver.v[j].r.val;
					par->fparstep[p] = par->length_step;
					par->fpartol[p] = par->length_tol;
					par->fparabstol[p] = par->length_abstol;
					par->fpartype[p] = SHAPEPAR;
				}
			}
			break;
		default:
			bailout("mkparlist.c: can't do that type of model yet\n");
		}
	}

	/*  Photometric parameters  */
	for (ilaw=0; ilaw<mod->photo.nradlaws; ilaw++) {
		switch (mod->photo.radtype[ilaw]) {
		case COSINELAW_DIFF:
			if (mod->photo.radar[ilaw].RC.R.state == 'f') {
				par->fpntr[++p] = &mod->photo.radar[ilaw].RC.R.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			if (mod->photo.radar[ilaw].RC.C.state == 'f') {
				par->fpntr[++p] = &mod->photo.radar[ilaw].RC.C.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			break;
		case TABULARLAW:
			for (i=0; i<mod->photo.radar[ilaw].tabular.n; i++) {
				if (mod->photo.radar[ilaw].tabular.rho[i].state == 'f') {
					par->fpntr[++p] = &mod->photo.radar[ilaw].tabular.rho[i].val;
					par->fparstep[p] = par->photo_step;
					par->fpartol[p] = par->photo_tol;
					par->fparabstol[p] = par->photo_abstol;
					par->fpartype[p] = PHOTOPAR;
				}
			}
			break;
		case HARMCOSINE_DIFF:
			for (l=0; l<=mod->photo.radar[ilaw].harmcosine.R.nhar; l++)
				for (m=0; m<=l; m++) {
					if (mod->photo.radar[ilaw].harmcosine.R.a[l][m].state == 'f') {
						par->fpntr[++p] = &mod->photo.radar[ilaw].harmcosine.R.a[l][m].val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
					if (m > 0 && mod->photo.radar[ilaw].harmcosine.R.b[l][m].state == 'f') {
						par->fpntr[++p] = &mod->photo.radar[ilaw].harmcosine.R.b[l][m].val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
				}
			for (l=0; l<=mod->photo.radar[ilaw].harmcosine.C.nhar; l++)
				for (m=0; m<=l; m++) {
					if (mod->photo.radar[ilaw].harmcosine.C.a[l][m].state == 'f') {
						par->fpntr[++p] = &mod->photo.radar[ilaw].harmcosine.C.a[l][m].val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
					if (m > 0 && mod->photo.radar[ilaw].harmcosine.C.b[l][m].state == 'f') {
						par->fpntr[++p] = &mod->photo.radar[ilaw].harmcosine.C.b[l][m].val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
				}
			break;
		case GAUSSIANLAW :
		case HAGFORSLAW  :
		case COSINELAW_QS:
			if (mod->photo.radar[ilaw].quasispec.R.state == 'f') {
				par->fpntr[++p] = &mod->photo.radar[ilaw].quasispec.R.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			if (mod->photo.radar[ilaw].quasispec.C.state == 'f') {
				par->fpntr[++p] = &mod->photo.radar[ilaw].quasispec.C.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			break;
		case GAUSSIAN_COSINE:
		case HAGFORS_COSINE :
		case COSINE_COSINE  :
			if (mod->photo.radar[ilaw].hybrid.qs.R.state == 'f') {
				par->fpntr[++p] = &mod->photo.radar[ilaw].hybrid.qs.R.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			if (mod->photo.radar[ilaw].hybrid.qs.C.state == 'f') {
				par->fpntr[++p] = &mod->photo.radar[ilaw].hybrid.qs.C.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			if (mod->photo.radar[ilaw].hybrid.diff.R.state == 'f') {
				par->fpntr[++p] = &mod->photo.radar[ilaw].hybrid.diff.R.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			if (mod->photo.radar[ilaw].hybrid.diff.C.state == 'f') {
				par->fpntr[++p] = &mod->photo.radar[ilaw].hybrid.diff.C.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			break;
		case INHOCOSINE_DIFF:
			if (mod->photo.radar[ilaw].inhocosine.global.R.state == 'f') {
				par->fpntr[++p] = &mod->photo.radar[ilaw].inhocosine.global.R.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			if (mod->photo.radar[ilaw].inhocosine.global.C.state == 'f') {
				par->fpntr[++p] = &mod->photo.radar[ilaw].inhocosine.global.C.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			for (c=0; c<mod->shape.ncomp; c++)
				for (f=0; f<mod->shape.comp[c].real.nf; f++) {
					if (mod->photo.radar[ilaw].inhocosine.local[c][f].R.state == 'f') {
						par->fpntr[++p] = &mod->photo.radar[ilaw].inhocosine.local[c][f].R.val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
					if (mod->photo.radar[ilaw].inhocosine.local[c][f].C.state == 'f') {
						par->fpntr[++p] = &mod->photo.radar[ilaw].inhocosine.local[c][f].C.val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
				}
			break;
		case NOLAW:
			break;
		default:
			bailout("mkparlist.c: can't do that radar law yet\n");
		}
	}

	for (ilaw=0; ilaw<mod->photo.noptlaws; ilaw++) {
		switch (mod->photo.opttype[ilaw]) {
		case NOLAW:
			break;
		case GEOMETRICAL:
		case LAMBERTLAW:
		case LOMMEL:
			if (mod->photo.optical[ilaw].R.R.state == 'f') {
				par->fpntr[++p] = &mod->photo.optical[ilaw].R.R.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			break;
		case HARMLAMBERT:
		case HARMLOMMEL:
			for (l=0; l<=mod->photo.optical[ilaw].harmR.R.nhar; l++)
				for (m=0; m<=l; m++) {
					if (mod->photo.optical[ilaw].harmR.R.a[l][m].state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].harmR.R.a[l][m].val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
					if (m > 0 && mod->photo.optical[ilaw].harmR.R.b[l][m].state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].harmR.R.b[l][m].val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
				}
			break;
		case INHOLAMBERT:
		case INHOLOMMEL:
			if (mod->photo.optical[ilaw].inhoR.global.R.state == 'f') {
				par->fpntr[++p] = &mod->photo.optical[ilaw].inhoR.global.R.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			for (c=0; c<mod->shape.ncomp; c++)
				for (f=0; f<mod->shape.comp[c].real.nf; f++)
					if (mod->photo.optical[ilaw].inhoR.local[c][f].R.state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].inhoR.local[c][f].R.val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
			break;
		case HAPKE:
			if (mod->photo.optical[ilaw].hapke.w.state == 'f') {
				par->fpntr[++p] = &mod->photo.optical[ilaw].hapke.w.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			if (mod->photo.optical[ilaw].hapke.h.state == 'f') {
				par->fpntr[++p] = &mod->photo.optical[ilaw].hapke.h.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			if (mod->photo.optical[ilaw].hapke.B0.state == 'f') {
				par->fpntr[++p] = &mod->photo.optical[ilaw].hapke.B0.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			if (mod->photo.optical[ilaw].hapke.g.state == 'f') {
				par->fpntr[++p] = &mod->photo.optical[ilaw].hapke.g.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			if (mod->photo.optical[ilaw].hapke.theta.state == 'f') {
				par->fpntr[++p] = &mod->photo.optical[ilaw].hapke.theta.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			break;
		case HARMHAPKE:
			for (l=0; l<=mod->photo.optical[ilaw].harmhapke.w.nhar; l++)
				for (m=0; m<=l; m++) {
					if (mod->photo.optical[ilaw].harmhapke.w.a[l][m].state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].harmhapke.w.a[l][m].val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
					if (m > 0 && mod->photo.optical[ilaw].harmhapke.w.b[l][m].state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].harmhapke.w.b[l][m].val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
				}
			for (l=0; l<=mod->photo.optical[ilaw].harmhapke.h.nhar; l++)
				for (m=0; m<=l; m++) {
					if (mod->photo.optical[ilaw].harmhapke.h.a[l][m].state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].harmhapke.h.a[l][m].val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
					if (m > 0 && mod->photo.optical[ilaw].harmhapke.h.b[l][m].state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].harmhapke.h.b[l][m].val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
				}
			for (l=0; l<=mod->photo.optical[ilaw].harmhapke.B0.nhar; l++)
				for (m=0; m<=l; m++) {
					if (mod->photo.optical[ilaw].harmhapke.B0.a[l][m].state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].harmhapke.B0.a[l][m].val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
					if (m > 0 && mod->photo.optical[ilaw].harmhapke.B0.b[l][m].state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].harmhapke.B0.b[l][m].val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
				}
			for (l=0; l<=mod->photo.optical[ilaw].harmhapke.g.nhar; l++)
				for (m=0; m<=l; m++) {
					if (mod->photo.optical[ilaw].harmhapke.g.a[l][m].state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].harmhapke.g.a[l][m].val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
					if (m > 0 && mod->photo.optical[ilaw].harmhapke.g.b[l][m].state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].harmhapke.g.b[l][m].val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
				}
			for (l=0; l<=mod->photo.optical[ilaw].harmhapke.theta.nhar; l++)
				for (m=0; m<=l; m++) {
					if (mod->photo.optical[ilaw].harmhapke.theta.a[l][m].state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].harmhapke.theta.a[l][m].val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
					if (m > 0 && mod->photo.optical[ilaw].harmhapke.theta.b[l][m].state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].harmhapke.theta.b[l][m].val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
				}
			break;
		case INHOHAPKE:
			if (mod->photo.optical[ilaw].inhohapke.global.w.state == 'f') {
				par->fpntr[++p] = &mod->photo.optical[ilaw].inhohapke.global.w.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			if (mod->photo.optical[ilaw].inhohapke.global.h.state == 'f') {
				par->fpntr[++p] = &mod->photo.optical[ilaw].inhohapke.global.h.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			if (mod->photo.optical[ilaw].inhohapke.global.B0.state == 'f') {
				par->fpntr[++p] = &mod->photo.optical[ilaw].inhohapke.global.B0.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			if (mod->photo.optical[ilaw].inhohapke.global.g.state == 'f') {
				par->fpntr[++p] = &mod->photo.optical[ilaw].inhohapke.global.g.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			if (mod->photo.optical[ilaw].inhohapke.global.theta.state == 'f') {
				par->fpntr[++p] = &mod->photo.optical[ilaw].inhohapke.global.theta.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			for (c=0; c<mod->shape.ncomp; c++)
				for (f=0; f<mod->shape.comp[c].real.nf; f++) {
					if (mod->photo.optical[ilaw].inhohapke.local[c][f].w.state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].inhohapke.local[c][f].w.val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
					if (mod->photo.optical[ilaw].inhohapke.local[c][f].h.state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].inhohapke.local[c][f].h.val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
					if (mod->photo.optical[ilaw].inhohapke.local[c][f].B0.state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].inhohapke.local[c][f].B0.val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
					if (mod->photo.optical[ilaw].inhohapke.local[c][f].g.state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].inhohapke.local[c][f].g.val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
					if (mod->photo.optical[ilaw].inhohapke.local[c][f].theta.state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].inhohapke.local[c][f].theta.val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
				}
			break;
		case KAASALAINEN:
			if (mod->photo.optical[ilaw].kaas.R.state == 'f') {
				par->fpntr[++p] = &mod->photo.optical[ilaw].kaas.R.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			if (mod->photo.optical[ilaw].kaas.wt.state == 'f') {
				par->fpntr[++p] = &mod->photo.optical[ilaw].kaas.wt.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			if (mod->photo.optical[ilaw].kaas.A0.state == 'f') {
				par->fpntr[++p] = &mod->photo.optical[ilaw].kaas.A0.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			if (mod->photo.optical[ilaw].kaas.D.state == 'f') {
				par->fpntr[++p] = &mod->photo.optical[ilaw].kaas.D.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			if (mod->photo.optical[ilaw].kaas.k.state == 'f') {
				par->fpntr[++p] = &mod->photo.optical[ilaw].kaas.k.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			break;
		case HARMKAAS:
			for (l=0; l<=mod->photo.optical[ilaw].harmkaas.R.nhar; l++)
				for (m=0; m<=l; m++) {
					if (mod->photo.optical[ilaw].harmkaas.R.a[l][m].state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].harmkaas.R.a[l][m].val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
					if (m > 0 && mod->photo.optical[ilaw].harmkaas.R.b[l][m].state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].harmkaas.R.b[l][m].val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
				}
			for (l=0; l<=mod->photo.optical[ilaw].harmkaas.wt.nhar; l++)
				for (m=0; m<=l; m++) {
					if (mod->photo.optical[ilaw].harmkaas.wt.a[l][m].state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].harmkaas.wt.a[l][m].val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
					if (m > 0 && mod->photo.optical[ilaw].harmkaas.wt.b[l][m].state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].harmkaas.wt.b[l][m].val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
				}
			for (l=0; l<=mod->photo.optical[ilaw].harmkaas.A0.nhar; l++)
				for (m=0; m<=l; m++) {
					if (mod->photo.optical[ilaw].harmkaas.A0.a[l][m].state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].harmkaas.A0.a[l][m].val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
					if (m > 0 && mod->photo.optical[ilaw].harmkaas.A0.b[l][m].state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].harmkaas.A0.b[l][m].val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
				}
			for (l=0; l<=mod->photo.optical[ilaw].harmkaas.D.nhar; l++)
				for (m=0; m<=l; m++) {
					if (mod->photo.optical[ilaw].harmkaas.D.a[l][m].state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].harmkaas.D.a[l][m].val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
					if (m > 0 && mod->photo.optical[ilaw].harmkaas.D.b[l][m].state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].harmkaas.D.b[l][m].val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
				}
			for (l=0; l<=mod->photo.optical[ilaw].harmkaas.k.nhar; l++)
				for (m=0; m<=l; m++) {
					if (mod->photo.optical[ilaw].harmkaas.k.a[l][m].state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].harmkaas.k.a[l][m].val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
					if (m > 0 && mod->photo.optical[ilaw].harmkaas.k.b[l][m].state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].harmkaas.k.b[l][m].val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
				}
			break;
		case INHOKAAS:
			if (mod->photo.optical[ilaw].inhokaas.global.R.state == 'f') {
				par->fpntr[++p] = &mod->photo.optical[ilaw].inhokaas.global.R.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			if (mod->photo.optical[ilaw].inhokaas.global.wt.state == 'f') {
				par->fpntr[++p] = &mod->photo.optical[ilaw].inhokaas.global.wt.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			if (mod->photo.optical[ilaw].inhokaas.global.A0.state == 'f') {
				par->fpntr[++p] = &mod->photo.optical[ilaw].inhokaas.global.A0.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			if (mod->photo.optical[ilaw].inhokaas.global.D.state == 'f') {
				par->fpntr[++p] = &mod->photo.optical[ilaw].inhokaas.global.D.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			if (mod->photo.optical[ilaw].inhokaas.global.k.state == 'f') {
				par->fpntr[++p] = &mod->photo.optical[ilaw].inhokaas.global.k.val;
				par->fparstep[p] = par->photo_step;
				par->fpartol[p] = par->photo_tol;
				par->fparabstol[p] = par->photo_abstol;
				par->fpartype[p] = PHOTOPAR;
			}
			for (c=0; c<mod->shape.ncomp; c++)
				for (f=0; f<mod->shape.comp[c].real.nf; f++) {
					if (mod->photo.optical[ilaw].inhokaas.local[c][f].R.state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].inhokaas.local[c][f].R.val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
					if (mod->photo.optical[ilaw].inhokaas.local[c][f].wt.state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].inhokaas.local[c][f].wt.val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
					if (mod->photo.optical[ilaw].inhokaas.local[c][f].A0.state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].inhokaas.local[c][f].A0.val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
					if (mod->photo.optical[ilaw].inhokaas.local[c][f].D.state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].inhokaas.local[c][f].D.val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
					if (mod->photo.optical[ilaw].inhokaas.local[c][f].k.state == 'f') {
						par->fpntr[++p] = &mod->photo.optical[ilaw].inhokaas.local[c][f].k.val;
						par->fparstep[p] = par->photo_step;
						par->fpartol[p] = par->photo_tol;
						par->fparabstol[p] = par->photo_abstol;
						par->fpartype[p] = PHOTOPAR;
					}
				}
			break;
		default:
			bailout("mkparlist.c: can't do that optical law yet\n");
		}
	}

	/*  Spin parameters  */

	for (i=0; i<=2; i++) {
		if (mod->spin.angle[i].state == 'f') {
			par->fpntr[++p] = &mod->spin.angle[i].val;
			par->fparstep[p] = par->angle_step;
			par->fpartol[p] = par->angle_tol;
			par->fparabstol[p] = par->angle_abstol;
			par->fpartype[p] = SPINPAR;
		}
	}
	for (i=0; i<=2; i++) {
		if (mod->spin.omega[i].state == 'f') {
			par->fpntr[++p] = &mod->spin.omega[i].val;
			par->fparstep[p] = par->spin_step;
			par->fpartol[p] = par->spin_tol;
			par->fparabstol[p] = par->spin_abstol;
			par->fpartype[p] = SPINPAR;
		}
	}
	for (i=0; i<=2; i++) {
		if (mod->spin.inertia[i].state == 'f') {
			par->fpntr[++p] = &mod->spin.inertia[i].val;
			par->fparstep[p] = par->inertia_step;
			par->fpartol[p] = par->inertia_tol;
			par->fparabstol[p] = par->inertia_abstol;
			par->fpartype[p] = SPINPAR;
		}
	}
	for (i=0; i<=2; i++) {
		if (mod->spin.omegadot[i].state == 'f') {
			par->fpntr[++p] = &mod->spin.omegadot[i].val;
			par->fparstep[p] = par->spindot_step;
			par->fpartol[p] = par->spindot_tol;
			par->fparabstol[p] = par->spindot_abstol;
			par->fpartype[p] = SPINPAR;
		}
	}
	for (k=0; k<mod->spin.n_impulse; k++)
		for (i=0; i<=2; i++) {
			if (mod->spin.impulse[k][i].state == 'f') {
				par->fpntr[++p] = &mod->spin.impulse[k][i].val;
				par->fparstep[p] = par->spin_step;
				par->fpartol[p] = par->spin_tol;
				par->fparabstol[p] = par->spin_abstol;
				par->fpartype[p] = SPINPAR;
			}
		}

	if (mod->spin.lib_amp.state == 'f') {
		par->fpntr[++p] = &mod->spin.lib_amp.val;
		par->fparstep[p] = par->lib_amp_step;
		par->fpartol[p] = par->lib_amp_tol;
		par->fparabstol[p] = par->lib_amp_abstol;
		par->fpartype[p] = SPINPAR;
	}

	if (mod->spin.lib_freq.state == 'f') {
		par->fpntr[++p] = &mod->spin.lib_freq.val;
		par->fparstep[p] = par->lib_freq_step;
		par->fpartol[p] = par->lib_freq_tol;
		par->fparabstol[p] = par->lib_freq_abstol;
		par->fpartype[p] = SPINPAR;
	}

	if (mod->spin.lib_phase.state == 'f') {
		par->fpntr[++p] = &mod->spin.lib_phase.val;
		par->fparstep[p] = par->lib_phase_step;
		par->fpartol[p] = par->lib_phase_tol;
		par->fparabstol[p] = par->lib_phase_abstol;
		par->fpartype[p] = SPINPAR;
	}
	/*  Data parameters (i.e., those in the obs file, other than
      the "calfact" parameters which are computed analytically)  */

	for (s=0; s<dat->nsets; s++) {
		for (i=0; i<=2; i++) {
			if (dat->set[s].angleoff[i].state == 'f') {
				par->fpntr[++p] = &dat->set[s].angleoff[i].val;
				par->fparstep[p] = par->angle_step;
				par->fpartol[p] = par->angle_tol;
				par->fparabstol[p] = par->angle_abstol;
				par->fpartype[p] = SPINPAR;
			}
		}
		for (i=0; i<=2; i++) {
			if (dat->set[s].omegaoff[i].state == 'f') {
				par->fpntr[++p] = &dat->set[s].omegaoff[i].val;
				par->fparstep[p] = par->spin_step;
				par->fpartol[p] = par->spin_tol;
				par->fparabstol[p] = par->spin_abstol;
				par->fpartype[p] = SPINPAR;
			}
		}

		switch (dat->set[s].type) {
		case DELAY:
			for (i=0; i<=dat->set[s].desc.deldop.delcor.n; i++) {
				if (dat->set[s].desc.deldop.delcor.a[i].state == 'f') {
					j = (i < MAXDELCORPAR) ? i : 0;
					par->fpntr[++p] = &dat->set[s].desc.deldop.delcor.a[i].val;
					par->fparstep[p] = par->delcor_step[j];
					par->fpartol[p] = par->delcor_tol;
					par->fparabstol[p] = par->delcor_abstol[j];
					par->fpartype[p] = DELCORPAR;
				}
			}
			if (dat->set[s].desc.deldop.dopscale.state == 'f') {
				par->fpntr[++p] = &dat->set[s].desc.deldop.dopscale.val;
				par->fparstep[p] = par->ratio_step;
				par->fpartol[p] = par->ratio_tol;
				par->fparabstol[p] = par->ratio_abstol;
				par->fpartype[p] = DOPSCALEPAR;
			}
			break;
		case DOPPLER:
			for (i=0; i<=dat->set[s].desc.doppler.delcor.n; i++) {
				if (dat->set[s].desc.doppler.delcor.a[i].state == 'f') {
					j = (i < MAXDELCORPAR) ? i : 0;
					par->fpntr[++p] = &dat->set[s].desc.doppler.delcor.a[i].val;
					par->fparstep[p] = par->delcor_step[j];
					par->fpartol[p] = par->delcor_tol;
					par->fparabstol[p] = par->delcor_abstol[j];
					par->fpartype[p] = DELCORPAR;
				}
			}
			if (dat->set[s].desc.doppler.dopscale.state == 'f') {
				par->fpntr[++p] = &dat->set[s].desc.doppler.dopscale.val;
				par->fparstep[p] = par->ratio_step;
				par->fpartol[p] = par->ratio_tol;
				par->fparabstol[p] = par->ratio_abstol;
				par->fpartype[p] = DOPSCALEPAR;
			}
			break;
		case POS:
			for (i=0; i<dat->set[s].desc.poset.nframes; i++) {
				for (j=0; j<=1; j++) {
					if (dat->set[s].desc.poset.frame[i].off[j].state == 'f') {
						par->fpntr[++p] = &dat->set[s].desc.poset.frame[i].off[j].val;
						par->fparstep[p] = par->xyoff_step;
						par->fpartol[p] = par->xyoff_tol;
						par->fparabstol[p] = par->xyoff_abstol;
						par->fpartype[p] = XYOFFPAR;
					}
				}
			}
			break;
		case LGHTCRV:
			break;
		default:
			bailout("mkparlist.c: can't handle that type of data yet\n");
		}
	}
}
