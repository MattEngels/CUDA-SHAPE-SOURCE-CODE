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

#include "../shape/head.h"

void vary_params( struct par_t *par, struct mod_t *mod, struct dat_t *dat,
		int action, double *deldop_zmax, double *rad_xsec,
		double *opt_brightness, double *cos_subradarlat)
{
	double orbit_offset[3] = {0.0, 0.0, 0.0};
	int c, f, s, i, j, k, x, y, compute_xsec, compute_brightness, compute_zmax,
	compute_cosdelta, n, ncalc, b;
	double sum_deldop_zmax, sum_rad_xsec, sum_opt_brightness, sum_cos_subradarlat, weight,
	zmax, cross_section, oa[3][3], to_earth[3], cos_delta, intensityfactor;
	struct deldop_t *deldop;
	struct doppler_t *doppler;
	struct lghtcrv_t *lghtcrv;
	struct pos_t *pos;

	/*  Initialize variables  */
	sum_deldop_zmax = 0.0;
	sum_rad_xsec = 0.0;
	sum_opt_brightness = 0.0;
	sum_cos_subradarlat = 0.0;

	/*  Process each dataset in turn  */
	for (s=0; s<dat->nsets; s++) {

		switch (dat->set[s].type) {
		case DELAY:

			deldop = &dat->set[s].desc.deldop;

			for (f=0; f<deldop->nframes; f++) {
				compute_zmax = (par->vary_delcor0 != VARY_NONE
						&& deldop->delcor.a[0].state != 'c');
				compute_xsec = (par->vary_radalb != VARY_NONE
						&& deldop->frame[f].cal.state == 'c');
				if (compute_zmax || compute_xsec) {
					weight = deldop->frame[f].weight;
					pos = &deldop->frame[f].pos;
					for (i=0; i<=2; i++)
						for (j=0; j<=2; j++) {
							pos->ae[i][j] = deldop->frame[f].view[deldop->v0].ae[i][j];
							pos->oe[i][j] = deldop->frame[f].view[deldop->v0].oe[i][j];
						}
					pos->bistatic = 0;

					/* Initialize the plane-of-sky view  */
					posclr( pos);

					/* Determine which POS pixels cover the target, and get the
					 * distance toward Earth of each POS pixel   */
					for (c=0; c<mod->shape.ncomp; c++){
						posvis( &mod->shape.comp[c].real, orbit_offset, pos,
								(int) par->pos_smooth, 0, 0, c);
					}

					/* Zero out the fit delay-Doppler image and call pos2deldop
					 * to create the fit image by mapping power from the plane
					 * of sky to delay-Doppler space.                             */

					clrmat( deldop->frame[f].fit, 1, deldop->frame[f].ndel,
							1, deldop->frame[f].ndop);

					pos2deldop(par, &mod->photo, 0.0, 0.0, 0.0, deldop, 0, s, f, 0);

					/*  Compute distance toward Earth of the subradar point  */
					if (compute_zmax) {
						zmax = -HUGENUMBER;
						for (x=pos->xlim[0]; x<=pos->xlim[1]; x++)
							for (y=pos->ylim[0]; y<=pos->ylim[1]; y++)
								if (pos->cose[x][y] > 0.0)
									zmax = MAX( zmax, pos->z[x][y]);
						sum_deldop_zmax += zmax*weight;
					}

					/*  Compute cross section  */

					if (compute_xsec) {
						cross_section = deldop->frame[f].overflow_xsec;
						for (i=1; i<=deldop->frame[f].ndel; i++)
							for (j=1; j<=deldop->frame[f].ndop; j++)
								cross_section += deldop->frame[f].fit[i][j];
						cross_section *= deldop->frame[f].cal.val;
						sum_rad_xsec += cross_section*weight;
					}
				}
				compute_cosdelta = (par->vary_dopscale != VARY_NONE
						&& deldop->dopscale.state != 'c');
				if (compute_cosdelta) {

					/*        oa = matrix to transform body-fixed to observer coordinates  */
					/*  to_earth = normalized target-to-Earth vector in body-fixed coords  */
					mtrnsps( oa, deldop->frame[f].view[deldop->v0].ae);
					mmmul( oa, deldop->frame[f].view[deldop->v0].oe, oa);
					for (j=0; j<=2; j++)
						to_earth[j] = oa[2][j];
					cos_delta = sqrt(to_earth[0]*to_earth[0] + to_earth[1]*to_earth[1]);
					weight = deldop->frame[f].weight;
					sum_cos_subradarlat += cos_delta*weight;
				}
			}
			break;
		case DOPPLER:

			doppler = &dat->set[s].desc.doppler;
			for (f=0; f<doppler->nframes; f++) {

				compute_xsec = (par->vary_radalb != VARY_NONE && doppler->frame[f].cal.state == 'c');

				if (compute_xsec) {
					weight = doppler->frame[f].weight;
					pos = &doppler->frame[f].pos;
					for (i=0; i<=2; i++)
						for (j=0; j<=2; j++) {
							pos->ae[i][j] = doppler->frame[f].view[doppler->v0].ae[i][j];
							pos->oe[i][j] = doppler->frame[f].view[doppler->v0].oe[i][j];
						}
					pos->bistatic = 0;

					/*  Initialize the plane-of-sky view  */
					posclr( pos);

					/*  Determine which POS pixels cover the target  */
					for (c=0; c<mod->shape.ncomp; c++){
						posvis( &mod->shape.comp[c].real, orbit_offset, pos,
									(int) par->pos_smooth, 0, 0, c);
					}
					/*  Zero out the fit Doppler spectrum, then call pos2doppler to create the fit
					 * 	spectrum by mapping power from the plane of the sky to Doppler space.      */
					clrvect( doppler->frame[f].fit, 1, doppler->frame[f].ndop);

					pos2doppler(par, &mod->photo, 0.0, 0.0, 0.0, doppler, 0, s, f, 0);

					/*  Compute cross section  */
					cross_section = doppler->frame[f].overflow_xsec;
					for (j=1; j<=doppler->frame[f].ndop; j++)
						cross_section += doppler->frame[f].fit[j];

					cross_section *= doppler->frame[f].cal.val;
					sum_rad_xsec += cross_section*weight;
				}
				compute_cosdelta = (par->vary_dopscale != VARY_NONE	&& doppler->dopscale.state != 'c');
				if (compute_cosdelta) {

					/*  oa = matrix to transform body-fixed to observer coordinates  */
					/*  to_earth = normalized target-to-Earth vector in body-fixed coords  */
					mtrnsps( oa, doppler->frame[f].view[doppler->v0].ae);
					mmmul( oa, doppler->frame[f].view[doppler->v0].oe, oa);
					for (j=0; j<=2; j++)
						to_earth[j] = oa[2][j];
					cos_delta = sqrt(to_earth[0]*to_earth[0] + to_earth[1]*to_earth[1]);
					weight = doppler->frame[f].weight;
					sum_cos_subradarlat += cos_delta*weight;
				}
			}
			break;
		case POS:
			break;
		case LGHTCRV:
			lghtcrv = &dat->set[s].desc.lghtcrv;
			compute_brightness = (par->vary_optalb != VARY_NONE
					&& lghtcrv->cal.state == 'c');
			if (compute_brightness) {
				n = lghtcrv->n;
				ncalc = lghtcrv->ncalc;
				weight = lghtcrv->weight;
				for (i=1; i<=ncalc; i++) {
					pos = &lghtcrv->rend[i].pos;
					for (j=0; j<=2; j++)
						for (k=0; k<=2; k++) {
							pos->ae[j][k] = lghtcrv->rend[i].ae[j][k];
							pos->oe[j][k] = lghtcrv->rend[i].oe[j][k];
							pos->se[j][k] = lghtcrv->rend[i].se[j][k];
						}
					pos->bistatic = 1;

					/* Initialize the plane-of-sky view */
					posclr( pos);

					/* Determine which POS pixels cover the target */
					for (c=0; c<mod->shape.ncomp; c++)
						posvis( &mod->shape.comp[c].real, orbit_offset, pos,
								(int) par->pos_smooth, 0, 0, c);

				 /* Now view the model from the source (sun) and get the facet
				  * number and distance toward the source of each pixel in this
				  * projected view; use this information to determine which POS
				  * pixels are shadowed */
					if (pos->bistatic) {
						for (c=0; c<mod->shape.ncomp; c++)
							posvis( &mod->shape.comp[c].real, orbit_offset, pos, 0, 1, 0, c);

						/* Identify and mask out shadowed POS pixels */
						posmask( pos, par->mask_tol);
					}

					/* Compute the model brightness for this model lightcurve
					 * point */
					intensityfactor = pow( pos->km_per_pixel/AU, 2.0);
					lghtcrv->y[i] = apply_photo( mod, lghtcrv->ioptlaw,
							lghtcrv->solar_phase[i],
							intensityfactor, pos, 0, s, i);
				}

				/* Now that we have calculated the model lightcurve brightnesses
				 * y at each of the epochs x, we use cubic spline interpolation
				 * (Numerical Recipes routines spline and splint) to get model
				 * lightcurve brightness fit[i] at each OBSERVATION epoch t[i],
				 * with i=1,2,...,n.  This will allow us (in routine chi2) to
				 * compare model to data (fit[i] to obs[i]) to get chi squared.
				 * Note that vector y2 contains the second derivatives of the
				 * interpolating function at the calculation epochs x. */
				spline( lghtcrv->x, lghtcrv->y, ncalc, 2.0e30, 2.0e30, lghtcrv->y2);
				for (i=1; i<=n; i++) {
					splint( lghtcrv->x, lghtcrv->y, lghtcrv->y2, ncalc,
							lghtcrv->t[i][lghtcrv->v0], &lghtcrv->fit[i]);
					sum_opt_brightness += lghtcrv->fit[i]*weight;
				}
			}
			break;
		default:
			bailout("vary_params.c: can't handle this dataset type yet\n");
		}
	}

	if (dat->sum_deldop_zmax_weights > 0.0)
		*deldop_zmax = sum_deldop_zmax / dat->sum_deldop_zmax_weights;
	else
		*deldop_zmax = 0.0;
	if (dat->sum_rad_xsec_weights > 0.0)
		*rad_xsec = sum_rad_xsec / dat->sum_rad_xsec_weights;
	else
		*rad_xsec = 0.0;
	if (dat->sum_opt_brightness_weights > 0.0)
		*opt_brightness = sum_opt_brightness / dat->sum_opt_brightness_weights;
	else
		*opt_brightness = 0.0;
	if (dat->sum_cos_subradarlat_weights > 0.0)
		*cos_subradarlat = sum_cos_subradarlat / dat->sum_cos_subradarlat_weights;
	else
		*cos_subradarlat = 0.0;
}
