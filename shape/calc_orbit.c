/*****************************************************************************************
                                                                             calc_orbit.c

This routine, a highly modified version of calc_fits, implements the "orbit" action.  It
takes two models -- for the two bodies of an orbiting binary system -- and calculates the
fits to each data frame given the two sets of model parameters and the set of orbital
parameters.  For example, for each delay-Doppler frame it calls routine posvis twice (once
for each input model) to create a single model plane-of-sky image that shows both bodies,
and then calls routine pos2deldop twice to create the model delay-Doppler image from this
POS image.

calc_orbit also performs the file output that the calc_fits routine would handle for the
"write" action; in particular, it carries out tasks that require information associated
with plane-of-sky renderings, since such information is quickly overwritten if the
"pos_scope" parameter is set to "global" (i.e., if all frames and lightcurve points share
the same memory for their "pos" structures).

The code is rather unlovely: shape's mod_t and dat_t structures are NOT set up to handle
multiple models per observation epoch.

Modified 2015 June 10 by CM:
    Implement smearing for the "fit" and "write" actions

Modified 2014 February 12 by CM:
    Add "ilaw" argument to the radlaw and apply_photo routines to implement multiple
        radar and optical scattering laws

Modified 2013 June 25 by CM:
    Allow POS images written for optical data to be annotated with principal-axis shafts
        and the angular momentum vector
    Get colors of plot annotation back into synch with the colors assigned by the
        write_pos routine
    For POS images (sky renderings) created for radar data, if the "radposmax" parameter
        is nonzero, display the value of that parameter vs. the maximum pixel value
    Add "posmax" argument to plot_arrow and plot_subradar routines

Modified 2013 April 24 by CM:
    Aesthetic changes to keep code similar to that in calc_fits.c
    Adjust names of output images so they are in alphanumeric order if > 100 per dataset

Modified 2012 July 4 by CM:
    Initialize variable in "write_pos_orbit" routine to avoid compilation warning

Modified 2012 April 2 by CM:
    Take Doppler scaling factor into account when converting orbital velocities to Doppler
        offsets relative to the COM
    Fix bug (introduced 2012 March 5) in "write_pos_orbit" routine: grayscale images
        weren't showing the spin vector even when "plot_spinvec" was turned on

Modified 2012 March 5 by CM:
    Have to renumber first argument to plot_arrow, plot_com, and plot_subradar routines

Modified 2010 September 1 by CM:
    Initialize variables to avoid compilation warnings

Modified 2010 June 15 by CM:
    Revise arguments to pos2deldop and pos2doppler routines

Modified 2009 July 29 by CM:
    Bug fix: output ppm images rather than pgm images if the "plot_angmom"
        parameter is turned on
    Pass an argument to the "write_pos_orbit" routine explicitly telling
        it whether or not to produce a colored image

Modified 2009 April 3 by CM:
    Implement the "plot_angmom" parameter to plot angular momentum vectors
        in POS images
    Implement the "pierce_spinvec" and "pierce_angmom" parameters to
        determine the appearance of spin vectors and angular momentum
        vectors in POS images
    Add "badposet" parameter: initialize it here and then use the
        "checkposet" routine to adjust it for plane-of-sky fit images that
        are too small to "contain" the target
    Add "badradar" parameter: initialize it here and then use the
        "pos2deldop" and "pos2doppler" routines (which are now int rather
        than void) to adjust it for models that are too wide in
        delay-Doppler space for the routines to handle
    Add "warn_badradar" argument to pos2deldop and pos2doppler routines

Modified 2008 July 11 by CM:
    Modify code to handle triple systems

Modified 2007 August 12 by CM:
    Add "body" argument to plot_com routine so that center-of-mass crosses
        aren't drawn on eclipsed bodies

Modified 2007 August 10 by CM:
    Eliminate unused variable

Written 2007 August 4 by CM
 *****************************************************************************************/

#include "head.h"

static double xobs1[3], xobs2[3], xobs3[3];

void calc_orbit_deldop( struct par_t *par,
		struct mod_t *mod1, struct mod_t *mod2, struct mod_t *mod3,
		struct deldop_t *deldop1, struct deldop_t *deldop2,
		struct deldop_t *deldop3, int s);
void calc_orbit_doppler( struct par_t *par,
		struct mod_t *mod1, struct mod_t *mod2, struct mod_t *mod3,
		struct doppler_t *doppler1, struct doppler_t *doppler2,
		struct doppler_t *doppler3, int s);
void calc_orbit_poset( struct par_t *par,
		struct mod_t *mod1, struct mod_t *mod2, struct mod_t *mod3,
		struct poset_t *poset1, struct poset_t *poset2,
		struct poset_t *poset3, int s);
void calc_orbit_lghtcrv( struct par_t *par, struct mod_t *mod1, struct mod_t *mod2, struct mod_t *mod3,
		struct lghtcrv_t *lghtcrv1, struct lghtcrv_t *lghtcrv2,
		struct lghtcrv_t *lghtcrv3, int s);
void write_pos_orbit_deldop( struct par_t *par,
		struct mod_t *mod1, struct mod_t *mod2, struct mod_t *mod3,
		struct deldop_t *deldop1, struct deldop_t *deldop2,
		struct deldop_t *deldop3, int s, int f);
void write_pos_orbit_doppler( struct par_t *par,
		struct mod_t *mod1, struct mod_t *mod2, struct mod_t *mod3,
		struct doppler_t *doppler1, struct doppler_t *doppler2,
		struct doppler_t *doppler3, int s, int f);
void write_pos_orbit_poset( struct par_t *par,
		struct mod_t *mod1, struct mod_t *mod2, struct mod_t *mod3,
		struct poset_t *poset1, struct poset_t *poset2,
		struct poset_t *poset3, int s, int f);
void write_pos_orbit_lghtcrv( struct par_t *par,
		struct mod_t *mod1, struct mod_t *mod2, struct mod_t *mod3,
		struct lghtcrv_t *lghtcrv1, struct lghtcrv_t *lghtcrv2,
		struct lghtcrv_t *lghtcrv3, int s, int i);
void write_pos_orbit( struct par_t *par,
		struct mod_t *mod1, struct mod_t *mod2, struct mod_t *mod3,
		struct pos_t *pos1, struct pos_t *pos2, struct pos_t *pos3,
		double spin_ecl1[3], double spin_ecl2[3], double spin_ecl3[3],
		int is_optical, int color_output, char *name);
void copy_pos( struct pos_t *pos1, struct pos_t *pos2, int mode);


void calc_orbit( struct par_t *par,
		struct mod_t *mod1, struct mod_t *mod2, struct mod_t *mod3,
		struct dat_t *dat1, struct dat_t *dat2, struct dat_t *dat3)
{
	int c, f, s, frm, i;

	/*  Initialize the flags that indicate that the model extends beyond the
      POS frame, that plane-of-sky fit images are too small to "contain"
      the target, and that the model is too wide in (delay-)Doppler space
      to create (delay-)Doppler fit frames                                  */

	par->posbnd = 0;
	par->badposet = 0;
	par->badradar = 0;

	/*  Initialize the flags which indicate whether or not each facet
      of each model component is ever visible and unshadowed from Earth  */

	for (c=0; c<mod1->shape.ncomp; c++)
		for (f=0; f<mod1->shape.comp[c].real.nf; f++)
			mod1->shape.comp[c].real.f[f].seen = 0;
	for (c=0; c<mod2->shape.ncomp; c++)
		for (f=0; f<mod2->shape.comp[c].real.nf; f++)
			mod2->shape.comp[c].real.f[f].seen = 0;
	if (par->is_triple)
		for (c=0; c<mod3->shape.ncomp; c++)
			for (f=0; f<mod3->shape.comp[c].real.nf; f++)
				mod3->shape.comp[c].real.f[f].seen = 0;

	/*  Calculate the fits for each dataset in turn  */
	for (s=0; s<dat1->nsets; s++) {
		switch (dat1->set[s].type) {
		case DELAY:
			calc_orbit_deldop( par, mod1, mod2, mod3,
					&dat1->set[s].desc.deldop, &dat2->set[s].desc.deldop,
					&dat3->set[s].desc.deldop, s);
			break;
		case DOPPLER:
			calc_orbit_doppler( par, mod1, mod2, mod3,
					&dat1->set[s].desc.doppler, &dat2->set[s].desc.doppler,
					&dat3->set[s].desc.doppler, s);
			break;
		case POS:
			calc_orbit_poset( par, mod1, mod2, mod3,
					&dat1->set[s].desc.poset, &dat2->set[s].desc.poset,
					&dat3->set[s].desc.poset, s);
			break;
		case LGHTCRV:
			calc_orbit_lghtcrv( par, mod1, mod2, mod3,
					&dat1->set[s].desc.lghtcrv, &dat2->set[s].desc.lghtcrv,
					&dat3->set[s].desc.lghtcrv, s);
			break;
		default:
			bailout("calc_orbit.c: can't handle this type yet\n");
		}
	}

	/*  If the "mark_unseen" parameter is turned on,
      go back and create the colored POS images     */

	if (par->mark_unseen) {
		for (s=0; s<dat1->nsets; s++) {
			switch (dat1->set[s].type) {
			case DELAY:
				for (frm=0; frm<dat1->set[s].desc.deldop.nframes; frm++)
					write_pos_orbit_deldop( par, mod1, mod2, mod3,
							&dat1->set[s].desc.deldop,
							&dat2->set[s].desc.deldop,
							&dat3->set[s].desc.deldop,
							s, frm);
				break;
			case DOPPLER:
				for (frm=0; frm<dat1->set[s].desc.doppler.nframes; frm++)
					write_pos_orbit_doppler( par, mod1, mod2, mod3,
							&dat1->set[s].desc.doppler,
							&dat2->set[s].desc.doppler,
							&dat3->set[s].desc.doppler,
							s, frm);
				break;
			case POS:
				for (frm=0; frm<dat1->set[s].desc.poset.nframes; frm++)
					write_pos_orbit_poset( par, mod1, mod2, mod3,
							&dat1->set[s].desc.poset,
							&dat2->set[s].desc.poset,
							&dat3->set[s].desc.poset,
							s, frm);
				break;
			case LGHTCRV:
				if (par->lcrv_pos)
					for (i=1; i<=dat1->set[s].desc.lghtcrv.ncalc; i++)
						write_pos_orbit_lghtcrv( par, mod1, mod2, mod3,
								&dat1->set[s].desc.lghtcrv,
								&dat2->set[s].desc.lghtcrv,
								&dat3->set[s].desc.lghtcrv,
								s, i);
				break;
			}
		}
	}
}


void calc_orbit_deldop( struct par_t *par,
		struct mod_t *mod1, struct mod_t *mod2, struct mod_t *mod3,
		struct deldop_t *deldop1, struct deldop_t *deldop2,
		struct deldop_t *deldop3, int s)
{
	int f, i, j, c, k, l, facetnum;
	double xecl[3], vecl[3], xobs12[3], xobs13[3], vobs12[3], vobs13[3],
	vobs1[3], vobs2[3], vobs3[3], xoff1, yoff1, xoff2, yoff2, xoff3, yoff3,
	dopoff1, dopoff2, dopoff3, volume_ratio2, volume_ratio3, dopfact;
	struct deldopfrm_t *frame1, *frame2, *frame3;
	struct deldopview_t *view0_1, *view0_2, *view0_3;
	struct pos_t *pos1, *pos2, *pos3;

	/*  Initialize variables to avoid compilation warnings  */

	xoff3 = yoff3 = dopoff3 = 0.0;

	/*  Compute conversion factor from speed (km/day) toward
      the radar (observer v_z) to Doppler columns           */

	dopfact = deldop1->dopscale.val * KM2HZFACT * deldop1->Ftx / deldop1->dop_per_pixel;

	/*  Loop through frames  */

	for (f=0; f<deldop1->nframes; f++) {

		frame1 = &deldop1->frame[f];
		frame2 = &deldop2->frame[f];
		frame3 = &deldop3->frame[f];
		view0_1 = &frame1->view[deldop1->v0];
		view0_2 = &frame2->view[deldop2->v0];
		view0_3 = &frame3->view[deldop3->v0];
		pos1 = &frame1->pos;
		pos2 = &frame2->pos;
		pos3 = &frame3->pos;

		/*  Solve the Keplerian orbit for this frame's epoch
        (relative displacement xecl and velocity vecl are in ecliptic coordinates)  */

		if (kepler(par->binary_gravparam[0], par->eccentricity[0], par->r_pericenter[0],
				view0_1->t - par->t_pericenter[0],
				par->long_asc_node[0], par->inclination[0], par->arg_pericenter[0],
				xecl, vecl) != 0) {
			printf("Kepler's equation didn't converge for frame %d of set %d for body 2\n", f, s);
			bailout("calc_orbit.c\n");
		}

		/*  Convert the relative displacement and velocity to observer coordinates;
        note that the oe transformation matrix is the same for both models       */

		cotrans( xobs12, view0_1->oe, xecl, 1);
		cotrans( vobs12, view0_1->oe, vecl, 1);

		/*  Repeat all of the above for the third body, if present  */

		if (par->is_triple) {
			if (kepler(par->binary_gravparam[1], par->eccentricity[1], par->r_pericenter[1],
					view0_1->t - par->t_pericenter[1],
					par->long_asc_node[1], par->inclination[1], par->arg_pericenter[1],
					xecl, vecl) != 0) {
				printf("Kepler's equation didn't converge for frame %d of set %d for body 3\n", f, s);
				bailout("calc_orbit.c\n");
			}

			cotrans( xobs13, view0_1->oe, xecl, 1);
			cotrans( vobs13, view0_1->oe, vecl, 1);
		}

		/*  If requested, get absolute displacements and velocities by assuming
        equal densities; otherwise fix the primary motionless and treat
        the relative motion as the secondary's absolute motion.              */

		if (par->is_triple) {
			volume_ratio2 = mod2->shape.volume /
					(mod1->shape.volume + mod2->shape.volume + mod3->shape.volume);
			volume_ratio3 = mod3->shape.volume /
					(mod1->shape.volume + mod2->shape.volume + mod3->shape.volume);
			for (j=0; j<=2; j++) {
				if (par->orbit_reflex) {
					xobs1[j] = -(volume_ratio2*xobs12[j] + volume_ratio3*xobs13[j]);
					vobs1[j] = -(volume_ratio2*vobs12[j] + volume_ratio3*vobs13[j]);
				} else {
					xobs1[j] = 0.0;
					vobs1[j] = 0.0;
				}
				xobs2[j] = xobs1[j] + xobs12[j];
				vobs2[j] = vobs1[j] + vobs12[j];
				xobs3[j] = xobs1[j] + xobs13[j];
				vobs3[j] = vobs1[j] + vobs13[j];
			}
		} else {
			volume_ratio2 = mod2->shape.volume /
					(mod1->shape.volume + mod2->shape.volume);
			for (j=0; j<=2; j++) {
				if (par->orbit_reflex) {
					xobs1[j] = -volume_ratio2*xobs12[j];
					vobs1[j] = -volume_ratio2*vobs12[j];
				} else {
					xobs1[j] = 0.0;
					vobs1[j] = 0.0;
				}
				xobs2[j] = xobs1[j] + xobs12[j];
				vobs2[j] = vobs1[j] + vobs12[j];
			}
		}

		/*  Convert absolute x and y displacements from km to POS pixels  */

		xoff1 = xobs1[0] / pos1->km_per_pixel;
		yoff1 = xobs1[1] / pos1->km_per_pixel;
		xoff2 = xobs2[0] / pos2->km_per_pixel;
		yoff2 = xobs2[1] / pos2->km_per_pixel;
		if (par->is_triple) {
			xoff3 = xobs3[0] / pos3->km_per_pixel;
			yoff3 = xobs3[1] / pos3->km_per_pixel;
		}

		/*  Convert absolute velocities to Doppler offsets (image columns);
        the posvis and pos2deldop routines already know how to convert
        absolute displacements to delay offsets (image rows).            */

		dopoff1 = vobs1[2]*dopfact;
		dopoff2 = vobs2[2]*dopfact;
		if (par->is_triple)
			dopoff3 = vobs3[2]*dopfact;

		/*  Assign some values to POS structures  */

		for (i=0; i<=2; i++)
			for (j=0; j<=2; j++) {
				pos1->ae[i][j] = view0_1->ae[i][j];
				pos1->oe[i][j] = view0_1->oe[i][j];
				pos2->ae[i][j] = view0_2->ae[i][j];
				pos2->oe[i][j] = view0_2->oe[i][j];
				if (par->is_triple) {
					pos3->ae[i][j] = view0_3->ae[i][j];
					pos3->oe[i][j] = view0_3->oe[i][j];
				}
			}
		pos1->bistatic = 0;
		pos2->bistatic = 0;
		if (par->is_triple)
			pos3->bistatic = 0;

		/*  Initialize the plane-of-sky views  */

		posclr( pos1);
		posclr( pos2);
		if (par->is_triple)
			posclr( pos3);

		/*  Call routine posvis for the first model to get the facet number,
        scattering angle, and distance toward Earth at the center of
        each POS pixel; set the posbnd parameter to 1 if any portion of
        the model extends beyond the POS frame limits.                    */

		for (c=0; c<mod1->shape.ncomp; c++)
			if (posvis( &mod1->shape.comp[c].real, xobs1, pos1,
					(int) par->pos_smooth, 0, 0, c))
				par->posbnd = 1;


		/*  Now add in the POS contributions for the second model.
        This is a bit convoluted because the posvis results depend on
        the ae (body-fixed-to-ecliptic) coordinate transformation
        matrix, which is different for the two models and which is
        stored in the pos structure for this frame.  Hence we have to
        copy the relevant parts of that structure from model 1 to 2,
        then call posvis for the second model, and then copy back to
        the first model's pos structure.                               */

		copy_pos( pos1, pos2, 1);

		for (c=0; c<mod2->shape.ncomp; c++)
			if (posvis( &mod2->shape.comp[c].real, xobs2, pos2,
					(int) par->pos_smooth, 0, 1, c))
				par->posbnd = 1;

		copy_pos( pos2, pos1, 1);

		/*  Now add in the POS contributions for the third model, if present  */

		if (par->is_triple) {
			copy_pos( pos1, pos3, 1);

			for (c=0; c<mod3->shape.ncomp; c++)
				if (posvis( &mod3->shape.comp[c].real, xobs3, pos3,
						(int) par->pos_smooth, 0, 2, c))
					par->posbnd = 1;

			copy_pos( pos3, pos1, 1);
		}

		/*  Go through all POS pixels that are visible with sufficiently low
        scattering angle, and mark the facets that project onto their
        centers as having been "seen" at least once                       */

		if (s != par->exclude_seen) {
			for (k=pos1->xlim[0]; k<=pos1->xlim[1]; k++)
				for (l=pos1->ylim[0]; l<=pos1->ylim[1]; l++) {
					if ((pos1->cose[k][l] > par->mincosine_seen)
							&& (pos1->f[k][l] >= 0)) {
						facetnum = pos1->f[k][l];
						c = pos1->comp[k][l];
						if (pos1->body[k][l] == 0)
							mod1->shape.comp[c].real.f[facetnum].seen = 1;
						else if (pos2->body[k][l] == 1)
							mod2->shape.comp[c].real.f[facetnum].seen = 1;
						else
							mod3->shape.comp[c].real.f[facetnum].seen = 1;
					}
				}
		}

		/*  For each model, zero out the fit delay-Doppler image, then
        call pos2deldop to create the fit image by mapping power
        from the plane of the sky to delay-Doppler space.           */

		clrmat( frame1->fit, 1, frame1->ndel, 1, frame1->ndop);
		clrmat( frame2->fit, 1, frame2->ndel, 1, frame2->ndop);
		if (par->is_triple)
			clrmat( frame3->fit, 1, frame3->ndel, 1, frame3->ndop);
		if (pos2deldop( par, &mod1->photo, xoff1, yoff1, dopoff1, deldop1, 0, s, f, 0))
			par->badradar = 1;
		if (pos2deldop( par, &mod2->photo, xoff2, yoff2, dopoff2, deldop2, 1, s, f, 0))
			par->badradar = 1;
		if (par->is_triple)
			if (pos2deldop( par, &mod3->photo, xoff3, yoff3, dopoff3, deldop3, 2, s, f, 0))
				par->badradar = 1;

		/*  Sum the two contributions to the fit delay-Doppler image and
        associated quantities, and store them in data structure #1    */

		for (i=1; i<=frame1->ndel; i++)
			for (j=1; j<=frame1->ndop; j++)
				frame1->fit[i][j] += frame2->fit[i][j];
		if (par->is_triple)
			for (i=1; i<=frame1->ndel; i++)
				for (j=1; j<=frame1->ndop; j++)
					frame1->fit[i][j] += frame3->fit[i][j];

		frame1->overflow_o2 += frame2->overflow_o2;
		frame1->overflow_m2 += frame2->overflow_m2;
		frame1->overflow_delmean *= frame1->overflow_xsec;
		frame1->overflow_delmean += frame2->overflow_delmean * frame2->overflow_xsec;
		frame1->overflow_dopmean *= frame1->overflow_xsec;
		frame1->overflow_dopmean += frame2->overflow_dopmean * frame2->overflow_xsec;
		frame1->overflow_xsec += frame2->overflow_xsec;
		if (par->is_triple) {
			frame1->overflow_o2 += frame3->overflow_o2;
			frame1->overflow_m2 += frame3->overflow_m2;
			frame1->overflow_delmean += frame3->overflow_delmean * frame3->overflow_xsec;
			frame1->overflow_dopmean += frame3->overflow_dopmean * frame3->overflow_xsec;
			frame1->overflow_xsec += frame3->overflow_xsec;
		}
		frame1->overflow_delmean /= frame1->overflow_xsec;
		frame1->overflow_dopmean /= frame1->overflow_xsec;

		frame1->idellim[0] = MIN( frame1->idellim[0], frame2->idellim[0]);
		frame1->idellim[1] = MAX( frame1->idellim[1], frame2->idellim[1]);
		frame1->idoplim[0] = MIN( frame1->idoplim[0], frame2->idoplim[0]);
		frame1->idoplim[1] = MAX( frame1->idoplim[1], frame2->idoplim[1]);
		frame1->dellim[0] = MIN( frame1->dellim[0], frame2->dellim[0]);
		frame1->dellim[1] = MAX( frame1->dellim[1], frame2->dellim[1]);
		frame1->doplim[0] = MIN( frame1->doplim[0], frame2->doplim[0]);
		frame1->doplim[1] = MAX( frame1->doplim[1], frame2->doplim[1]);
		if (par->is_triple) {
			frame1->idellim[0] = MIN( frame1->idellim[0], frame3->idellim[0]);
			frame1->idellim[1] = MAX( frame1->idellim[1], frame3->idellim[1]);
			frame1->idoplim[0] = MIN( frame1->idoplim[0], frame3->idoplim[0]);
			frame1->idoplim[1] = MAX( frame1->idoplim[1], frame3->idoplim[1]);
			frame1->dellim[0] = MIN( frame1->dellim[0], frame3->dellim[0]);
			frame1->dellim[1] = MAX( frame1->dellim[1], frame3->dellim[1]);
			frame1->doplim[0] = MIN( frame1->doplim[0], frame3->doplim[0]);
			frame1->doplim[1] = MAX( frame1->doplim[1], frame3->doplim[1]);
		}

		/*  Carry out a gamma transformation on the fit image if requested  */

		if (par->dd_gamma != 1.0) {
			for (i=1; i<=frame1->ndel; i++)
				for (j=1; j<=frame1->ndop; j++)
					gamma_trans( &frame1->fit[i][j], par->dd_gamma);
		}

		/*  Create a plane-of-sky image which assumes front illumination
        and Lambert scattering: pixel brightness level proportional
        to cos(scattering angle)                                      */

		if (!par->mark_unseen)
			write_pos_orbit_deldop( par, mod1, mod2, mod3, deldop1, deldop2, deldop3, s, f);
	}
}


void calc_orbit_doppler( struct par_t *par,
		struct mod_t *mod1, struct mod_t *mod2, struct mod_t *mod3,
		struct doppler_t *doppler1, struct doppler_t *doppler2,
		struct doppler_t *doppler3, int s)
{
	int f, i, j, c, k, l, facetnum;
	double xecl[3], vecl[3], xobs12[3], xobs13[3], vobs12[3], vobs13[3],
	vobs1[3], vobs2[3], vobs3[3], xoff1, yoff1, xoff2, yoff2, xoff3, yoff3,
	dopoff1, dopoff2, dopoff3, volume_ratio2, volume_ratio3, dopfact;
	struct dopfrm_t *frame1, *frame2, *frame3;
	struct dopview_t *view0_1, *view0_2, *view0_3;
	struct pos_t *pos1, *pos2, *pos3;

	/*  Initialize variables to avoid compilation warnings  */

	xoff3 = yoff3 = dopoff3 = 0.0;

	/*  Compute conversion factor from speed (km/day) toward
      the radar (observer v_z) to Doppler columns           */

	dopfact = doppler1->dopscale.val * KM2HZFACT * doppler1->Ftx / doppler1->dop_per_bin;

	/*  Loop through frames  */

	for (f=0; f<doppler1->nframes; f++) {

		frame1 = &doppler1->frame[f];
		frame2 = &doppler2->frame[f];
		frame3 = &doppler3->frame[f];
		view0_1 = &frame1->view[doppler1->v0];
		view0_2 = &frame2->view[doppler2->v0];
		view0_3 = &frame3->view[doppler3->v0];
		pos1 = &frame1->pos;
		pos2 = &frame2->pos;
		pos3 = &frame3->pos;

		/*  Solve the Keplerian orbit for this frame's epoch
        (relative displacement xecl and velocity vecl are in ecliptic coordinates)  */

		if (kepler(par->binary_gravparam[0], par->eccentricity[0], par->r_pericenter[0],
				view0_1->t - par->t_pericenter[0],
				par->long_asc_node[0], par->inclination[0], par->arg_pericenter[0],
				xecl, vecl) != 0) {
			printf("Kepler's equation didn't converge for frame %d of set %d for body 2\n", f, s);
			bailout("calc_orbit.c\n");
		}

		/*  Convert the relative displacement and velocity to observer coordinates;
        note that the oe transformation matrix is the same for both models       */

		cotrans( xobs12, view0_1->oe, xecl, 1);
		cotrans( vobs12, view0_1->oe, vecl, 1);

		/*  Repeat all of the above for the third body, if present  */

		if (par->is_triple) {
			if (kepler(par->binary_gravparam[1], par->eccentricity[1], par->r_pericenter[1],
					view0_1->t - par->t_pericenter[1],
					par->long_asc_node[1], par->inclination[1], par->arg_pericenter[1],
					xecl, vecl) != 0) {
				printf("Kepler's equation didn't converge for frame %d of set %d for body 3\n", f, s);
				bailout("calc_orbit.c\n");
			}

			cotrans( xobs13, view0_1->oe, xecl, 1);
			cotrans( vobs13, view0_1->oe, vecl, 1);
		}

		/*  If requested, get absolute displacements and velocities by assuming
        equal densities; otherwise fix the primary motionless and treat
        the relative motion as the secondary's absolute motion.              */

		if (par->is_triple) {
			volume_ratio2 = mod2->shape.volume /
					(mod1->shape.volume + mod2->shape.volume + mod3->shape.volume);
			volume_ratio3 = mod3->shape.volume /
					(mod1->shape.volume + mod2->shape.volume + mod3->shape.volume);
			for (j=0; j<=2; j++) {
				if (par->orbit_reflex) {
					xobs1[j] = -(volume_ratio2*xobs12[j] + volume_ratio3*xobs13[j]);
					vobs1[j] = -(volume_ratio2*vobs12[j] + volume_ratio3*vobs13[j]);
				} else {
					xobs1[j] = 0.0;
					vobs1[j] = 0.0;
				}
				xobs2[j] = xobs1[j] + xobs12[j];
				vobs2[j] = vobs1[j] + vobs12[j];
				xobs3[j] = xobs1[j] + xobs13[j];
				vobs3[j] = vobs1[j] + vobs13[j];
			}
		} else {
			volume_ratio2 = mod2->shape.volume /
					(mod1->shape.volume + mod2->shape.volume);
			for (j=0; j<=2; j++) {
				if (par->orbit_reflex) {
					xobs1[j] = -volume_ratio2*xobs12[j];
					vobs1[j] = -volume_ratio2*vobs12[j];
				} else {
					xobs1[j] = 0.0;
					vobs1[j] = 0.0;
				}
				xobs2[j] = xobs1[j] + xobs12[j];
				vobs2[j] = vobs1[j] + vobs12[j];
			}
		}

		/*  Convert absolute x and y displacements from km to POS pixels  */

		xoff1 = xobs1[0] / pos1->km_per_pixel;
		yoff1 = xobs1[1] / pos1->km_per_pixel;
		xoff2 = xobs2[0] / pos2->km_per_pixel;
		yoff2 = xobs2[1] / pos2->km_per_pixel;
		if (par->is_triple) {
			xoff3 = xobs3[0] / pos3->km_per_pixel;
			yoff3 = xobs3[1] / pos3->km_per_pixel;
		}

		/*  Convert absolute velocities to Doppler offsets (spectral bins)  */

		dopoff1 = vobs1[2]*dopfact;
		dopoff2 = vobs2[2]*dopfact;
		if (par->is_triple)
			dopoff3 = vobs3[2]*dopfact;

		/*  Assign some values to POS structures  */

		for (i=0; i<=2; i++)
			for (j=0; j<=2; j++) {
				pos1->ae[i][j] = view0_1->ae[i][j];
				pos1->oe[i][j] = view0_1->oe[i][j];
				pos2->ae[i][j] = view0_2->ae[i][j];
				pos2->oe[i][j] = view0_2->oe[i][j];
				if (par->is_triple) {
					pos3->ae[i][j] = view0_3->ae[i][j];
					pos3->oe[i][j] = view0_3->oe[i][j];
				}
			}
		pos1->bistatic = 0;
		pos2->bistatic = 0;
		if (par->is_triple)
			pos3->bistatic = 0;

		/*  Initialize the plane-of-sky views  */

		posclr( pos1);
		posclr( pos2);
		if (par->is_triple)
			posclr( pos3);

		/*  Call routine posvis for the first model to get the facet number,
        scattering angle, and distance toward Earth at the center of
        each POS pixel; set the posbnd parameter to 1 if any portion of
        the model extends beyond the POS frame limits.                    */

		for (c=0; c<mod1->shape.ncomp; c++)
			if (posvis( &mod1->shape.comp[c].real, xobs1, pos1,
					(int) par->pos_smooth, 0, 0, c))
				par->posbnd = 1;

		/*  Now add in the POS contributions for the second model.
        This is a bit convoluted because the posvis results depend on
        the ae (body-fixed-to-ecliptic) coordinate transformation
        matrix, which is different for the two models and which is
        stored in the pos structure for this frame.  Hence we have to
        copy the relevant parts of that structure from model 1 to 2,
        then call posvis for the second model, and then copy back to
        the first model's pos structure.                               */

		copy_pos( pos1, pos2, 1);

		for (c=0; c<mod2->shape.ncomp; c++)
			if (posvis( &mod2->shape.comp[c].real, xobs2, pos2,
					(int) par->pos_smooth, 0, 1, c))
				par->posbnd = 1;

		copy_pos( pos2, pos1, 1);

		/*  Now add in the POS contributions for the third model, if present  */

		if (par->is_triple) {
			copy_pos( pos1, pos3, 1);

			for (c=0; c<mod3->shape.ncomp; c++)
				if (posvis( &mod3->shape.comp[c].real, xobs3, pos3,
						(int) par->pos_smooth, 0, 2, c))
					par->posbnd = 1;

			copy_pos( pos3, pos1, 1);
		}

		/*  Go through all POS pixels that are visible with sufficiently low
        scattering angle, and mark the facets that project onto their
        centers as having been "seen" at least once                       */

		if (s != par->exclude_seen) {
			for (k=pos1->xlim[0]; k<=pos1->xlim[1]; k++)
				for (l=pos1->ylim[0]; l<=pos1->ylim[1]; l++) {
					if ((pos1->cose[k][l] > par->mincosine_seen)
							&& (pos1->f[k][l] >= 0)) {
						facetnum = pos1->f[k][l];
						c = pos1->comp[k][l];
						if (pos1->body[k][l] == 0)
							mod1->shape.comp[c].real.f[facetnum].seen = 1;
						else if (pos1->body[k][l] == 1)
							mod2->shape.comp[c].real.f[facetnum].seen = 1;
						else
							mod3->shape.comp[c].real.f[facetnum].seen = 1;
					}
				}
		}

		/*  For each model, zero out the fit Doppler spectrum, then
        call pos2doppler to create the fit image by mapping power
        from the plane of the sky to Doppler space.                */

		clrvect( frame1->fit, 1, frame1->ndop);
		clrvect( frame2->fit, 1, frame2->ndop);
		if (par->is_triple)
			clrvect( frame3->fit, 1, frame3->ndop);
		if (pos2doppler( par, &mod1->photo, xoff1, yoff1, dopoff1, doppler1, 0, s, f, 0))
			par->badradar = 1;
		if (pos2doppler( par, &mod2->photo, xoff2, yoff2, dopoff2, doppler2, 1, s, f, 0))
			par->badradar = 1;
		if (par->is_triple)
			if (pos2doppler( par, &mod3->photo, xoff3, yoff3, dopoff3, doppler3, 2, s, f, 0))
				par->badradar = 1;

		/*  Sum the two contributions to the fit Doppler spectrum and
        associated quantities, and store them in data structure #1  */

		for (j=1; j<=frame1->ndop; j++)
			frame1->fit[j] += frame2->fit[j];
		if (par->is_triple)
			for (j=1; j<=frame1->ndop; j++)
				frame1->fit[j] += frame3->fit[j];

		frame1->overflow_o2 += frame2->overflow_o2;
		frame1->overflow_m2 += frame2->overflow_m2;
		frame1->overflow_dopmean *= frame1->overflow_xsec;
		frame1->overflow_dopmean +=
				frame2->overflow_dopmean * frame2->overflow_xsec;
		frame1->overflow_xsec += frame2->overflow_xsec;
		if (par->is_triple) {
			frame1->overflow_o2 += frame3->overflow_o2;
			frame1->overflow_m2 += frame3->overflow_m2;
			frame1->overflow_dopmean +=
					frame3->overflow_dopmean * frame3->overflow_xsec;
			frame1->overflow_xsec += frame3->overflow_xsec;
		}
		frame1->overflow_dopmean /= frame1->overflow_xsec;

		frame1->idoplim[0] =
				MIN( frame1->idoplim[0], frame2->idoplim[0]);
		frame1->idoplim[1] =
				MAX( frame1->idoplim[1], frame2->idoplim[1]);
		frame1->doplim[0] =
				MIN( frame1->doplim[0], frame2->doplim[0]);
		frame1->doplim[1] =
				MAX( frame1->doplim[1], frame2->doplim[1]);
		if (par->is_triple) {
			frame1->idoplim[0] =
					MIN( frame1->idoplim[0], frame3->idoplim[0]);
			frame1->idoplim[1] =
					MAX( frame1->idoplim[1], frame3->idoplim[1]);
			frame1->doplim[0] =
					MIN( frame1->doplim[0], frame3->doplim[0]);
			frame1->doplim[1] =
					MAX( frame1->doplim[1], frame3->doplim[1]);
		}

		/*  Create a plane-of-sky image which assumes front illumination
        and Lambert scattering: pixel brightness level proportional
        to cos(scattering angle)                                      */

		if (!par->mark_unseen)
			write_pos_orbit_doppler( par, mod1, mod2, mod3, doppler1, doppler2, doppler3, s, f);
	}
}


void calc_orbit_poset( struct par_t *par,
		struct mod_t *mod1, struct mod_t *mod2, struct mod_t *mod3,
		struct poset_t *poset1, struct poset_t *poset2,
		struct poset_t *poset3, int s)
{
	int f, c, i, j, k, l, nrow_fit, ncol_fit, n_pos, facetnum;
	double xoff, yoff, resamp_fact, resamp_x0, resamp_y0, xcom_fit, ycom_fit,
	resamp_xwidth, resamp_ywidth, resamp_angle, intensityfactor,
	xecl[3], vecl[3], xobs12[3], xobs13[3], xsun12[3], xsun13[3],
	xsun1[3], xsun2[3], xsun3[3], volume_ratio2, volume_ratio3,
	badposet_logfactor_frame;
	struct posetfrm_t *frame1, *frame2, *frame3;
	struct posetview_t *view0_1, *view0_2, *view0_3;
	struct pos_t *pos1, *pos2, *pos3;

	/*  Loop through frames  */

	for (f=0; f<poset1->nframes; f++) {

		frame1 = &poset1->frame[f];
		frame2 = &poset2->frame[f];
		frame3 = &poset3->frame[f];
		view0_1 = &frame1->view[poset1->v0];
		view0_2 = &frame2->view[poset2->v0];
		view0_3 = &frame3->view[poset3->v0];
		pos1 = &frame1->pos;
		pos2 = &frame2->pos;
		pos3 = &frame3->pos;

		/*  Solve the Keplerian orbit for this frame's epoch
        (relative displacement xecl and velocity vecl are in ecliptic coordinates)  */

		if (kepler(par->binary_gravparam[0], par->eccentricity[0], par->r_pericenter[0],
				view0_1->t - par->t_pericenter[0],
				par->long_asc_node[0], par->inclination[0], par->arg_pericenter[0],
				xecl, vecl) != 0) {
			printf("Kepler's equation didn't converge for frame %d of set %d for body 2\n", f, s);
			bailout("calc_orbit.c\n");
		}

		/*  Convert the relative displacement to observer coordinates
        and also to source (sun) coordinates; note that the oe and se
        coordinate transformation matrices are the same for both models  */

		cotrans( xobs12, view0_1->oe, xecl, 1);
		cotrans( xsun12, view0_1->se, xecl, 1);

		/*  Repeat all of the above for the third body, if present  */

		if (par->is_triple) {
			if (kepler(par->binary_gravparam[1], par->eccentricity[1], par->r_pericenter[1],
					view0_1->t - par->t_pericenter[1],
					par->long_asc_node[1], par->inclination[1], par->arg_pericenter[1],
					xecl, vecl) != 0) {
				printf("Kepler's equation didn't converge for frame %d of set %d for body 3\n", f, s);
				bailout("calc_orbit.c\n");
			}
			cotrans( xobs13, view0_1->oe, xecl, 1);
			cotrans( xsun13, view0_1->se, xecl, 1);
		}

		/*  If requested, get absolute displacements by assuming
        equal densities; otherwise fix the primary motionless and treat
        the relative motion as the secondary's absolute motion.          */

		if (par->is_triple) {
			volume_ratio2 = mod2->shape.volume /
					(mod1->shape.volume + mod2->shape.volume + mod3->shape.volume);
			volume_ratio3 = mod3->shape.volume /
					(mod1->shape.volume + mod2->shape.volume + mod3->shape.volume);
			for (j=0; j<=2; j++) {
				if (par->orbit_reflex) {
					xobs1[j] = -(volume_ratio2*xobs12[j] + volume_ratio3*xobs13[j]);
					xsun1[j] = -(volume_ratio2*xsun12[j] + volume_ratio3*xsun13[j]);
				} else {
					xobs1[j] = 0.0;
					xsun1[j] = 0.0;
				}
				xobs2[j] = xobs1[j] + xobs12[j];
				xobs3[j] = xobs1[j] + xobs13[j];
				xsun2[j] = xsun1[j] + xsun12[j];
				xsun3[j] = xsun1[j] + xsun13[j];
			}
		} else {
			volume_ratio2 = mod2->shape.volume /
					(mod1->shape.volume + mod2->shape.volume);
			for (j=0; j<=2; j++) {
				if (par->orbit_reflex) {
					xobs1[j] = -volume_ratio2*xobs12[j];
					xsun1[j] = -volume_ratio2*xsun12[j];
				} else {
					xobs1[j] = 0.0;
					xsun1[j] = 0.0;
				}
				xobs2[j] = xobs1[j] + xobs12[j];
				xsun2[j] = xsun1[j] + xsun12[j];
			}
		}

		/*  Assign some values to POS structures  */

		for (i=0; i<=2; i++)
			for (j=0; j<=2; j++) {
				pos1->ae[i][j] = view0_1->ae[i][j];
				pos1->oe[i][j] = view0_1->oe[i][j];
				pos2->ae[i][j] = view0_2->ae[i][j];
				pos2->oe[i][j] = view0_2->oe[i][j];
				if (par->is_triple) {
					pos3->ae[i][j] = view0_3->ae[i][j];
					pos3->oe[i][j] = view0_3->oe[i][j];
				}
			}
		pos1->bistatic = 1;
		pos2->bistatic = 1;
		if (par->is_triple)
			pos3->bistatic = 1;

		/*  Initialize the plane-of-sky views  */

		posclr( pos1);
		posclr( pos2);
		if (par->is_triple)
			posclr( pos3);

		/*  Call routine posvis for the first model to get the facet number,
        scattering angle, and distance toward Earth at the center of
        each POS pixel; set the posbnd parameter to 1 if any portion of
        the model extends beyond the POS frame limits.                    */

		for (c=0; c<mod1->shape.ncomp; c++)
			if (posvis( &mod1->shape.comp[c].real, xobs1, pos1,
					(int) par->pos_smooth, 0, 0, c))
				par->posbnd = 1;

		/*  Now view the model from the source (sun) and get the facet number
        and distance toward the source of each pixel in this projected view;
        use this information to determine which POS pixels are shadowed       */

		if (pos1->bistatic)
			for (c=0; c<mod1->shape.ncomp; c++)
				if (posvis( &mod1->shape.comp[c].real, xsun1, pos1,
						0, 1, 0, c))
					par->posbnd = 1;

		/*  Now add in the POS contributions for the second model.
        This is a bit convoluted because the posvis results depend on
        the ae (body-fixed-to-ecliptic) coordinate transformation
        matrix, which is different for the two models and which is
        stored in the pos structure for this frame.  Hence we have to
        copy the relevant parts of that structure from model 1 to 2,
        then call posvis for the second model.                         */

		copy_pos( pos1, pos2, 3);

		for (c=0; c<mod2->shape.ncomp; c++)
			if (posvis( &mod2->shape.comp[c].real, xobs2, pos2,
					(int) par->pos_smooth, 0, 1, c))
				par->posbnd = 1;

		if (pos2->bistatic)
			for (c=0; c<mod2->shape.ncomp; c++)
				if (posvis( &mod2->shape.comp[c].real, xsun2, pos2,
						0, 1, 1, c))
					par->posbnd = 1;

		/*  Identify and mask out shadowed POS pixels in the second model's
        pos structure, then copy back to the first model's pos structure  */

		if (pos2->bistatic)
			posmask( pos2, par->mask_tol);

		copy_pos( pos2, pos1, 3);

		/*  Now add in the POS contributions for the third model, if present  */

		if (par->is_triple) {
			copy_pos( pos1, pos3, 3);

			for (c=0; c<mod3->shape.ncomp; c++)
				if (posvis( &mod3->shape.comp[c].real, xobs3, &(*pos3),
						(int) par->pos_smooth, 0, 2, c))
					par->posbnd = 1;

			if (pos3->bistatic)
				for (c=0; c<mod3->shape.ncomp; c++)
					if (posvis( &mod3->shape.comp[c].real, xsun3, &(*pos3),
							0, 1, 2, c))
						par->posbnd = 1;

			if (pos3->bistatic)
				posmask( &(*pos3), par->mask_tol);

			copy_pos( pos3, pos1, 3);
		}

		/*  Go through all POS pixels that are visible and unshadowed with
        sufficiently low scattering and incidence angles, and mark the facets
        that project onto their centers as having been "seen" at least once    */

		if (s != par->exclude_seen) {
			for (k=pos1->xlim[0]; k<=pos1->xlim[1]; k++)
				for (l=pos1->ylim[0]; l<=pos1->ylim[1]; l++) {
					if ((pos1->cose[k][l] > par->mincosine_seen)
							&& (pos1->cosi[k][l] > par->mincosine_seen)
							&& (pos1->f[k][l] >= 0)) {
						facetnum = pos1->f[k][l];
						c = pos1->comp[k][l];
						if (pos1->body[k][l] == 0)
							mod1->shape.comp[c].real.f[facetnum].seen = 1;
						else if (pos1->body[k][l] == 1)
							mod2->shape.comp[c].real.f[facetnum].seen = 1;
						else
							mod3->shape.comp[c].real.f[facetnum].seen = 1;
					}
				}
		}

		/*  Compute the sky rendering and store it in data structure #1  */

		intensityfactor = pow( pos1->km_per_pixel/AU, 2.0);
		apply_photo( mod1, poset1->ioptlaw, view0_1->solar_phase, intensityfactor,
				pos1, 0, s, i);
		apply_photo( mod2, poset1->ioptlaw, view0_1->solar_phase, intensityfactor,
				pos1, 1, s, i);
		if (par->is_triple)
			apply_photo( mod3, poset1->ioptlaw, view0_1->solar_phase, intensityfactor,
					pos1, 2, s, i);

		/*  Resample the sky rendering to get the model plane-of-sky image    */
		/*  (if using bicubic interpolation or cubic convolution, force       */
		/*  all model pixel values to be nonnegative)                         */
		/*                                                                    */
		/*  Implement the x and y COM offsets, xoff and yoff, by first        */
		/*  using them to compute xcom_fit and ycom_fit -- the COM position   */
		/*  in the fit image, relative to the center of the fit image -- and  */
		/*  then shifting the resampled region in the *opposite* direction    */
		/*  by the appropriate proportional amount.  Then implement the       */
		/*  "northangle" setting (clockwise heading of north)  by rotating    */
		/*  the resampling grid *counterclockwise* by northangle.             */

		n_pos = pos1->n;
		ncol_fit = frame1->ncol;
		nrow_fit = frame1->nrow;
		xoff = frame1->off[0].val;
		yoff = frame1->off[1].val;
		xcom_fit = (frame1->colcom_vig - (ncol_fit + 1)/2.0) + xoff;
		ycom_fit = (frame1->rowcom_vig - (nrow_fit + 1)/2.0) + yoff;
		resamp_fact = frame1->fit.km_per_pixel / pos1->km_per_pixel;
		resamp_x0 = -xcom_fit*resamp_fact;
		resamp_y0 = -ycom_fit*resamp_fact;
		resamp_xwidth = resamp_fact*(ncol_fit - 1);
		resamp_ywidth = resamp_fact*(nrow_fit - 1);
		resamp_angle = -frame1->northangle;
		resampim( pos1->b, -n_pos, n_pos, -n_pos, n_pos,
				frame1->fit.b, 1, ncol_fit, 1, nrow_fit,
				resamp_x0, resamp_xwidth, resamp_y0, resamp_ywidth, resamp_angle,
				(int) par->poset_resample, (int) par->image_rebin);
		if (par->poset_resample == BICUBIC || par->poset_resample == CUBICCONV) {
			for (k=1; k<=ncol_fit; k++)
				for (l=1; l<=nrow_fit; l++)
					frame1->fit.b[k][l] = MAX( 0.0, frame1->fit.b[k][l]);
		}

		/*  Set the badposet flag if the model plane-of-sky image is
        too small to "contain" all of the sky rendering's nonzero pixels.  */

		if (checkposet( pos1->b, -n_pos, n_pos, -n_pos, n_pos,
				resamp_x0, resamp_xwidth, resamp_y0, resamp_ywidth, resamp_angle,
				&badposet_logfactor_frame))
			par->badposet = 1;

		/*  Output the sky rendering as a pgm or ppm image: This must be done now
        because it will be overwritten as soon as we move to the next epoch
        (if the pos_scope parameter is set to "global").                       */

		if (!par->mark_unseen)
			write_pos_orbit_poset( par, mod1, mod2, mod3, poset1, poset2, poset3, s, f);
	}
}


void calc_orbit_lghtcrv( struct par_t *par,
		struct mod_t *mod1, struct mod_t *mod2, struct mod_t *mod3,
		struct lghtcrv_t *lghtcrv1, struct lghtcrv_t *lghtcrv2,
		struct lghtcrv_t *lghtcrv3, int s)
{
	int n, ncalc, c, i, j, k, l, facetnum;
	double intensityfactor, xecl[3], vecl[3], xobs12[3], xobs13[3], xsun12[3], xsun13[3],
	xsun1[3], xsun2[3], xsun3[3], volume_ratio2, volume_ratio3;
	struct crvrend_t *rend1, *rend2, *rend3;
	struct pos_t *pos1, *pos2, *pos3;

	/*  Get n, the number of observed points for this lightcurve,
      and ncalc, the number of epochs at which model lightcurve
      brightnesses are to be computed                            */

	n = lghtcrv1->n;
	ncalc = lghtcrv1->ncalc;

	/*  Calculate the model lightcurve values at each of the user-specified
      epochs x[i], with i=1,2,...,ncalc; these may or may not be the same as the
      epochs t[i] (i=1,2,...,n) at which actual lightcurve observations were made.  */

	for (i=1; i<=ncalc; i++) {

		rend1 = &lghtcrv1->rend[i];
		rend2 = &lghtcrv2->rend[i];
		rend3 = &lghtcrv3->rend[i];
		pos1 = &rend1->pos;
		pos2 = &rend2->pos;
		pos3 = &rend3->pos;

		/*  Solve the Keplerian orbit at this lightcurve point's epoch x
        (relative displacement xecl and velocity vecl are in ecliptic coordinates)  */

		if (kepler(par->binary_gravparam[0], par->eccentricity[0], par->r_pericenter[0],
				lghtcrv1->x[i] - par->t_pericenter[0],
				par->long_asc_node[0], par->inclination[0], par->arg_pericenter[0],
				xecl, vecl) != 0) {
			printf("Kepler's equation didn't converge for point %d of set %d for body 2\n", i, s);
			bailout("calc_orbit.c\n");
		}

		/*  Convert the relative displacement to observer coordinates
        and also to source (sun) coordinates; note that the oe and se
        coordinate transformation matrices are the same for both models  */

		cotrans( xobs12, rend1->oe, xecl, 1);
		cotrans( xsun12, rend1->se, xecl, 1);

		/*  Repeat all of the above for the third body, if present  */

		if (par->is_triple) {
			if (kepler(par->binary_gravparam[1], par->eccentricity[1], par->r_pericenter[1],
					lghtcrv1->x[i] - par->t_pericenter[1],
					par->long_asc_node[1], par->inclination[1], par->arg_pericenter[1],
					xecl, vecl) != 0) {
				printf("Kepler's equation didn't converge for point %d of set %d for body 3\n", i, s);
				bailout("calc_orbit.c\n");
			}
			cotrans( xobs13, rend1->oe, xecl, 1);
			cotrans( xsun13, rend1->se, xecl, 1);
		}

		/*  If requested, get absolute displacements by assuming
        equal densities; otherwise fix the primary motionless and treat
        the relative motion as the secondary's absolute motion.          */

		if (par->is_triple) {
			volume_ratio2 = mod2->shape.volume /
					(mod1->shape.volume + mod2->shape.volume + mod3->shape.volume);
			volume_ratio3 = mod3->shape.volume /
					(mod1->shape.volume + mod2->shape.volume + mod3->shape.volume);
			for (j=0; j<=2; j++) {
				if (par->orbit_reflex) {
					xobs1[j] = -(volume_ratio2*xobs12[j] + volume_ratio3*xobs13[j]);
					xsun1[j] = -(volume_ratio2*xsun12[j] + volume_ratio3*xsun13[j]);
				} else {
					xobs1[j] = 0.0;
					xsun1[j] = 0.0;
				}
				xobs2[j] = xobs1[j] + xobs12[j];
				xobs3[j] = xobs1[j] + xobs13[j];
				xsun2[j] = xsun1[j] + xsun12[j];
				xsun3[j] = xsun1[j] + xsun13[j];
			}
		} else {
			volume_ratio2 = mod2->shape.volume /
					(mod1->shape.volume + mod2->shape.volume);
			for (j=0; j<=2; j++) {
				if (par->orbit_reflex) {
					xobs1[j] = -volume_ratio2*xobs12[j];
					xsun1[j] = -volume_ratio2*xsun12[j];
				} else {
					xobs1[j] = 0.0;
					xsun1[j] = 0.0;
				}
				xobs2[j] = xobs1[j] + xobs12[j];
				xsun2[j] = xsun1[j] + xsun12[j];
			}
		}

		/*  Assign some values to POS structures  */

		for (j=0; j<=2; j++)
			for (k=0; k<=2; k++) {
				pos1->ae[j][k] = rend1->ae[j][k];
				pos1->oe[j][k] = rend1->oe[j][k];
				pos1->se[j][k] = rend1->se[j][k];
				pos2->ae[j][k] = rend2->ae[j][k];
				pos2->oe[j][k] = rend2->oe[j][k];
				pos2->se[j][k] = rend2->se[j][k];
				if (par->is_triple) {
					pos3->ae[j][k] = rend3->ae[j][k];
					pos3->oe[j][k] = rend3->oe[j][k];
					pos3->se[j][k] = rend3->se[j][k];
				}
			}
		pos1->bistatic = 1;
		pos2->bistatic = 1;
		if (par->is_triple)
			pos3->bistatic = 1;

		/*  Initialize the plane-of-sky views  */

		posclr( pos1);
		posclr( pos2);
		if (par->is_triple)
			posclr( pos3);

		/*  Call routine posvis for the first model to get the facet number,
        scattering angle, and distance toward Earth at the center of
        each POS pixel; set the posbnd parameter to 1 if any portion of
        the model extends beyond the POS frame limits.                    */

		for (c=0; c<mod1->shape.ncomp; c++)
			if (posvis( &mod1->shape.comp[c].real, xobs1, pos1,
					(int) par->pos_smooth, 0, 0, c))
				par->posbnd = 1;

		/*  Now view the model from the source (sun) and get the facet number
        and distance toward the source of each pixel in this projected view;
        use this information to determine which POS pixels are shadowed       */

		if (pos1->bistatic)
			for (c=0; c<mod1->shape.ncomp; c++)
				if (posvis( &mod1->shape.comp[c].real, xsun1, pos1,
						0, 1, 0, c))
					par->posbnd = 1;

		/*  Now add in the POS contributions for the second model.
        This is a bit convoluted because the posvis results depend on
        the ae (body-fixed-to-ecliptic) coordinate transformation
        matrix, which is different for the two models and which is
        stored in the pos structure for this frame.  Hence we have to
        copy the relevant parts of that structure from model 1 to 2,
        then call posvis for the second model.                         */

		copy_pos( pos1, pos2, 3);

		for (c=0; c<mod2->shape.ncomp; c++)
			if (posvis( &mod2->shape.comp[c].real, xobs2, pos2,
					(int) par->pos_smooth, 0, 1, c))
				par->posbnd = 1;

		if (pos2->bistatic)
			for (c=0; c<mod2->shape.ncomp; c++)
				if (posvis( &mod2->shape.comp[c].real, xsun2, pos2,
						0, 1, 1, c))
					par->posbnd = 1;

		/*  Identify and mask out shadowed POS pixels in the second model's
        pos structure, then copy back to the first model's pos structure  */

		if (pos2->bistatic)
			posmask( pos2, par->mask_tol);

		copy_pos( pos2, pos1, 3);

		/*  Now add in the POS contributions for the third model, if present  */

		if (par->is_triple) {
			copy_pos( pos1, pos3, 3);

			for (c=0; c<mod3->shape.ncomp; c++)
				if (posvis( &mod3->shape.comp[c].real, xobs3, pos3,
						(int) par->pos_smooth, 0, 2, c))
					par->posbnd = 1;

			if (pos3->bistatic)
				for (c=0; c<mod3->shape.ncomp; c++)
					if (posvis( &mod3->shape.comp[c].real, xsun3, pos3,
							0, 1, 2, c))
						par->posbnd = 1;

			if (pos3->bistatic)
				posmask( pos3, par->mask_tol);

			copy_pos( pos3, pos1, 3);
		}

		/*  Go through all POS pixels that are visible and unshadowed with
        sufficiently low scattering and incidence angles, and mark the facets
        that project onto their centers as having been "seen" at least once    */

		if (s != par->exclude_seen) {
			for (k=pos1->xlim[0]; k<=pos1->xlim[1]; k++)
				for (l=pos1->ylim[0]; l<=pos1->ylim[1]; l++) {
					if ((pos1->cose[k][l] > par->mincosine_seen)
							&& (pos1->cosi[k][l] > par->mincosine_seen)
							&& (pos1->f[k][l] >= 0)) {
						facetnum = pos1->f[k][l];
						c = pos1->comp[k][l];
						if (pos1->body[k][l] == 0)
							mod1->shape.comp[c].real.f[facetnum].seen = 1;
						else if (pos1->body[k][l] == 1)
							mod2->shape.comp[c].real.f[facetnum].seen = 1;
						else
							mod3->shape.comp[c].real.f[facetnum].seen = 1;
					}
				}
		}

		/*  Compute the model brightness for this model lightcurve point,
        and store it (and the associated sky rendering) in data structure #1  */

		intensityfactor = pow( pos1->km_per_pixel/AU, 2.0);
		lghtcrv1->y[i]  = apply_photo( mod1, lghtcrv1->ioptlaw,
				lghtcrv1->solar_phase[i], intensityfactor,
				pos1, 0, s, i);
		lghtcrv1->y[i] += apply_photo( mod2, lghtcrv1->ioptlaw,
				lghtcrv1->solar_phase[i],
				intensityfactor, pos1, 1, s, i);
		if (par->is_triple)
			lghtcrv1->y[i] += apply_photo( mod3, lghtcrv1->ioptlaw,
					lghtcrv1->solar_phase[i], intensityfactor,
					pos1, 2, s, i);

		/*  If specified, output the model plane-of-sky image for each epoch
        at which a model lightcurve brightness has been calculated: This
        requires POS information which will be overwritten as soon as we
        move to the next epoch (if the pos_scope parameter is set to "global").  */

		if (par->lcrv_pos && !par->mark_unseen)
			write_pos_orbit_lghtcrv( par, mod1, mod2, mod3, lghtcrv1, lghtcrv2, lghtcrv3, s, i);
	}

	/*  Now that we have calculated the model lightcurve brightnesses y at each
      of the epochs x, we use cubic spline interpolation (Numerical Recipes
      routines spline and splint) to get model lightcurve brightness fit[i]
      at each OBSERVATION epoch t[i], with i=1,2,...,n.  This will allow us
      (in routine chi2) to compare model to data (fit[i] to obs[i]) to get
      chi-square.  Note that vector y2 contains the second derivatives of
      the interpolating function at the calculation epochs x.                  */

	spline( lghtcrv1->x, lghtcrv1->y, ncalc, 2.0e30, 2.0e30, lghtcrv1->y2);
	for (i=1; i<=n; i++)
		splint( lghtcrv1->x, lghtcrv1->y, lghtcrv1->y2, ncalc,
				lghtcrv1->t[i][0], &lghtcrv1->fit[i]);
}


void write_pos_orbit_deldop( struct par_t *par,
		struct mod_t *mod1, struct mod_t *mod2, struct mod_t *mod3,
		struct deldop_t *deldop1, struct deldop_t *deldop2,
		struct deldop_t *deldop3, int s, int f)
{
	char name[MAXLEN];
	int color_output;

	if (par->plot_pa[0] || par->plot_pa[1] || par->plot_pa[2]
	                                                       || par->plot_angmom || par->mark_unseen) {
		color_output = 1;
		if (deldop1->nframes > 100)
			sprintf( name, "sky_%02d_%03d.ppm", s, f);
		else
			sprintf( name, "sky_%02d_%02d.ppm", s, f);
	} else {
		color_output = 0;
		if (deldop1->nframes > 100)
			sprintf( name, "sky_%02d_%03d.pgm", s, f);
		else
			sprintf( name, "sky_%02d_%02d.pgm", s, f);
	}
	write_pos_orbit( par, mod1, mod2, mod3,
			&deldop1->frame[f].pos, &deldop2->frame[f].pos,
			&deldop3->frame[f].pos,
			deldop1->frame[f].view[deldop1->v0].intspin,
			deldop2->frame[f].view[deldop2->v0].intspin,
			deldop3->frame[f].view[deldop3->v0].intspin,
			deldop1->iradlaw, color_output, name);
}


void write_pos_orbit_doppler( struct par_t *par,
		struct mod_t *mod1, struct mod_t *mod2, struct mod_t *mod3,
		struct doppler_t *doppler1, struct doppler_t *doppler2,
		struct doppler_t *doppler3, int s, int f)
{
	char name[MAXLEN];
	int color_output;

	if (par->plot_pa[0] || par->plot_pa[1] || par->plot_pa[2]
	                                                       || par->plot_angmom || par->mark_unseen) {
		color_output = 1;
		if (doppler1->nframes > 100)
			sprintf( name, "sky_%02d_%03d.ppm", s, f);
		else
			sprintf( name, "sky_%02d_%02d.ppm", s, f);
	} else {
		color_output = 0;
		if (doppler1->nframes > 100)
			sprintf( name, "sky_%02d_%03d.pgm", s, f);
		else
			sprintf( name, "sky_%02d_%02d.pgm", s, f);
	}
	write_pos_orbit( par, mod1, mod2, mod3,
			&doppler1->frame[f].pos, &doppler2->frame[f].pos,
			&doppler3->frame[f].pos,
			doppler1->frame[f].view[doppler1->v0].intspin,
			doppler2->frame[f].view[doppler2->v0].intspin,
			doppler3->frame[f].view[doppler3->v0].intspin,
			doppler1->iradlaw, color_output, name);
}


void write_pos_orbit_poset( struct par_t *par,
		struct mod_t *mod1, struct mod_t *mod2, struct mod_t *mod3,
		struct poset_t *poset1, struct poset_t *poset2,
		struct poset_t *poset3, int s, int f)
{
	char name[MAXLEN];
	int color_output;

	if (par->plot_pa[0] || par->plot_pa[1] || par->plot_pa[2]
	                                                       || par->plot_angmom || par->mark_unseen) {
		color_output = 1;
		if (poset1->nframes > 100)
			sprintf( name, "sky_%02d_%03d.ppm", s, f);
		else
			sprintf( name, "sky_%02d_%02d.ppm", s, f);
	} else {
		color_output = 0;
		if (poset1->nframes > 100)
			sprintf( name, "sky_%02d_%03d.pgm", s, f);
		else
			sprintf( name, "sky_%02d_%02d.pgm", s, f);
	}
	write_pos_orbit( par, mod1, mod2, mod3,
			&poset1->frame[f].pos, &poset2->frame[f].pos,
			&poset3->frame[f].pos,
			poset1->frame[f].view[poset1->v0].intspin,
			poset2->frame[f].view[poset2->v0].intspin,
			poset3->frame[f].view[poset3->v0].intspin,
			-1, color_output, name);
}


void write_pos_orbit_lghtcrv( struct par_t *par,
		struct mod_t *mod1, struct mod_t *mod2, struct mod_t *mod3,
		struct lghtcrv_t *lghtcrv1, struct lghtcrv_t *lghtcrv2,
		struct lghtcrv_t *lghtcrv3, int s, int i)
{
	char name[MAXLEN];
	int color_output;

	if (par->plot_pa[0] || par->plot_pa[1] || par->plot_pa[2]
	                                                       || par->plot_angmom || par->mark_unseen) {
		color_output = 1;
		if (lghtcrv1->ncalc > 100)
			sprintf( name, "sky_%02d_%03d.ppm", s, i-1);
		else
			sprintf( name, "sky_%02d_%02d.ppm", s, i-1);
	} else {
		color_output = 0;
		if (lghtcrv1->ncalc > 100)
			sprintf( name, "sky_%02d_%03d.pgm", s, i-1);
		else
			sprintf( name, "sky_%02d_%02d.pgm", s, i-1);
	}
	write_pos_orbit( par, mod1, mod2, mod3,
			&lghtcrv1->rend[i].pos, &lghtcrv2->rend[i].pos,
			&lghtcrv3->rend[i].pos,
			lghtcrv1->rend[i].intspin, lghtcrv2->rend[i].intspin,
			lghtcrv3->rend[i].intspin,
			-1, color_output, name);
}


/*  write_pos_orbit writes a model plane-of-sky image of a binary system to
    disk after adding various plotting elements, such as an X that marks
    the subradar point.  It is a modified version of the write_pos routine.  */

void write_pos_orbit( struct par_t *par,
		struct mod_t *mod1, struct mod_t *mod2, struct mod_t *mod3,
		struct pos_t *pos1, struct pos_t *pos2, struct pos_t *pos3,
		double spin_ecl1[3], double spin_ecl2[3], double spin_ecl3[3],
		int iradlaw, int color_output, char *name)
{
	/* max grayscale levels for asteroid surface, unseen asteroid surface,
     highlighted facet, spin vector, COM, subradar point */
	const int levels[6] = {255, 255, 0, 255, 0, 0};

	/* max RGB levels for various plotting elements */
	const int clevels[10][3] = {{255, 255, 255},     /* asteroid surface  */
			{255, 255,   0},     /* unseen ast. surf. */
			{  0,   0, 255},     /* highlighted facet */
			{255,   0, 255},     /* spin vector       */
			{  0,   0,   0},     /* COM               */
			{127,  63,   0},     /* subradar point    */
			{255,   0,   0},     /* long PA           */
			{  0, 255,   0},     /* intermediate PA   */
			{  0, 127, 255},     /* short PA          */
			{127, 255, 255}};    /* angular momentum  */

	unsigned char scatlaw;
	int i, j, n, c, f, pa_indices1[3], pa_indices2[3], pa_indices3[3],
	icolor, do_com, do_subradar, do_pa, do_angmom, is_optical;
	int **color;
	double oa1[3][3], oa2[3][3], oa3[3][3], op1[3][3], op2[3][3], op3[3][3],
	spin_obs1[3], spin_obs2[3], spin_obs3[3], com_obs1[3], com_obs2[3], com_obs3[3],
	pmoment1[3], pmoment2[3], pmoment3[3], ap1[3][3], ap2[3][3], ap3[3][3],
	pa_obs1[3], pa_obs2[3], pa_obs3[3], min_moment, max_moment, posmax, maxbrightness,
	angmom1[3], angmom2[3], angmom3[3], spin_body1[3], spin_body2[3], spin_body3[3];
	double **brightness, **z, ***cbrightness;

	for (i=0; i<=2; i++)
		pa_indices1[0] = -1;  /* avoid compilation warning */

	/*
      Prepare storage matrices to list three quantities for each
         line of sight (each POS pixel):

         1) color type (0 = asteroid surface, 1 = unseen ast. surf., 2 = spin vector, etc.)
         2) brightness (grayscale)
         3) z-coordinate (distance toward observer) of the closest
               plotting element (asteroid surface, arrow, etc.)
	 */

	n = pos1->n;
	color = imatrix( -n, n, -n, n);
	brightness = matrix( -n, n, -n, n);
	z = matrix( -n, n, -n, n);

	/*  Figure out which plotting elements to add  */

	do_com = par->plot_com;
	do_subradar = par->plot_subradar;
	do_pa =  par->plot_pa[0] || par->plot_pa[1] || par->plot_pa[2];
	do_angmom = par->plot_angmom;

	/*  Use the desired scattering law to fill in the asteroid surface  */

	is_optical = (iradlaw < 0);
	scatlaw = (is_optical) ? par->sky_optlaw : par->sky_radlaw;
	maxbrightness = -HUGENUMBER;
	for (i=(-n); i<=n; i++)
		for (j=(-n); j<=n; j++) {
			if (par->mark_unseen) {
				if (pos1->f[i][j] < 0) {
					color[i][j] = 0;       /* blank sky: just use any value */
				} else {
					c = pos1->comp[i][j];
					f = pos1->f[i][j];
					if (pos1->body[i][j] == 0)
						color[i][j] = (mod1->shape.comp[c].real.f[f].seen) ? 0 : 1;
					else if (pos1->body[i][j] == 1)
						color[i][j] = (mod2->shape.comp[c].real.f[f].seen) ? 0 : 1;
					else
						color[i][j] = (mod3->shape.comp[c].real.f[f].seen) ? 0 : 1;
				}
			} else {
				color[i][j] = 0;
			}
			if (scatlaw == OPTICALVIEW)
				brightness[i][j] = pos1->b[i][j];
			else if (scatlaw == RADARVIEW)
				if (pos1->cose[i][j] > 0.0)
					if (pos1->body[i][j] == 0)
						brightness[i][j] = radlaw( &mod1->photo, iradlaw, pos1->cose[i][j],
								pos1->comp[i][j], pos1->f[i][j]);
					else if (pos1->body[i][j] == 1)
						brightness[i][j] = radlaw( &mod2->photo, iradlaw, pos1->cose[i][j],
								pos1->comp[i][j], pos1->f[i][j]);
					else
						brightness[i][j] = radlaw( &mod3->photo, iradlaw, pos1->cose[i][j],
								pos1->comp[i][j], pos1->f[i][j]);
				else
					brightness[i][j] = 0.0;
			else
				brightness[i][j] = pos1->cose[i][j];
			maxbrightness = MAX( maxbrightness, brightness[i][j]);
			z[i][j] = pos1->z[i][j];
		}

	/*  Get pixel value above which image will saturate at white  */

	posmax = (is_optical) ? par->optposmax : par->radposmax;

	/*  Print the name of the image file to the screen, and if the "optposmax" or
      "radposmax" parameter is nonzero (for optical or radar data, respectively),
      also print the value of that parameter vs. the image's maximum pixel value   */

	if (posmax != 0.0)
		printf("# %s  (max pixel value = %10.4e, bright white >= %10.4e)\n",
				name, maxbrightness, posmax);
	else
		printf("# %s\n", name);
	fflush(stdout);

	/*  Get oa = matrix to transform body-fixed to observer coordinates  */

	mtrnsps( oa1, pos1->ae);
	mmmul( oa1, pos1->oe, oa1);
	mtrnsps( oa2, pos2->ae);
	mmmul( oa2, pos2->oe, oa2);
	if (par->is_triple) {
		mtrnsps( oa3, pos3->ae);
		mmmul( oa3, pos3->oe, oa3);
	}

	/*  Convert each body's center of mass to observer coordinates
      and then adjust it for orbital motion                       */

	cotrans( com_obs1, oa1, mod1->shape.com, 1);
	cotrans( com_obs2, oa2, mod2->shape.com, 1);
	for (i=0; i<=2; i++) {
		com_obs1[i] += xobs1[i];
		com_obs2[i] += xobs2[i];
	}
	if (par->is_triple) {
		cotrans( com_obs3, oa3, mod3->shape.com, 1);
		for (i=0; i<=2; i++)
			com_obs3[i] += xobs3[i];
	}

	/*  Add each of the plotting elements that has been specified  */

	if (par->plot_spinvec) {

		/*  arrow along spin vector  */

		cotrans( spin_obs1, pos1->oe, spin_ecl1, 1);
		cotrans( spin_obs2, pos2->oe, spin_ecl2, 1);
		plot_arrow( 3, com_obs1, spin_obs1, 1, par->pierce_spinvec, par->orbit_posfactor[0],
				maxbrightness, posmax, pos1, color, brightness, z);
		plot_arrow( 3, com_obs2, spin_obs2, 1, par->pierce_spinvec, par->orbit_posfactor[1],
				maxbrightness, posmax, pos1, color, brightness, z);
		if (par->is_triple) {
			cotrans( spin_obs3, pos3->oe, spin_ecl3, 1);
			plot_arrow( 3, com_obs3, spin_obs3, 1, par->pierce_spinvec, par->orbit_posfactor[2],
					maxbrightness, posmax, pos1, color, brightness, z);
		}
	}

	if (do_com) {

		/*  cross at projected center of mass  */

		plot_com( 4, 0, com_obs1, par->orbit_posfactor[0], maxbrightness,
				pos1, color, brightness, z);
		plot_com( 4, 1, com_obs2, par->orbit_posfactor[1], maxbrightness,
				pos1, color, brightness, z);
		if (par->is_triple)
			plot_com( 4, 2, com_obs3, par->orbit_posfactor[2], maxbrightness,
					pos1, color, brightness, z);
	}

	if (do_subradar) {

		/*  X at subradar/subearth point  */

		plot_subradar( 5, 0, par->orbit_posfactor[0], maxbrightness, posmax,
				pos1, color, brightness, z);
		plot_subradar( 5, 1, par->orbit_posfactor[1], maxbrightness, posmax,
				pos1, color, brightness, z);
		if (par->is_triple)
			plot_subradar( 5, 2, par->orbit_posfactor[2], maxbrightness, posmax,
					pos1, color, brightness, z);
	}

	if (do_pa) {

		/*  cylindrical shaft (headless arrow) along positive end
        of one or more principal axes                          */

		/*  First, take the inertia tensor (computed assuming uniform density),
        diagonalize it in order to get the principal moments (pmoment)
        and the principal-axis-to-body-fixed coordinate transformation matrix (ap)  */

		diag_inertia( mod1->shape.inertia, pmoment1, ap1);
		diag_inertia( mod2->shape.inertia, pmoment2, ap2);
		if (par->is_triple)
			diag_inertia( mod3->shape.inertia, pmoment3, ap3);

		/*  Get op = matrix to transform principal-axis to observer coordinates  */

		mmmul( op1, oa1, ap1);
		mmmul( op2, oa2, ap2);
		if (par->is_triple)
			mmmul( op3, oa3, ap3);

		/*  Determine which principal axis is long vs. intermediate vs. short  */

		min_moment = 1.0e20;
		for (i=0; i<=2; i++)
			if (pmoment1[i] < min_moment) {
				min_moment = pmoment1[i];
				pa_indices1[0] = i;
			}
		max_moment = -1.0e20;
		for (i=0; i<=2; i++)
			if (i != pa_indices1[0] && pmoment1[i] >= max_moment) {
				max_moment = pmoment1[i];
				pa_indices1[2] = i;
			}
		for (i=0; i<=2; i++)
			if (i != pa_indices1[0] && i != pa_indices1[2])
				pa_indices1[1] = i;

		min_moment = 1.0e20;
		for (i=0; i<=2; i++)
			if (pmoment2[i] < min_moment) {
				min_moment = pmoment2[i];
				pa_indices2[0] = i;
			}
		max_moment = -1.0e20;
		for (i=0; i<=2; i++)
			if (i != pa_indices2[0] && pmoment2[i] >= max_moment) {
				max_moment = pmoment2[i];
				pa_indices2[2] = i;
			}
		for (i=0; i<=2; i++)
			if (i != pa_indices2[0] && i != pa_indices2[2])
				pa_indices2[1] = i;

		if (par->is_triple) {
			min_moment = 1.0e20;
			for (i=0; i<=2; i++)
				if (pmoment3[i] < min_moment) {
					min_moment = pmoment3[i];
					pa_indices3[0] = i;
				}
			max_moment = -1.0e20;
			for (i=0; i<=2; i++)
				if (i != pa_indices3[0] && pmoment3[i] >= max_moment) {
					max_moment = pmoment3[i];
					pa_indices3[2] = i;
				}
			for (i=0; i<=2; i++)
				if (i != pa_indices3[0] && i != pa_indices3[2])
					pa_indices3[1] = i;
		}

		/*  Create the cylindrical shaft(s)  */

		for (i=0; i<=2; i++) {
			if (par->plot_pa[pa_indices1[i]]) {
				for (j=0; j<=2; j++)
					pa_obs1[j] = op1[j][pa_indices1[i]];
				plot_arrow( 6+pa_indices1[i], com_obs1, pa_obs1, 0, 0, par->orbit_posfactor[0],
						maxbrightness, posmax, pos1, color, brightness, z);
			}
		}
		for (i=0; i<=2; i++) {
			if (par->plot_pa[pa_indices2[i]]) {
				for (j=0; j<=2; j++)
					pa_obs2[j] = op2[j][pa_indices2[i]];
				plot_arrow( 6+pa_indices2[i], com_obs2, pa_obs2, 0, 0, par->orbit_posfactor[1],
						maxbrightness, posmax, pos1, color, brightness, z);
			}
		}
		if (par->is_triple) {
			for (i=0; i<=2; i++) {
				if (par->plot_pa[pa_indices3[i]]) {
					for (j=0; j<=2; j++)
						pa_obs3[j] = op3[j][pa_indices3[i]];
					plot_arrow( 6+pa_indices3[i], com_obs3, pa_obs3, 0, 0, par->orbit_posfactor[2],
							maxbrightness, posmax, pos1, color, brightness, z);
				}
			}
		}
	}

	if (do_angmom) {

		/*  arrow along angular momentum vector  */

		/*  Transform the intrinsic spin vector from ecliptic to body-fixed coordinates  */

		cotrans( spin_body1, pos1->ae, spin_ecl1, 1);
		cotrans( spin_body2, pos2->ae, spin_ecl2, 1);
		if (par->is_triple)
			cotrans( spin_body3, pos3->ae, spin_ecl3, 1);

		/*  Compute the angular momentum vector in principal-axis coordinates

        For an NPA rotator, the spin state is evolved (via Euler's equations)
        with body-fixed coordinates treated as principal-axis coordinates;
        so to ensure that the angular momentum vector is constant in time,
        we must also equate these two coordinate systems here.  (For a
        PA rotator it doesn't matter: only one spin component is nonzero.)     */

		for (i=0; i<=2; i++) {
			angmom1[i] = mod1->spin.inertia[i].val * spin_body1[i];
			angmom2[i] = mod2->spin.inertia[i].val * spin_body2[i];
			if (par->is_triple)
				angmom3[i] = mod3->spin.inertia[i].val * spin_body3[i];
		}

		/*  Transform the angular momentum vector to observer coordinates  */

		cotrans( angmom1, oa1, angmom1, 1);
		cotrans( angmom2, oa2, angmom2, 1);
		if (par->is_triple)
			cotrans( angmom3, oa3, angmom3, 1);

		/*  Create the arrow  */

		plot_arrow( 9, com_obs1, angmom1, 1, par->pierce_angmom, par->orbit_posfactor[0],
				maxbrightness, posmax, pos1, color, brightness, z);
		plot_arrow( 9, com_obs2, angmom2, 1, par->pierce_angmom, par->orbit_posfactor[1],
				maxbrightness, posmax, pos1, color, brightness, z);
		if (par->is_triple)
			plot_arrow( 9, com_obs3, angmom3, 1, par->pierce_angmom, par->orbit_posfactor[2],
					maxbrightness, posmax, pos1, color, brightness, z);
	}

	/*  Output to disk  */

	if (color_output) {

		/*  color (ppm) output  */

		/*  Set RGB levels according to the maximum RGB levels and
          the relative brightness values determined above         */

		cbrightness = d3tensor( -n, n, -n, n, 0, 2);

		for (i=(-n); i<=n; i++)
			for (j=(-n); j<=n; j++)
				for (icolor=0; icolor<=2; icolor++)
					cbrightness[i][j][icolor] = brightness[i][j]*clevels[color[i][j]][icolor]/255.0;

		if (posmax == 0.0)
			wimasppm0( cbrightness, -n, n, -n, n, 0, 0, 0, name);
		else
			wimasppmsc( cbrightness, -n, n, -n, n, 0.0, posmax, 0, 0, 0, name);

		free_d3tensor( cbrightness, -n, n, -n, n, 0, 2);

	} else {

		/*  grayscale (pgm) output  */

		/*  Set grayscale levels according to the maximum levels and
          the relative brightness values determined above           */

		for (i=(-n); i<=n; i++)
			for (j=(-n); j<=n; j++)
				brightness[i][j] = brightness[i][j]*levels[color[i][j]]/255.0;

		if (posmax == 0.0)
			wimaspgm0( brightness, -n, n, -n, n, 0, 0, 0, name);
		else
			wimaspgmsc( brightness, -n, n, -n, n, 0.0, posmax, 0, 0, 0, name);
	}

	/*  Clean up storage space  */

	free_imatrix( color, -n, n, -n, n);
	free_matrix( brightness, -n, n, -n, n);
	free_matrix( z, -n, n, -n, n);
}


/*  Copy POS structure pos1 to pos2                */
/*                                                 */
/*  mode = 1: copy POS frame as viewed from Earth  */
/*         2: copy POS frame as viewed from Sun    */      
/*         3: copy both                            */

void copy_pos( struct pos_t *pos1, struct pos_t *pos2, int mode)
{
	int n, i, j;

	if (pos1->n != pos2->n) {
		printf("ERROR in copy_pos: different POS matrix dimensions (n=%d vs. n=%d)\n",
				pos1->n, pos2->n);
		bailout("calc_orbit.c\n");
	}

	n = pos1->n;

	if (mode == 1 || mode == 3) {
		for (i=-n; i<=n; i++)
			for (j=-n; j<=n; j++) {
				pos2->cosi[i][j] = pos1->cosi[i][j];
				pos2->cose[i][j] = pos1->cose[i][j];
				pos2->z[i][j]    = pos1->z[i][j];
				pos2->body[i][j] = pos1->body[i][j];
				pos2->comp[i][j] = pos1->comp[i][j];
				pos2->f[i][j]    = pos1->f[i][j];
			}
		pos2->xlim[0] = pos1->xlim[0];
		pos2->xlim[1] = pos1->xlim[1];
		pos2->ylim[0] = pos1->ylim[0];
		pos2->ylim[1] = pos1->ylim[1];
	}

	if (mode == 2 || mode == 3) {
		for (i=-n; i<=n; i++)
			for (j=-n; j<=n; j++) {
				pos2->cosill[i][j]  = pos1->cosill[i][j];
				pos2->zill[i][j]    = pos1->zill[i][j];
				pos2->bodyill[i][j] = pos1->bodyill[i][j];
				pos2->compill[i][j] = pos1->compill[i][j];
				pos2->fill[i][j]    = pos1->fill[i][j];
			}
		pos2->xlim2[0] = pos1->xlim2[0];
		pos2->xlim2[1] = pos1->xlim2[1];
		pos2->ylim2[0] = pos1->ylim2[0];
		pos2->ylim2[1] = pos1->ylim2[1];
	}
}
