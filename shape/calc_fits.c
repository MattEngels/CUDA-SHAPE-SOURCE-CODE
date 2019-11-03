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

#include "head.h"

void calc_fits( struct par_t *par, struct mod_t *mod, struct dat_t *dat)
{
	int c, f, s, frm, i;

	/* Initialize flags that indicate the model extends beyond POS frame, that
	 * plane-of-sky fit images are too small to "contain" the target, and that
	 * model is too wide in (delay-)Doppler space to create (delay-)Doppler fit
	 * frames                                  */

	par->posbnd = 0;
	par->badposet = 0;
	par->badradar = 0;
	par->posbnd_logfactor = 0.0;
	par->badposet_logfactor = 0.0;
	par->badradar_logfactor = 0.0;

	/*  Initialize the flags that indicate whether or not each facet
      of each model component is ever visible and unshadowed from Earth  */

	for (c=0; c<mod->shape.ncomp; c++)
		for (f=0; f<mod->shape.comp[c].real.nf; f++)
			mod->shape.comp[c].real.f[f].seen = 0;

	/*  For the "write" action, announce whether or not epochs
      have been corrected for one-way light travel time,
      and create the "listpos_path" directory if necessary    */

	if (par->action == WRITE) {
		printf("#\n");
		if (par->perform_ltc) {
			printf("# Epochs listed below are times when light left the target\n");
		} else {
			printf("# Epochs listed below are times when a terrestrial observer\n");
			printf("# would receive light from the target\n");
		}
		printf("#\n");

		if (par->listpos_deldop || par->listpos_opt)
			if (!createdir(par->listpos_path)) {
				printf("Unable to create 'listpos_path' directory\n");
				bailout("calc_fits.c: program halted\n");
			}
	}

	/*  Calculate the fits for each dataset in turn  */

	for (s=0; s<dat->nsets; s++) {
		switch (dat->set[s].type) {
		case DELAY:
			calc_deldop( par, mod, &dat->set[s].desc.deldop, s);
			break;
		case DOPPLER:
			calc_doppler( par, mod, &dat->set[s].desc.doppler, s);
			break;
		case POS:
			calc_poset( par, mod, &dat->set[s].desc.poset, s);
			break;
		case LGHTCRV:
			calc_lghtcrv( par, mod, &dat->set[s].desc.lghtcrv, s);
			break;
		default:
			bailout("calc_fits.c: can't handle this type yet\n");
		}
	}
	/*  Complete the calculations of values that will be used during a fit
      to increase the objective function for models with bad properties   */

	par->posbnd_logfactor /= dat->dof;
	par->badposet_logfactor /= dat->dof_poset;
	par->badradar_logfactor /= (dat->dof_deldop + dat->dof_doppler);
	fflush(stdout);

	/*  For the "write" action with the "mark_unseen" parameter
      turned on, go back and create the colored POS images     */

	if (par->action == WRITE && par->mark_unseen) {
		for (s=0; s<dat->nsets; s++) {
			switch (dat->set[s].type) {
			case DELAY:
				for (frm=0; frm<dat->set[s].desc.deldop.nframes; frm++)
					write_pos_deldop( par, mod, &dat->set[s].desc.deldop, s, frm);
				break;
			case DOPPLER:
				for (frm=0; frm<dat->set[s].desc.doppler.nframes; frm++)
					write_pos_doppler( par, mod, &dat->set[s].desc.doppler, s, frm);
				break;
			case POS:
				for (frm=0; frm<dat->set[s].desc.poset.nframes; frm++)
					write_pos_poset( par, mod, &dat->set[s].desc.poset, s, frm);
				break;
			case LGHTCRV:
				if (par->lcrv_pos)
					for (i=1; i<=dat->set[s].desc.lghtcrv.ncalc; i++)
						write_pos_lghtcrv( par, mod, &dat->set[s].desc.lghtcrv, s, i);
				break;
			}
		}
	}
}

void calc_deldop( struct par_t *par, struct mod_t *mod, struct deldop_t *deldop,
		int s)
{
	const char *monthName[12] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
			"Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
	double orbit_offset[3] = {0.0, 0.0, 0.0};

	FILE *fpdel, *fpdop;
	char tempstring[MAXLEN], name[MAXLEN];
	int year, mon, day, hour, min, sec, f, i, j, c, min_nsinc2, k, l,
	n_nonblankpixels, facetnum, x, y, v, v2;
	double w[3], dopfact, ax, ay, ax_abs, ay_abs, a_max, spin_colat, spin_azim,
	projected_area, oa[3][3], to_earth[3], to_earth_lat, to_earth_long,
	rotphase, LEtoCOM_usec, LEtoCOM_km, LEtoTE_usec, LEtoTE_km,
	bandwidth, folded_bandwidth, breadth, cross_section, phi, theta, psi,
	intspin_body[3], delfact, delshift, dopshift, delPOS_usec, dopPOS_Hz,
	overflow_o2_store, overflow_m2_store, overflow_xsec_store,
	overflow_delmean_store, overflow_dopmean_store;
	double **fit_store;
	struct deldopfrm_t *frame;
	struct deldopview_t *view0;
	struct pos_t *pos;

	overflow_o2_store = overflow_m2_store = overflow_xsec_store
	= overflow_delmean_store = overflow_dopmean_store = 0.0;

	for (f=0; f<deldop->nframes; f++) {

		frame = &deldop->frame[f];
		view0 = &frame->view[deldop->v0];
		pos = &frame->pos;

		/*  If smearing is being modeled, initialize variables that
        will be used to sum results calculated for individual views  */

		if (deldop->nviews > 1) {
			fit_store = matrix( 1, frame->ndel, 1, frame->ndop);
			for (i=1; i<=frame->ndel; i++)
				for (j=1; j<=frame->ndop; j++)
					fit_store[i][j] = 0.0;
			overflow_o2_store = overflow_m2_store = overflow_xsec_store
					= overflow_delmean_store = overflow_dopmean_store = 0.0;
		}

		/*  Loop over all views for this (smeared) frame, going in an order that
        ends with the view corresponding to the epoch listed for this frame
        in the obs file; this way we can use the calculated information for
        that view in the "write" action screen and disk output that follows   */

		for (v2=deldop->v0+1; v2<=deldop->v0+deldop->nviews; v2++) {
			v = v2 % deldop->nviews;

			for (i=0; i<=2; i++)
				for (j=0; j<=2; j++) {
					pos->ae[i][j] = frame->view[v].ae[i][j];
					pos->oe[i][j] = frame->view[v].oe[i][j];
				}
			pos->bistatic = 0;

			/*  Initialize the plane-of-sky view  */

			posclr( pos);

			/*  Call routine posvis to get the facet number, scattering angle,
          and distance toward Earth at the center of each POS pixel;
          set the posbnd parameter to 1 if any portion of the model
          extends beyond the POS frame limits.                            */

			for (c=0; c<mod->shape.ncomp; c++)
				if (posvis( &mod->shape.comp[c].real, orbit_offset, pos,
						(int) par->pos_smooth, 0, 0, c) && v == deldop->v0) {
					par->posbnd = 1;
					par->posbnd_logfactor += frame->dof * pos->posbnd_logfactor;
				}

			/*  Go through all POS pixels which are visible with sufficiently low
          scattering angle, and mark the facets which project onto their
          centers as having been "seen" at least once                        */

			if (s != par->exclude_seen && v == deldop->v0) {
				for (k=pos->xlim[0]; k<=pos->xlim[1]; k++)
					for (l=pos->ylim[0]; l<=pos->ylim[1]; l++) {
						if ((pos->cose[k][l] > par->mincosine_seen)
								&& (pos->f[k][l] >= 0)) {
							facetnum = pos->f[k][l];
							c = pos->comp[k][l];
							mod->shape.comp[c].real.f[facetnum].seen = 1;
						}
					}
			}

			/*  Zero out the fit delay-Doppler image, then call pos2deldop
          to create the fit image by mapping power from the plane of
          the sky to delay-Doppler space.                             */

			clrmat( frame->fit, 1, frame->ndel, 1, frame->ndop);
			if (HMT) {
				if (pos2deldop_hmt( par, &mod->photo, 0.0, 0.0, 0.0, deldop, 0, s, f, v)) {
					par->badradar = 1;
					par->badradar_logfactor += frame->dof * frame->badradar_logfactor
							/ deldop->nviews;
				}
			}
			else {
				if (pos2deldop( par, &mod->photo, 0.0, 0.0, 0.0, deldop, 0, s, f, v)) {
					par->badradar = 1;
					par->badradar_logfactor += frame->dof * frame->badradar_logfactor
							/ deldop->nviews;
				}
			}

			/*  If smearing is being modeled, include the delay-Doppler
          calculations from this view in the summed results for this frame  */

			if (deldop->nviews > 1) {
				for (i=1; i<=frame->ndel; i++)
					for (j=1; j<=frame->ndop; j++)
						fit_store[i][j] += frame->fit[i][j];
				overflow_o2_store += frame->overflow_o2;
				overflow_m2_store += frame->overflow_m2;
				overflow_xsec_store += frame->overflow_xsec;
				overflow_delmean_store += frame->overflow_delmean;
				overflow_dopmean_store += frame->overflow_dopmean;
			}

		}

		/*  If smearing is being modeled, compute mean values over all views
        for this frame and store them in the standard frame structure     */

		if (deldop->nviews > 1) {
			for (i=1; i<=frame->ndel; i++)
				for (j=1; j<=frame->ndop; j++)
					frame->fit[i][j] = fit_store[i][j] / deldop->nviews;
			free_matrix( fit_store, 1, frame->ndel, 1, frame->ndop);
			frame->overflow_o2 = overflow_o2_store / deldop->nviews;
			frame->overflow_m2 = overflow_m2_store / deldop->nviews;
			frame->overflow_xsec = overflow_xsec_store / deldop->nviews;
			frame->overflow_delmean = overflow_delmean_store / deldop->nviews;
			frame->overflow_dopmean = overflow_dopmean_store / deldop->nviews;
		}

		/*  Carry out a gamma transformation on the fit image if requested  */

		if (par->dd_gamma != 1.0) {
			for (i=1; i<=frame->ndel; i++)
				for (j=1; j<=frame->ndop; j++)
					gamma_trans( &frame->fit[i][j], par->dd_gamma);
		}

		/*  Perform some output tasks for the "write" action:
        Some of these tasks require POS information which will
        be overwritten as soon as we move to the next frame
        (if the pos_scope parameter is set to "global").        */

		if (par->action == WRITE) {

			/*  Display the observation epoch for this frame  */

			jd2cal( &year, &mon, &day, &hour, &min, &sec, view0->t);
			printf("# %02d_%02d JD %.5lf (UT %4d %s %2.2d %2.2d:%2.2d:%2.2d)\n",
					s, f, view0->t,
					year, monthName[mon-1], day, hour, min, sec);
			fflush(stdout);

			/*  For an NPA rotator, display the Euler angles
          of the body-fixed axes in ecliptic coordinates  */

			if (!mod->spin.pa) {
				mat2euler( view0->ae, &phi, &theta, &psi);
				if (phi < 0.0)
					phi += 2*PIE;
				if (psi < 0.0)
					psi += 2*PIE;
				printf("#       Euler angles for body-fixed axes:  %f %f %f deg\n",
						phi*R2D, theta*R2D, psi*R2D);
				fflush(stdout);
			}

			/*  For an NPA rotator, or a PA rotator with spin impulses used,
          display the sidereal spin vector in body-fixed coordinates    */

			if (!mod->spin.pa || mod->spin.n_impulse > 0) {
				cotrans( intspin_body, view0->ae, view0->intspin, 1);
				printf("#       sidereal spin vector (body-fixed): %f %f %f deg/day\n",
						intspin_body[0]*R2D, intspin_body[1]*R2D, intspin_body[2]*R2D);
				fflush(stdout);
			}

			/*  Display the linear-distance-to-Doppler proportionality factor and the
          orientation of the apparent spin vector in observer coordinates (w).   */

			cotrans( w, view0->oe, view0->spin, 1);
			spin_colat = 90.0 - atan2(w[2], sqrt(w[0]*w[0] + w[1]*w[1]))*R2D;
			spin_azim = atan2(w[1], w[0])*R2D - 90.0;
			if (spin_azim < 0.0)
				spin_azim += 360.0;
			printf("#       delcorr %f usec  dopcorr %f Hz  dopconv %f Hz/km\n",
					view0->deloff * deldop->del_per_pixel,
					view0->dopoff * deldop->dop_per_pixel, view0->km2Hz);
			printf("#       app. spin vector points %7.3f deg from Earth at %7.3f deg E of N\n",
					spin_colat, spin_azim);
			fflush(stdout);

			/*  Display the delay (and distance) from the leading edge to the
          center of mass and from the leading edge to the trailing edge;
          the delay limits were computed PRIOR to convolution with the
          delay response function.                                        */

			LEtoCOM_usec = -frame->dellim[0];
			LEtoCOM_km = LEtoCOM_usec / KM2US;
			LEtoTE_usec = frame->dellim[1] - frame->dellim[0];
			LEtoTE_km = LEtoTE_usec / KM2US;
			printf("#       LE to COM %f usec (%f km); LE to TE %f usec (%f km)\n",
					LEtoCOM_usec, LEtoCOM_km, LEtoTE_usec, LEtoTE_km);

			/*  Display the instantaneous zero-crossing bandwidth, the
          instantaneous maximum breadth, and the instantaneous folded
          zero-crossing bandwidth; the Doppler limits were computed PRIOR
          to convolution with the (truncated) Doppler response function.   */

			bandwidth = frame->doplim[1] - frame->doplim[0];  /* Hz */
			printf("#       bandwidth %f Hz", bandwidth);
			if (spin_colat > 0.01 && spin_colat < 179.99) {
				breadth = bandwidth
						/ (deldop->dopscale.val * KM2HZFACT
								* fabs(mod->spin.omega[2].val)
				* deldop->Ftx * sin(spin_colat*D2R));        /* km */
				printf(" (breadth %f km)", breadth);
			} else {
				breadth = 0.0;
			}
			folded_bandwidth = 2 * MAX(  frame->doplim[1],
					-frame->doplim[0] );  /* Hz */
					printf(", folded bandwidth %f Hz\n", folded_bandwidth);

					/*  Display the cross section, projected area, and albedo  */

					cross_section = frame->overflow_xsec;
					for (i=1; i<=frame->ndel; i++)
						for (j=1; j<=frame->ndop; j++)
							cross_section += frame->fit[i][j];
					cross_section *= frame->cal.val;

					n_nonblankpixels = 0;
					for (k=pos->xlim[0]; k<=pos->xlim[1]; k++)
						for (l=pos->ylim[0]; l<=pos->ylim[1]; l++)
							if (pos->cose[k][l] > 0.0)
								n_nonblankpixels++;
					projected_area = n_nonblankpixels * pos->km_per_pixel
							* pos->km_per_pixel;

					if (frame->sdev == 1.0 && frame->cal.state == 'f') {

						/* uncalibrated: dummy sdev value, floating calfact value */

						printf("#       proj area %12.6e km^2\n", projected_area);
					} else {
						printf("#       x-section %12.6e km^2  proj area %12.6e km^2",
								cross_section, projected_area);
						if (projected_area > 0.0)
							printf("  albedo %f", cross_section/projected_area);
						printf("\n");
					}
					fflush(stdout);

					/*  Display the body-fixed angular coordinates of the target-to-Earth line  */
					/*                                                                          */
					/*        oa = matrix to transform body-fixed to observer coordinates       */
					/*  to_earth = normalized target-to-Earth vector in body-fixed coords       */

					mtrnsps( oa, view0->ae);
					mmmul( oa, view0->oe, oa);
					for (j=0; j<=2; j++)
						to_earth[j] = oa[2][j];
					to_earth_lat = atan2( to_earth[2], sqrt(to_earth[0]*to_earth[0] +
							to_earth[1]*to_earth[1]   ))*R2D;
					to_earth_long = atan2( to_earth[1], to_earth[0])*R2D;
					if (to_earth_long < 0.0)
						to_earth_long += 360.0;
					rotphase = (360.0 - to_earth_long) + par->delta_rotphase*R2D;
					rotphase -= 360.0*floor(rotphase/360.0);
					printf("#       LOS at body-fixed long/lat = (%7.3f, %+7.3f) deg, rot phase %7.3f deg\n",
							to_earth_long, to_earth_lat, rotphase);
					fflush(stdout);

					/*  Compute the Doppler bin increments per POS pixel westward (ax)
          and northward (ay).  Use these to see if the sinc^2 response
          function is being evaluated at enough points per POS pixel.     */

					dopfact = deldop->dopscale.val * KM2HZFACT * pos->km_per_pixel
							* deldop->Ftx / deldop->dop_per_pixel;
					ax = -w[1]*dopfact;
					ay = w[0]*dopfact;
					ax_abs = fabs(ax);
					ay_abs = fabs(ay);
					a_max = (ax_abs > ay_abs) ? ax_abs : ay_abs;
					min_nsinc2 = (int) floor(2*a_max + 0.5);     /* two points per Doppler bin */
					if (par->nsinc2 < min_nsinc2)
						printf("Each POS pixel spans ~ %.2f Doppler bins:"
								" use nsinc2 = %d (or smaller pixels)\n", a_max, min_nsinc2);

					/*  If desired, write delay and Doppler values at the center
          of each POS pixel to two separate disk files              */

					if (par->listpos_deldop) {

						/*  Get conversion from km towards radar (observer z) to delay bins  */

						delfact = -KM2US/deldop->del_per_pixel;

						/*  Get the COM delay and Doppler bins, corrected for ephemeris drift  */

						delshift = frame->delcom_vig + view0->deloff;
						dopshift = frame->dopcom_vig + view0->dopoff;

						/*  Create the two output filenames (including the path),
            open the files, and write the headers                  */

						if (deldop->nframes > 100)
							sprintf( tempstring, "del_%02d_%03d.posdat", s, f);
						else
							sprintf( tempstring, "del_%02d_%02d.posdat", s, f);
						changepath( name, MAXLEN, tempstring, par->listpos_path);
						FOPEN( fpdel, name, "w");
						fprintf(fpdel, "# pos_pixels %4d  pos_width %9.6f km\n",
								par->pos_pixels, par->pos_width);

						if (deldop->nframes > 100)
							sprintf( tempstring, "dop_%02d_%03d.posdat", s, f);
						else
							sprintf( tempstring, "dop_%02d_%02d.posdat", s, f);
						changepath( name, MAXLEN, tempstring, par->listpos_path);
						FOPEN( fpdop, name, "w");
						fprintf(fpdop, "# pos_pixels %4d  pos_width %9.6f km\n",
								par->pos_pixels, par->pos_width);

						/*  Write the delay (usec) at the center of each POS pixel to one file and the
            Doppler (Hz) to the other file, starting at the southeast corner of the
            POS image and moving east to west and then south to north.  Note that
            functions posclr and posvis flag blank-sky pixels by assigning "cose" =
            cos(scattering angle) = 0, in which case absurdly large negative delay and
            Doppler values are written to disk.                                         */

						for (y=-pos->n; y<=pos->n; y++)
							for (x=-pos->n; x<=pos->n; x++) {
								if (pos->cose[x][y] > 0.0) {
									delPOS_usec = (pos->z[x][y]*delfact + delshift) * deldop->del_per_pixel;
									dopPOS_Hz = (ax*x + ay*y + dopshift) * deldop->dop_per_pixel;
								} else {
									delPOS_usec = dopPOS_Hz = -HUGENUMBER;  /* blank sky */
								}
								fprintf(fpdel, "%13.6e\n", delPOS_usec);
								fprintf(fpdop, "%13.6e\n", dopPOS_Hz);
							}

						/*  Close the files  */

						fclose( fpdel);
						fclose( fpdop);
					}

					/*  Create a plane-of-sky image that assumes front illumination and
          Lambert scattering: pixel brightness level proportional to
          cos(scattering angle).  This must be done now if the "pos_scope"
          parameter is set to "global" because the POS frame will be
          overwritten as soon as we move to the next observation epoch.     */

					if (!par->mark_unseen)
						write_pos_deldop( par, mod, deldop, s, f);

		}  /* end code for "write" action */

	}  /* end loop over frames */
}

void calc_doppler( struct par_t *par, struct mod_t *mod, struct doppler_t *doppler,
		int s)
{
	const char *monthName[12] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
			"Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
	double orbit_offset[3] = {0.0, 0.0, 0.0};

	FILE *fpdop;
	char tempstring[MAXLEN], name[MAXLEN];
	int year, mon, day, hour, min, sec, f, i, j, c, min_nsinc2, k, l,
	n_nonblankpixels, facetnum, x, y, v, v2;
	double w[3], dopfact, ax, ay, ax_abs, ay_abs, a_max, spin_colat, spin_azim, projected_area,
	oa[3][3], to_earth[3], to_earth_lat, to_earth_long, rotphase, bandwidth,
	folded_bandwidth, breadth, cross_section, phi, theta, psi, intspin_body[3],
	dopshift, dopPOS_Hz, overflow_o2_store, overflow_m2_store, overflow_xsec_store,
	overflow_delmean_store, overflow_dopmean_store;
	double *fit_store;
	struct dopfrm_t *frame;
	struct dopview_t *view0;
	struct pos_t *pos;

	for (f=0; f<doppler->nframes; f++) {

		frame = &doppler->frame[f];
		view0 = &frame->view[doppler->v0];
		pos = &frame->pos;

		/*  If smearing is being modeled, initialize variables that
        will be used to sum results calculated for individual views  */

		if (doppler->nviews > 1) {
			fit_store = vector( 1, frame->ndop);
			for (j=1; j<=frame->ndop; j++)
				fit_store[j] = 0.0;
			overflow_o2_store = overflow_m2_store = overflow_xsec_store
					= overflow_dopmean_store = 0.0;
		}

		/*  Loop over all views for this (smeared) frame, going in an order that
        ends with the view corresponding to the epoch listed for this frame
        in the obs file; this way we can use the calculated information for
        that view in the "write" action screen and disk output that follows   */

		for (v2=doppler->v0+1; v2<=doppler->v0+doppler->nviews; v2++) {
			v = v2 % doppler->nviews;

			for (i=0; i<=2; i++)
				for (j=0; j<=2; j++) {
					pos->ae[i][j] = frame->view[v].ae[i][j];
					pos->oe[i][j] = frame->view[v].oe[i][j];
				}
			pos->bistatic = 0;

			/*  Initialize the plane-of-sky view  */

			posclr( pos);

			/*  Call routine posvis to get the facet number, scattering angle,
          and distance toward Earth at the center of each POS pixel;
          set the posbnd parameter to 1 if any portion of the model
          extends beyond the POS frame limits.                            */

			for (c=0; c<mod->shape.ncomp; c++)
				if (posvis( &mod->shape.comp[c].real, orbit_offset, pos,
						(int) par->pos_smooth, 0, 0, c) && v == doppler->v0) {
					par->posbnd = 1;
					par->posbnd_logfactor += frame->dof * pos->posbnd_logfactor;
				}

			/*  Go through all POS pixels which are visible with sufficiently low
          scattering angle, and mark the facets which project onto their
          centers as having been "seen" at least once                        */

			if (s != par->exclude_seen && v == doppler->v0) {
				for (k=pos->xlim[0]; k<=pos->xlim[1]; k++)
					for (l=pos->ylim[0]; l<=pos->ylim[1]; l++) {
						if ((pos->cose[k][l] > par->mincosine_seen)
								&& (pos->f[k][l] >= 0)) {
							facetnum = pos->f[k][l];
							c = pos->comp[k][l];
							mod->shape.comp[c].real.f[facetnum].seen = 1;
						}
					}
			}

			/*  Zero out the fit Doppler spectrum, then call pos2doppler
          to create the fit spectrum by mapping power from the
          plane of the sky to Doppler space.                        */

			clrvect( frame->fit, 1, frame->ndop);

			if (HMT) {
				if (pos2doppler_hmt( par, &mod->photo, 0.0, 0.0, 0.0, doppler, 0, s, f, v)) {
					par->badradar = 1;
					par->badradar_logfactor += frame->dof * frame->badradar_logfactor
							/ doppler->nviews;
				}
			}
			else {
				if (pos2doppler( par, &mod->photo, 0.0, 0.0, 0.0, doppler, 0, s, f, v)) {
					par->badradar = 1;
					par->badradar_logfactor += frame->dof * frame->badradar_logfactor
							/ doppler->nviews;
				}
			}

			/*  If smearing is being modeled, include the Doppler
          calculations from this view in the summed results for this frame  */

			if (doppler->nviews > 1) {
				for (j=1; j<=frame->ndop; j++)
					fit_store[j] += frame->fit[j];
				overflow_o2_store += frame->overflow_o2;
				overflow_m2_store += frame->overflow_m2;
				overflow_xsec_store += frame->overflow_xsec;
				overflow_dopmean_store += frame->overflow_dopmean;
			}

		}

		/*  If smearing is being modeled, compute mean values over all views
        for this frame and store them in the standard frame structure     */

		if (doppler->nviews > 1) {
			for (j=1; j<=frame->ndop; j++)
				frame->fit[j] = fit_store[j] / doppler->nviews;
			free_vector( fit_store, 1, frame->ndop);
			frame->overflow_o2 = overflow_o2_store / doppler->nviews;
			frame->overflow_m2 = overflow_m2_store / doppler->nviews;
			frame->overflow_xsec = overflow_xsec_store / doppler->nviews;
			frame->overflow_dopmean = overflow_dopmean_store / doppler->nviews;
		}

		/*  Perform some output tasks for the "write" action:
        Some of these tasks require POS information which will
        be overwritten as soon as we move to the next frame
        (if the pos_scope parameter is set to "global").        */

		if (par->action == WRITE) {

			/*  Display the observation epoch for this frame  */

			jd2cal( &year, &mon, &day, &hour, &min, &sec, view0->t);
			printf("# %02d_%02d JD %.5lf (UT %4d %s %2.2d %2.2d:%2.2d:%2.2d)\n",
					s, f, view0->t,
					year, monthName[mon-1], day, hour, min, sec);
			fflush(stdout);

			/*  For an NPA rotator, display the Euler angles
          of the body-fixed axes in ecliptic coordinates  */

			if (!mod->spin.pa) {
				mat2euler( view0->ae, &phi, &theta, &psi);
				if (phi < 0.0)
					phi += 2*PIE;
				if (psi < 0.0)
					psi += 2*PIE;
				printf("#       Euler angles for body-fixed axes:  %f %f %f deg\n",
						phi*R2D, theta*R2D, psi*R2D);
				fflush(stdout);
			}

			/*  For an NPA rotator, or a PA rotator with spin impulses used,
          display the sidereal spin vector in body-fixed coordinates    */

			if (!mod->spin.pa || mod->spin.n_impulse > 0) {
				cotrans( intspin_body, view0->ae, view0->intspin, 1);
				printf("#       sidereal spin vector (body-fixed): %f %f %f deg/day\n",
						intspin_body[0]*R2D, intspin_body[1]*R2D, intspin_body[2]*R2D);
				fflush(stdout);
			}

			/*  Display the linear-distance-to-Doppler proportionality factor and the
          orientation of the apparent spin vector in observer coordinates (w).   */

			cotrans( w, view0->oe, view0->spin, 1);
			spin_colat = 90.0 - atan2(w[2], sqrt(w[0]*w[0] + w[1]*w[1]))*R2D;
			spin_azim = atan2(w[1], w[0])*R2D - 90.0;
			if (spin_azim < 0.0)
				spin_azim += 360.0;
			printf("#       dopcorr %f Hz  dopconv %f Hz/km\n",
					view0->dopoff * doppler->dop_per_bin, view0->km2Hz);
			printf("#       app. spin vector points %7.3f deg from Earth at %7.3f deg E of N\n",
					spin_colat, spin_azim);
			fflush(stdout);

			/*  Display the instantaneous zero-crossing bandwidth, the
          instantaneous maximum breadth, and the instantaneous folded
          zero-crossing bandwidth; the Doppler limits were computed PRIOR
          to convolution with the (truncated) Doppler response function.   */

			bandwidth = frame->doplim[1] - frame->doplim[0];  /* Hz */
			printf("#       bandwidth %f Hz", bandwidth);
			if (spin_colat > 0.01 && spin_colat < 179.99) {
				breadth = bandwidth
						/ (doppler->dopscale.val * KM2HZFACT
								* fabs(mod->spin.omega[2].val)
				* doppler->Ftx * sin(spin_colat*D2R));         /* km */
				printf(" (breadth %f km)", breadth);
			} else {
				breadth = 0.0;
			}
			folded_bandwidth = 2 * MAX(  frame->doplim[1],
					-frame->doplim[0] );  /* Hz */
					printf(", folded bandwidth %f Hz\n", folded_bandwidth);

					/*  Display the cross section, projected area, and albedo  */

					cross_section = frame->overflow_xsec;
					for (j=1; j<=frame->ndop; j++)
						cross_section += frame->fit[j];
					cross_section *= frame->cal.val;

					n_nonblankpixels = 0;
					for (k=pos->xlim[0]; k<=pos->xlim[1]; k++)
						for (l=pos->ylim[0]; l<=pos->ylim[1]; l++)
							if (pos->cose[k][l] > 0.0)
								n_nonblankpixels++;
					projected_area = n_nonblankpixels * pos->km_per_pixel
							* pos->km_per_pixel;

					if (frame->sdev == 1.0 && frame->cal.state == 'f') {

						/* uncalibrated: dummy sdev value, floating calfact value */

						printf("#       proj area %12.6e km^2\n", projected_area);
					} else {
						printf("#       x-section %12.6e km^2  proj area %12.6e km^2",
								cross_section, projected_area);
						if (projected_area > 0.0)
							printf("  albedo %f", cross_section/projected_area);
						printf("\n");
					}
					fflush(stdout);

					/*  Display the body-fixed angular coordinates of the target-to-Earth line  */
					/*                                                                          */
					/*        oa = matrix to transform body-fixed to observer coordinates       */
					/*  to_earth = normalized target-to-Earth vector in body-fixed coords       */

					mtrnsps( oa, view0->ae);
					mmmul( oa, view0->oe, oa);
					for (j=0; j<=2; j++)
						to_earth[j] = oa[2][j];
					to_earth_lat = atan2( to_earth[2], sqrt(to_earth[0]*to_earth[0] +
							to_earth[1]*to_earth[1]   ))*R2D;
					to_earth_long = atan2( to_earth[1], to_earth[0])*R2D;
					if (to_earth_long < 0.0)
						to_earth_long += 360.0;
					rotphase = (360.0 - to_earth_long) + par->delta_rotphase*R2D;
					rotphase -= 360.0*floor(rotphase/360.0);
					printf("#       LOS at body-fixed long/lat = (%7.3f, %+7.3f) deg, rot phase %7.3f deg\n",
							to_earth_long, to_earth_lat, rotphase);
					fflush(stdout);

					/*  Compute the Doppler bin increments per POS pixel westward (ax)
          and northward (ay).  Use these to see if the sinc^2 response
          function is being evaluated at enough points per POS pixel.    */

					dopfact = doppler->dopscale.val * KM2HZFACT * pos->km_per_pixel
							* doppler->Ftx / doppler->dop_per_bin;
					ax = -w[1]*dopfact;
					ay = w[0]*dopfact;
					ax_abs = fabs(ax);
					ay_abs = fabs(ay);
					a_max = (ax_abs > ay_abs) ? ax_abs : ay_abs;
					min_nsinc2 = (int) floor(2*a_max + 0.5);     /* two points per Doppler bin */
					if (par->nsinc2 < min_nsinc2)
						printf("Each POS pixel spans ~ %.2f Doppler bins:"
								" use nsinc2 = %d (or smaller pixels)\n", a_max, min_nsinc2);

					/*  If desired, write Doppler values at the
          center of each POS pixel to a disk file  */

					if (par->listpos_deldop) {

						/*  Get the COM Doppler bin, corrected for ephemeris drift  */

						dopshift = frame->dopcom_vig + view0->dopoff;

						/*  Create the output filename (including the path),
            open the file, and write the header               */

						if (doppler->nframes > 100)
							sprintf( tempstring, "dop_%02d_%03d.posdat", s, f);
						else
							sprintf( tempstring, "dop_%02d_%02d.posdat", s, f);
						changepath( name, MAXLEN, tempstring, par->listpos_path);
						FOPEN( fpdop, name, "w");
						fprintf(fpdop, "# pos_pixels %4d  pos_width %9.6f km\n",
								par->pos_pixels, par->pos_width);

						/*  Write the Doppler (Hz) at the center of each POS pixel to the file,
            starting at the southeast corner of the POS image and moving east to west
            and then south to north.  Note that functions posclr and posvis flag
            blank-sky pixels by assigning "cose" = cos(scattering angle) = 0, in
            which case absurdly large negative Doppler values are written to disk.     */

						for (y=-pos->n; y<=pos->n; y++)
							for (x=-pos->n; x<=pos->n; x++) {
								if (pos->cose[x][y] > 0.0)
									dopPOS_Hz = (ax*x + ay*y + dopshift) * doppler->dop_per_bin;
								else
									dopPOS_Hz = -HUGENUMBER;  /* blank sky */
								fprintf(fpdop, "%13.6e\n", dopPOS_Hz);
							}

						/*  Close the file  */

						fclose( fpdop);
					}

					/*  Create a plane-of-sky image that assumes front illumination and
          Lambert scattering: pixel brightness level proportional to
          cos(scattering angle).  This must be done now if the "pos_scope"
          parameter is set to "global" because the POS frame will be
          overwritten as soon as we move to the next observation epoch.     */

					if (!par->mark_unseen)
						write_pos_doppler( par, mod, doppler, s, f);

		}  /* end code for "write" action */

	}  /* end loop over frames */
}

void calc_poset( struct par_t *par, struct mod_t *mod, struct poset_t *poset,
		int s)
{
	const char *monthName[12] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
			"Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
	double orbit_offset[3] = {0.0, 0.0, 0.0};

	FILE *fpopt;
	char tempstring[MAXLEN], name[MAXLEN];
	int year, mon, day, hour, min, sec, f, c, i, j, k, l, nrow_fit, ncol_fit, n_pos,
	facetnum, x, y, v, v2;
	double w[3], spin_colat, spin_azim, xoff, yoff, resamp_fact, resamp_x0, resamp_y0,
	xcom_fit, ycom_fit, resamp_xwidth, resamp_ywidth, resamp_angle, oa[3][3],
	to_earth[3], to_earth_lat, to_earth_long, rotphase, sa[3][3], to_sun[3],
	to_sun_lat, to_sun_long, pab[3], pab_lat, pab_long, intensityfactor,
	phi, theta, psi, intspin_body[3], badposet_logfactor_view;
	double **fit_store;
	struct posetfrm_t *frame;
	struct posetview_t *view0;
	struct pos_t *pos;

	for (f=0; f<poset->nframes; f++) {

		frame = &poset->frame[f];
		view0 = &frame->view[poset->v0];
		pos = &frame->pos;

		ncol_fit = frame->ncol;
		nrow_fit = frame->nrow;

		/*  If smearing is being modeled, initialize variables that
        will be used to sum results calculated for individual views  */

		if (poset->nviews > 1) {
			fit_store = matrix( 1, ncol_fit, 1, nrow_fit);
			for (i=1; i<=ncol_fit; i++)
				for (j=1; j<=nrow_fit; j++)
					fit_store[i][j] = 0.0;
		}

		/*  Loop over all views for this (smeared) frame, going in an order that
        ends with the view corresponding to the epoch listed for this frame
        in the obs file; this way we can use the calculated information for
        that view in the "write" action screen and disk output that follows   */

		for (v2=poset->v0+1; v2<=poset->v0+poset->nviews; v2++) {
			v = v2 % poset->nviews;

			for (i=0; i<=2; i++)
				for (j=0; j<=2; j++) {
					pos->ae[i][j] = frame->view[v].ae[i][j];
					pos->oe[i][j] = frame->view[v].oe[i][j];
					pos->se[i][j] = frame->view[v].se[i][j];
				}
			pos->bistatic = 1;

			/*  Initialize the plane-of-sky view  */

			posclr( pos);

			/*  Call routine posvis to get the facet number, scattering angle,
          incidence angle, and distance toward Earth at the center of
          each POS pixel; set the posbnd parameter to 1 if any portion
          of the model extends beyond the POS frame limits.              */

			for (c=0; c<mod->shape.ncomp; c++)
				if (posvis( &mod->shape.comp[c].real, orbit_offset, pos,
						(int) par->pos_smooth, 0, 0, c) && v == poset->v0) {
					par->posbnd = 1;
					if (pos->bistatic)
						par->posbnd_logfactor += 0.5 * frame->dof * pos->posbnd_logfactor;
					else
						par->posbnd_logfactor += frame->dof * pos->posbnd_logfactor;
				}

			/*  Now view the model from the source (sun) and get the facet number
          and distance toward the source of each pixel in this projected view;
          use this information to determine which POS pixels are shadowed       */

			if (pos->bistatic) {
				for (c=0; c<mod->shape.ncomp; c++)
					if (posvis( &mod->shape.comp[c].real, orbit_offset, pos,
							0, 1, 0, c)) {
						par->posbnd = 1;
						par->posbnd_logfactor += 0.5 * frame->dof * pos->posbnd_logfactor;
					}

				/*  Identify and mask out shadowed POS pixels  */

				posmask( pos, par->mask_tol);
			}

			/*  Go through all POS pixels which are visible and unshadowed with
          sufficiently low scattering and incidence angles, and mark the facets
          which project onto their centers as having been "seen" at least once   */

			if (s != par->exclude_seen && v == poset->v0) {
				for (k=pos->xlim[0]; k<=pos->xlim[1]; k++)
					for (l=pos->ylim[0]; l<=pos->ylim[1]; l++) {
						if ((pos->cose[k][l] > par->mincosine_seen)
								&& (pos->cosi[k][l] > par->mincosine_seen)
								&& (pos->f[k][l] >= 0)) {
							facetnum = pos->f[k][l];
							c = pos->comp[k][l];
							mod->shape.comp[c].real.f[facetnum].seen = 1;
						}
					}
			}

			/*  Compute the sky rendering  */

			intensityfactor = pow( pos->km_per_pixel/AU, 2.0);
			apply_photo( mod, poset->ioptlaw, frame->view[v].solar_phase,
					intensityfactor, pos, 0, s, i);

			/*  Resample the sky rendering to get the model plane-of-sky image    */
			/*  (if using bicubic interpolation or cubic convolution, force       */
			/*  all model pixel values to be nonnegative)                         */
			/*                                                                    */
			/*  Implement the x and y COM offsets, xoff and yoff, by first        */
			/*  using them to compute xcom_fit and ycom_fit -- the COM position   */
			/*  in the fit image, relative to the center of the fit image -- and  */
			/*  then shifting the resampled region in the *opposite* direction    */
			/*  by the appropriate proportional amount.  Then implement the       */
			/*  "northangle" setting (clockwise heading of north) by rotating     */
			/*  the resampling grid *counterclockwise* by northangle.             */

			n_pos = pos->n;
			xoff = frame->off[0].val;
			yoff = frame->off[1].val;
			xcom_fit = (frame->colcom_vig - (ncol_fit + 1)/2.0) + xoff;
			ycom_fit = (frame->rowcom_vig - (nrow_fit + 1)/2.0) + yoff;
			resamp_fact = frame->fit.km_per_pixel / pos->km_per_pixel;
			resamp_x0 = -xcom_fit*resamp_fact;
			resamp_y0 = -ycom_fit*resamp_fact;
			resamp_xwidth = resamp_fact*(ncol_fit - 1);
			resamp_ywidth = resamp_fact*(nrow_fit - 1);
			resamp_angle = -frame->northangle;
			resampim( frame->pos.b, -n_pos, n_pos, -n_pos, n_pos,
					frame->fit.b, 1, ncol_fit, 1, nrow_fit,
					resamp_x0, resamp_xwidth, resamp_y0, resamp_ywidth, resamp_angle,
					(int) par->poset_resample, (int) par->image_rebin);
			if (par->poset_resample == BICUBIC || par->poset_resample == CUBICCONV) {
				for (k=1; k<=ncol_fit; k++)
					for (l=1; l<=nrow_fit; l++)
						frame->fit.b[k][l] = MAX( 0.0, frame->fit.b[k][l]);
			}

			/*  Set the badposet flag and increase badposet_logfactor if the model   */
			/*  plane-of-sky image is too small to "contain" all of the sky          */
			/*  rendering's nonzero pixels.                                          */

			if (checkposet( pos->b, -n_pos, n_pos, -n_pos, n_pos,
					resamp_x0, resamp_xwidth, resamp_y0, resamp_ywidth, resamp_angle,
					&badposet_logfactor_view)) {
				par->badposet = 1;
				par->badposet_logfactor += frame->dof * badposet_logfactor_view
						/ poset->nviews;
			}

			/*  If smearing is being modeled, include the plane-of-sky
          calculations from this view in the summed results for this frame  */

			if (poset->nviews > 1)
				for (i=1; i<=ncol_fit; i++)
					for (j=1; j<=nrow_fit; j++)
						fit_store[i][j] += frame->fit.b[i][j];

		}

		/*  If smearing is being modeled, compute mean values over all views
        for this frame and store them in the standard frame structure     */

		if (poset->nviews > 1) {
			for (i=1; i<=ncol_fit; i++)
				for (j=1; j<=nrow_fit; j++)
					frame->fit.b[i][j] = fit_store[i][j] / poset->nviews;
			free_matrix( fit_store, 1, ncol_fit, 1, nrow_fit);
		}

		/*  Perform some output tasks for the "write" action:
        Some of these tasks require POS information which will
        be overwritten as soon as we move to the next frame
        (if the pos_scope parameter is set to "global").        */

		if (par->action == WRITE) {

			/*  Display the observation epoch for this frame  */

			jd2cal( &year, &mon, &day, &hour, &min, &sec, view0->t);
			printf("# %02d_%02d JD %.5lf (UT %4d %s %2.2d %2.2d:%2.2d:%2.2d)\n",
					s, f, view0->t,
					year, monthName[mon-1], day, hour, min, sec);
			fflush(stdout);

			/*  For an NPA rotator, display the Euler angles
          of the body-fixed axes in ecliptic coordinates  */

			if (!mod->spin.pa) {
				mat2euler( view0->ae, &phi, &theta, &psi);
				if (phi < 0.0)
					phi += 2*PIE;
				if (psi < 0.0)
					psi += 2*PIE;
				printf("#       Euler angles for body-fixed axes:  %f %f %f deg\n",
						phi*R2D, theta*R2D, psi*R2D);
				fflush(stdout);
			}

			/*  For an NPA rotator, or a PA rotator with spin impulses used,
          display the sidereal spin vector in body-fixed coordinates    */

			if (!mod->spin.pa || mod->spin.n_impulse > 0) {
				cotrans( intspin_body, view0->ae, view0->intspin, 1);
				printf("#       sidereal spin vector (body-fixed): %f %f %f deg/day\n",
						intspin_body[0]*R2D, intspin_body[1]*R2D, intspin_body[2]*R2D);
				fflush(stdout);
			}

			/*  Display the fit frame's linear dimensions, the linear dimensions of the
          rectangular subset that contains the target, and the linear COM offsets  */

			frame->fit.xlim[0] = ncol_fit;
			frame->fit.xlim[1] = 1;
			frame->fit.ylim[0] = nrow_fit;
			frame->fit.ylim[1] = 1;
			for (k=1; k<=ncol_fit; k++)
				for (l=1; l<=nrow_fit; l++)
					if (frame->fit.b[k][l] > 0.0) {
						frame->fit.xlim[0] = MIN( frame->fit.xlim[0], k);
						frame->fit.xlim[1] = MAX( frame->fit.xlim[1], k);
						frame->fit.ylim[0] = MIN( frame->fit.ylim[0], l);
						frame->fit.ylim[1] = MAX( frame->fit.ylim[1], l);
					}
			printf("#       frame %.3f x %.3f km, target %.3f x %.3f km, offset %.3f x %.3f km\n",
					frame->fit.km_per_pixel * ncol_fit,
					frame->fit.km_per_pixel * nrow_fit,
					frame->fit.km_per_pixel * (frame->fit.xlim[1] - frame->fit.xlim[0] + 1),
					frame->fit.km_per_pixel * (frame->fit.ylim[1] - frame->fit.ylim[0] + 1),
					frame->fit.km_per_pixel * xoff,
					frame->fit.km_per_pixel * yoff);
			fflush(stdout);

			/*  Display the epoch, solar phase angle, solar azimuth angle, and
          orientation of the apparent spin vector in observer coordinates (w)  */

			cotrans( w, view0->oe, view0->spin, 1);
			spin_colat = 90.0 - atan2(w[2], sqrt(w[0]*w[0] + w[1]*w[1]))*R2D;
			spin_azim = atan2(w[1], w[0])*R2D - 90.0;
			if (spin_azim < 0.0)
				spin_azim += 360.0;
			printf("#       solar phase angle is    %7.3f deg, sun is at    %7.3f deg E of N\n",
					R2D*view0->solar_phase, R2D*view0->solar_azimuth);
			printf("#       app. spin vector points %7.3f deg from Earth at %7.3f deg E of N\n",
					spin_colat, spin_azim);
			fflush(stdout);

			/*  Compute the body-fixed angular coordinates of the target-to-Earth line  */
			/*                                                                          */
			/*        oa = matrix to transform body-fixed to observer coordinates       */
			/*  to_earth = normalized target-to-Earth vector in body-fixed coords       */

			mtrnsps( oa, view0->ae);
			mmmul( oa, view0->oe, oa);
			for (j=0; j<=2; j++)
				to_earth[j] = oa[2][j];
			to_earth_lat = atan2( to_earth[2], sqrt(to_earth[0]*to_earth[0] +
					to_earth[1]*to_earth[1]   ))*R2D;
			to_earth_long = atan2( to_earth[1], to_earth[0])*R2D;
			if (to_earth_long < 0.0)
				to_earth_long += 360.0;
			rotphase = (360.0 - to_earth_long) + par->delta_rotphase*R2D;
			rotphase -= 360.0*floor(rotphase/360.0);

			/*  Display the body-fixed angular coordinates of the target-to-Sun line  */
			/*                                                                        */
			/*      sa = matrix to transform body-fixed to Sun coordinates            */
			/*  to_sun = normalized target-to-Sun vector in body-fixed coords         */

			mtrnsps( sa, view0->ae);
			mmmul( sa, view0->se, sa);
			for (j=0; j<=2; j++)
				to_sun[j] = sa[2][j];
			to_sun_lat = atan2( to_sun[2], sqrt(to_sun[0]*to_sun[0] +
					to_sun[1]*to_sun[1]   ))*R2D;
			to_sun_long = atan2( to_sun[1], to_sun[0])*R2D;
			if (to_sun_long < 0.0)
				to_sun_long += 360.0;
			printf("#       Sun-to-asteroid line at body-fixed long/lat = (%7.3f, %+7.3f) deg\n",
					to_sun_long, to_sun_lat);
			fflush(stdout);

			/*  Display the body-fixed angular coordinates of the phase angle bisector
          (the line from the asteroid to the midpoint between Sun and Earth)      */

			for (j=0; j<=2; j++)
				pab[j] = to_sun[j] + to_earth[j];
			normalize( pab);
			pab_lat = atan2( pab[2], sqrt(pab[0]*pab[0] + pab[1]*pab[1]))*R2D;
			pab_long = atan2( pab[1], pab[0])*R2D;
			if (pab_long < 0.0)
				pab_long += 360.0;
			printf("#       phase angle bisector at body-fixed long/lat = (%7.3f, %+7.3f) deg\n",
					pab_long, pab_lat);
			fflush(stdout);

			/*  Display the body-fixed angular coordinates of the target-to-Earth line
          (which we computed earlier)                                             */

			printf("#       LOS at body-fixed long/lat = (%7.3f, %+7.3f) deg, rot phase %7.3f deg\n",
					to_earth_long, to_earth_lat, rotphase);
			fflush(stdout);

			/*  If desired, write the optical brightness of each POS pixel to a disk file  */

			if (par->listpos_opt) {

				/*  Create the output filename (including the path),
            open the file, and write the header               */

				if (poset->nframes > 100)
					sprintf( tempstring, "opt_%02d_%03d.posdat", s, f);
				else
					sprintf( tempstring, "opt_%02d_%02d.posdat", s, f);
				changepath( name, MAXLEN, tempstring, par->listpos_path);
				FOPEN( fpopt, name, "w");
				fprintf(fpopt, "# pos_pixels %4d  pos_width %9.6f km\n",
						par->pos_pixels, par->pos_width);

				/*  Write the optical brightness values to disk, starting at the southeast
            corner of the POS image and moving east to west and then south to north  */

				for (y=-pos->n; y<=pos->n; y++)
					for (x=-pos->n; x<=pos->n; x++)
						fprintf(fpopt, "%13.6e\n", pos->b[x][y]);

				/*  Close the file  */

				fclose( fpopt);
			}

			/*  Output the sky rendering as a pgm image: This must be done now
          because it will be overwritten as soon as we move to the next
          observation epoch (if the pos_scope parameter is set to "global").  */

			if (!par->mark_unseen)
				write_pos_poset( par, mod, poset, s, f);

		}  /* end code for "write" action */

	}  /* end loop over frames */
}

void calc_lghtcrv( struct par_t *par, struct mod_t *mod, struct lghtcrv_t *lghtcrv,
		int s)
{
	const char *monthName[12] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
			"Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
	double orbit_offset[3] = {0.0, 0.0, 0.0};

	FILE *fpopt;
	char tempstring[MAXLEN], name[MAXLEN];
	int year, mon, day, hour, min, sec, n, ncalc, c, i, i_mid, j, k, l, facetnum,
	n_cross360, n_projectedpixels, n_shadowedpixels, x, y, v;
	double epoch_mid, epoch_diff_min, epoch_diff, w[3], spin_colat, spin_azim, oa[3][3],
	rotphase, sa[3][3], to_sun[3], to_sun_lat, to_sun_long, pab[3], pab_lat,
	pab_long, intensityfactor, phi, theta, psi, intspin_body[3], posbnd_logfactor,
	projected_area, lambertdisk_intensity, interp;
	double **to_earth, *to_earth_lat, *to_earth_long, *rotphase_unwrapped;
	struct crvrend_t *rend;
	struct pos_t *pos;

	/*  Initialize variables to avoid compilation warning  */

	i_mid = 0;
	epoch_mid = epoch_diff = epoch_diff_min = 0.0;
	n_cross360 = 0;
	to_earth = NULL;
	to_earth_lat = to_earth_long = rotphase_unwrapped = NULL;

	/*  Initialize variables dealing with bad models  */

	posbnd_logfactor = 0.0;

	/*  Get n, the number of observed points for this lightcurve,
      and ncalc, the number of epochs at which model lightcurve
      brightnesses are to be computed                            */

	n = lghtcrv->n;
	ncalc = lghtcrv->ncalc;

	/*  For the write action, compute various quantities that will
      later be displayed to the screen for model lightcurve point(s)  */

	if (par->action == WRITE) {

		/*  Decide which model lightcurve point(s) should have screen display:
        if the "lcrv_writeall" parameter is turned on, process all points;
        otherwise process just the one point which falls closest in time
        to the midpoint of lightcurve observations                          */

		if (!par->lcrv_writeall) {
			epoch_mid = (lghtcrv->t[1][lghtcrv->v0] + lghtcrv->t[n][lghtcrv->v0]) / 2;
			epoch_diff_min = 1.0e20;
			for (i=1; i<=ncalc; i++) {
				epoch_diff = fabs(lghtcrv->x[i] - epoch_mid);
				if (epoch_diff < epoch_diff_min) {
					i_mid = i;
					epoch_diff_min = epoch_diff;
				}
			}
		}

		/*  Allocate memory for the body-fixed Cartesian and angular
        coordinates of the target-to-Earth line and for rotation phase
        that hasn't been wrapped around 360/0 deg                       */

		to_earth = matrix( 1, ncalc, 0, 2);
		to_earth_lat = vector( 1, ncalc);
		to_earth_long = vector( 1, ncalc);
		rotphase_unwrapped = vector( 1, ncalc);

		/*  Do the computations  */

		for (i=1; i<=ncalc; i++) {

			rend = &lghtcrv->rend[i];

			/*  Compute the body-fixed angular coordinates of the target-to-Earth line  */
			/*                                                                          */
			/*           oa = matrix to transform body-fixed to observer coordinates    */
			/*  to_earth[i] = normalized target-to-Earth vector in body-fixed coords    */

			mtrnsps( oa, rend->ae);
			mmmul( oa, rend->oe, oa);
			for (j=0; j<=2; j++)
				to_earth[i][j] = oa[2][j];
			to_earth_lat[i] = atan2( to_earth[i][2], sqrt(to_earth[i][0]*to_earth[i][0] +
					to_earth[i][1]*to_earth[i][1]   ))*R2D;
			to_earth_long[i] = atan2( to_earth[i][1], to_earth[i][0])*R2D;
			if (to_earth_long[i] < 0.0)
				to_earth_long[i] += 360.0;

			/*  Calculate rotation phase  */

			rotphase = (360.0 - to_earth_long[i]) + par->delta_rotphase*R2D;
			rotphase -= 360.0*floor(rotphase/360.0);
			lghtcrv->rotphase_calc[i] = rotphase;
			if (i > 1 && rotphase - lghtcrv->rotphase_calc[i-1] < -180)
				n_cross360++;
			rotphase_unwrapped[i] = rotphase + n_cross360*360.0;
			fflush(stdout);
		}

		/*  Now that we have calculated the rotation phase at each of the epochs x, use
        cubic spline interpolation (Numerical Recipes routines spline and splint)
        to get the rotation phase at each OBSERVATION epoch t[i], with i=1,2,...,n.
        We work with the calculated phases that have not been wrapped around 360/0,
        then wrap the results around 360/0.  Note that vector y2 contains the second
        derivatives of the interpolating function at the calculation epochs x.        */

		spline( lghtcrv->x, rotphase_unwrapped, ncalc, 2.0e30, 2.0e30, lghtcrv->y2);
		for (i=1; i<=n; i++) {
			splint( lghtcrv->x, rotphase_unwrapped, lghtcrv->y2, ncalc,
					lghtcrv->t[i][lghtcrv->v0], &lghtcrv->rotphase_obs[i]);
			lghtcrv->rotphase_obs[i] -= 360.0*floor(lghtcrv->rotphase_obs[i]/360.0);
		}
	}

	/*  Calculate the model lightcurve values at each of the user-specified
      epochs x[i], with i=1,2,...,ncalc; these may or may not be the same as the
      epochs t[i] (i=1,2,...,n) at which actual lightcurve observations were made.  */

	for (i=1; i<=ncalc; i++) {

		rend = &lghtcrv->rend[i];
		pos = &rend->pos;

		for (j=0; j<=2; j++)
			for (k=0; k<=2; k++) {
				pos->ae[j][k] = rend->ae[j][k];
				pos->oe[j][k] = rend->oe[j][k];
				pos->se[j][k] = rend->se[j][k];
			}
		pos->bistatic = 1;

		/*  Initialize the plane-of-sky view  */

		posclr( pos);

		/*  Call routine posvis to get the facet number, scattering angle,
        incidence angle, and distance toward Earth at the center of
        each POS pixel; set the posbnd parameter to 1 if any portion
        of the model extends beyond the POS frame limits.              */

		for (c=0; c<mod->shape.ncomp; c++)
			if (posvis( &mod->shape.comp[c].real, orbit_offset, pos,
					(int) par->pos_smooth, 0, 0, c)) {
				par->posbnd = 1;
				if (pos->bistatic)
					posbnd_logfactor += 0.5 * pos->posbnd_logfactor;
				else
					posbnd_logfactor += pos->posbnd_logfactor;
			}

		/*  Now view the model from the source (sun) and get the facet number
        and distance toward the source of each pixel in this projected view;
        use this information to determine which POS pixels are shadowed       */

		if (pos->bistatic) {
			for (c=0; c<mod->shape.ncomp; c++)
				if (posvis( &mod->shape.comp[c].real, orbit_offset, pos,
						0, 1, 0, c)) {
					par->posbnd = 1;
					posbnd_logfactor += 0.5 * pos->posbnd_logfactor;
				}

			/*  Identify and mask out shadowed POS pixels  */
			posmask( pos, par->mask_tol);
		}
		/*  Go through all POS pixels which are visible and unshadowed with
        sufficiently low scattering and incidence angles, and mark the facets
        which project onto their centers as having been "seen" at least once   */

		if (s != par->exclude_seen) {
			for (k=pos->xlim[0]; k<=pos->xlim[1]; k++)
				for (l=pos->ylim[0]; l<=pos->ylim[1]; l++) {
					if ((pos->cose[k][l] > par->mincosine_seen)
							&& (pos->cosi[k][l] > par->mincosine_seen)
							&& (pos->f[k][l] >= 0)) {
						facetnum = pos->f[k][l];
						c = pos->comp[k][l];
						mod->shape.comp[c].real.f[facetnum].seen = 1;
					}
				}
		}


		/*  Compute the model brightness for this model lightcurve point  */
		intensityfactor = pow( pos->km_per_pixel/AU, 2.0);
		lghtcrv->y[i] = apply_photo( mod, lghtcrv->ioptlaw, lghtcrv->solar_phase[i],
				intensityfactor, pos, 0, s, i);


		/*  Carry out screen and disk output for the write action  */

		if (par->action == WRITE) {

			/*  Do the screen output  */

			if (par->lcrv_writeall || i == i_mid) {

				/*  Display the observation epoch for this lightcurve point  */

				jd2cal( &year, &mon, &day, &hour, &min, &sec, lghtcrv->x[i]);
				printf("# %02d_%02d JD %.5lf (UT %4d %s %2.2d %2.2d:%2.2d:%2.2d)\n",
						s, i-1, lghtcrv->x[i],
						year, monthName[mon-1], day, hour, min, sec);
				fflush(stdout);

				/*  For an NPA rotator, display the Euler angles
            of the body-fixed axes in ecliptic coordinates  */

				if (!mod->spin.pa) {
					mat2euler( rend->ae, &phi, &theta, &psi);
					if (phi < 0.0)
						phi += 2*PIE;
					if (psi < 0.0)
						psi += 2*PIE;
					printf("#       Euler angles for body-fixed axes:  %f %f %f deg\n",
							phi*R2D, theta*R2D, psi*R2D);
					fflush(stdout);
				}

				/*  For an NPA rotator, or a PA rotator with spin impulses used,
            display the sidereal spin vector in body-fixed coordinates    */

				if (!mod->spin.pa || mod->spin.n_impulse > 0) {
					cotrans( intspin_body, rend->ae, rend->intspin, 1);
					printf("#       sidereal spin vector (body-fixed): %f %f %f deg/day\n",
							intspin_body[0]*R2D, intspin_body[1]*R2D, intspin_body[2]*R2D);
					fflush(stdout);
				}

				/*  Display the solar phase angle and azimuth and the
            orientation of the apparent spin vector in observer coordinates (w).   */

				cotrans( w, rend->oe, rend->spin, 1);
				spin_colat = 90.0 - atan2(w[2], sqrt(w[0]*w[0] + w[1]*w[1]))*R2D;
				spin_azim = atan2(w[1], w[0])*R2D - 90.0;
				if (spin_azim < 0.0)
					spin_azim += 360.0;
				printf("#       solar phase angle is    %7.3f deg, sun is at    %7.3f deg E of N\n",
						R2D*lghtcrv->solar_phase[i], R2D*lghtcrv->solar_azimuth[i]);
				printf("#       app. spin vector points %7.3f deg from Earth at %7.3f deg E of N\n",
						spin_colat, spin_azim);
				fflush(stdout);

				/*  Display projected area; if it's absolute photometry, also display
            geometric albedo (which is defined for zero degrees solar phase
            angle, but we apply no explicit phase correction here)             */

				n_projectedpixels = n_shadowedpixels = 0;
				for (k=pos->xlim[0]; k<=pos->xlim[1]; k++)
					for (l=pos->ylim[0]; l<=pos->ylim[1]; l++)
						if (pos->cosi[k][l] > 0.0 && pos->f[k][l] >= 0) {
							n_projectedpixels++;
							if (pos->cose[k][l] <= 0.0)
								n_shadowedpixels++;
						}
				projected_area = n_projectedpixels * pos->km_per_pixel
						* pos->km_per_pixel;
				lambertdisk_intensity = n_projectedpixels*intensityfactor/PIE;
				printf("#       proj area %12.6e km^2 (%4.1f%% shadowed)",
						projected_area, (100.0*n_shadowedpixels)/n_projectedpixels);
				if(lghtcrv->cal.state == 'c' && n_projectedpixels > 0)
					printf("  geom albedo %f", lghtcrv->y[i]/lambertdisk_intensity);
				printf("\n");
				fflush(stdout);

				/*  Display the body-fixed angular coordinates of the target-to-Sun line  */
				/*                                                                        */
				/*      sa = matrix to transform body-fixed to Sun coordinates            */
				/*  to_sun = normalized target-to-Sun vector in body-fixed coords         */

				mtrnsps( sa, rend->ae);
				mmmul( sa, rend->se, sa);
				for (j=0; j<=2; j++)
					to_sun[j] = sa[2][j];
				to_sun_lat = atan2( to_sun[2], sqrt(to_sun[0]*to_sun[0] +
						to_sun[1]*to_sun[1]   ))*R2D;
				to_sun_long = atan2( to_sun[1], to_sun[0])*R2D;
				if (to_sun_long < 0.0)
					to_sun_long += 360.0;
				printf("#       Sun-to-asteroid line at body-fixed long/lat = (%7.3f, %+7.3f) deg\n",
						to_sun_long, to_sun_lat);
				fflush(stdout);

				/*  Display the body-fixed angular coordinates of the phase angle bisector
            (the line from the asteroid to the midpoint between Sun and Earth)      */

				for (j=0; j<=2; j++)
					pab[j] = to_sun[j] + to_earth[i][j];
				normalize( pab);
				pab_lat = atan2( pab[2], sqrt(pab[0]*pab[0] + pab[1]*pab[1]))*R2D;
				pab_long = atan2( pab[1], pab[0])*R2D;
				if (pab_long < 0.0)
					pab_long += 360.0;
				printf("#       phase angle bisector at body-fixed long/lat = (%7.3f, %+7.3f) deg\n",
						pab_long, pab_lat);
				fflush(stdout);

				/*  Display the body-fixed angular coordinates of the target-to-Earth line  */

				printf("#       LOS at body-fixed long/lat = (%7.3f, %+7.3f) deg",
						to_earth_long[i], to_earth_lat[i]);
				fflush(stdout);
			}

			/*  Finish the screen output: calculate and display rotation phase
          (must be done in two pieces if the lcrv_writeall parameter is turned off)  */

			if (par->lcrv_writeall) {
				printf(", rot phase %7.3f deg\n", lghtcrv->rotphase_calc[i]);
			} else if (i == i_mid) {
				printf("\n");
				printf("#       rot phase %7.3f deg", lghtcrv->rotphase_calc[i]);
			} else if (i == ncalc) {
				printf("  (range %7.3f - %7.3f deg",
						lghtcrv->rotphase_calc[1], lghtcrv->rotphase_calc[ncalc]);
				if (n_cross360 == 1)
					printf(", crossing 360/0 once");
				else if (n_cross360 > 1)
					printf(", crossing 360/0 %d times", n_cross360);
				printf(")\n");
			}
			fflush(stdout);

			/*  If desired, write the optical brightness of each POS pixel to a disk file  */

			if (par->listpos_opt) {

				/*  Create the output filename (including the path),
            open the file, and write the header               */

				if (ncalc > 100)
					sprintf( tempstring, "opt_%02d_%03d.posdat", s, i-1);
				else
					sprintf( tempstring, "opt_%02d_%02d.posdat", s, i-1);
				changepath( name, MAXLEN, tempstring, par->listpos_path);
				FOPEN( fpopt, name, "w");
				fprintf(fpopt, "# pos_pixels %4d  pos_width %9.6f km\n",
						par->pos_pixels, par->pos_width);

				/*  Write the optical brightness values to disk, starting at the southeast
            corner of the POS image and moving east to west and then south to north  */

				for (y=-pos->n; y<=pos->n; y++)
					for (x=-pos->n; x<=pos->n; x++)
						fprintf(fpopt, "%13.6e\n", pos->b[x][y]);

				/*  Close the file  */

				fclose( fpopt);
			}

			/*  If specified, output the model plane-of-sky image for each epoch
          at which a model lightcurve brightness has been calculated: This
          requires POS information that will be overwritten as soon as we
          move to the next epoch (if the pos_scope parameter is set to "global").  */

			if (par->lcrv_pos && !par->mark_unseen)
				write_pos_lghtcrv( par, mod, lghtcrv, s, i);

			/*  End of screen and disk output for write action  */

		}
		/*  Finished with this calculated lightcurve point  */
	}

	/*  Now that we have calculated the model lightcurve brightnesses y at each
      of the epochs x, we use cubic spline interpolation (Numerical Recipes
      routines spline and splint) to get model lightcurve brightness fit[i]
      at each OBSERVATION epoch t[i], with i=1,2,...,n.  This will allow us
      (in routine chi2) to compare model to data (fit[i] to obs[i]) to get
      chi-square.  Note that vector y2 contains the second derivatives of
      the interpolating function at the calculation epochs x.

      Smearing is handled by interpolating the brightness at the time t of
      each individual view and then taking the mean of all views that
      correspond to a given observed lightcurve point.                         */

	spline( lghtcrv->x, lghtcrv->y, ncalc, 2.0e30, 2.0e30, lghtcrv->y2);

	for (i=1; i<=n; i++) {
		lghtcrv->fit[i] = 0.0;
		for (v=0; v<lghtcrv->nviews; v++) {
			splint( lghtcrv->x, lghtcrv->y, lghtcrv->y2, ncalc,
					lghtcrv->t[i][v], &interp);
			lghtcrv->fit[i] += interp;
		}
		lghtcrv->fit[i] /= lghtcrv->nviews;
	}

	/*  Deal with flags for model that extends beyond the POS frame  */

	par->posbnd_logfactor += lghtcrv->dof * (posbnd_logfactor/ncalc);

	/*  Deallocate memory  */

	if (par->action == WRITE) {
		free_matrix( to_earth, 1, ncalc, 0, 2);
		free_vector( to_earth_lat, 1, ncalc);
		free_vector( to_earth_long, 1, ncalc);
		free_vector( rotphase_unwrapped, 1, ncalc);
	}


}

void write_pos_deldop( struct par_t *par, struct mod_t *mod,
		struct deldop_t *deldop, int s, int f)
{
	char name[MAXLEN];
	int color_output;

	if (par->plot_pa[0] || par->plot_pa[1] || par->plot_pa[2]
	                                                       || par->plot_angmom || par->mark_unseen
	                                                       || par->write_highlight) {
		color_output = 1;
		if (deldop->nframes > 100)
			sprintf( name, "sky_%02d_%03d.ppm", s, f);
		else
			sprintf( name, "sky_%02d_%02d.ppm", s, f);
	} else {
		color_output = 0;
		if (deldop->nframes > 100)
			sprintf( name, "sky_%02d_%03d.pgm", s, f);
		else
			sprintf( name, "sky_%02d_%02d.pgm", s, f);
	}
	write_pos( par, mod, &deldop->frame[f].pos,
			deldop->frame[f].view[deldop->v0].intspin, deldop->iradlaw,
			color_output, name);
}

void write_pos_doppler( struct par_t *par, struct mod_t *mod,
		struct doppler_t *doppler, int s, int f)
{
	char name[MAXLEN];
	int color_output;

	if (par->plot_pa[0] || par->plot_pa[1] || par->plot_pa[2]
	                                                       || par->plot_angmom || par->mark_unseen
	                                                       || par->write_highlight) {
		color_output = 1;
		if (doppler->nframes > 100)
			sprintf( name, "sky_%02d_%03d.ppm", s, f);
		else
			sprintf( name, "sky_%02d_%02d.ppm", s, f);
	} else {
		color_output = 0;
		if (doppler->nframes > 100)
			sprintf( name, "sky_%02d_%03d.pgm", s, f);
		else
			sprintf( name, "sky_%02d_%02d.pgm", s, f);
	}
	write_pos( par, mod, &doppler->frame[f].pos,
			doppler->frame[f].view[doppler->v0].intspin, doppler->iradlaw,
			color_output, name);
}

void write_pos_poset( struct par_t *par, struct mod_t *mod,
		struct poset_t *poset, int s, int f)
{
	char name[MAXLEN];
	int color_output;

	if (par->plot_pa[0] || par->plot_pa[1] || par->plot_pa[2]
	                                                       || par->plot_angmom || par->mark_unseen
	                                                       || par->write_highlight) {
		color_output = 1;
		if (poset->nframes > 100)
			sprintf( name, "sky_%02d_%03d.ppm", s, f);
		else
			sprintf( name, "sky_%02d_%02d.ppm", s, f);
	} else {
		color_output = 0;
		if (poset->nframes > 100)
			sprintf( name, "sky_%02d_%03d.pgm", s, f);
		else
			sprintf( name, "sky_%02d_%02d.pgm", s, f);
	}
	write_pos( par, mod, &poset->frame[f].pos,
			poset->frame[f].view[poset->v0].intspin, -1,
			color_output, name);
}

void write_pos_lghtcrv( struct par_t *par, struct mod_t *mod,
		struct lghtcrv_t *lghtcrv, int s, int i)
{
	char name[MAXLEN];
	int color_output;

	if (par->plot_pa[0] || par->plot_pa[1] || par->plot_pa[2]
	                                                       || par->plot_angmom || par->mark_unseen
	                                                       || par->write_highlight) {
		color_output = 1;
		if (lghtcrv->ncalc > 100)
			sprintf( name, "sky_%02d_%03d.ppm", s, i-1);
		else
			sprintf( name, "sky_%02d_%02d.ppm", s, i-1);
	} else {
		color_output = 0;
		if (lghtcrv->ncalc > 100)
			sprintf( name, "sky_%02d_%03d.pgm", s, i-1);
		else
			sprintf( name, "sky_%02d_%02d.pgm", s, i-1);
	}
	write_pos( par, mod, &lghtcrv->rend[i].pos,
			lghtcrv->rend[i].intspin, -1, color_output, name);
}
