/*****************************************************************************************
                                                                                   chi2.c

Compute chi-square for a model that was previously created.

Modified 2015 June 10 by CM:
    Implement smearing for the "fit" and "write" actions

Modified 2013 April 24 by CM:
    Adjust names of output files so they are in alphanumeric order if > 100 per dataset

Modified 2012 December 6 by CM:
    Fix bug introduced on December 6: take "calval" into account when determining whether
        or not model signal is greater than 'chi2fit0_thresh' sigmas

Modified 2012 December 5 by CM:
    When the "write_chi2fit0" parameter is turned on, display the number of degrees of
        freedom with zero model signal in addition to the chi2 contribution, and list
        chi2 and dof for all data combined
    Implement the 'chi2fit0_thresh' parameter

Modified 2012 March 24 by CM:
    When the root node calls chi2 with list_breakdown = 1 (e.g., for the "write" action),
        print a warning if the value of any (delay-)Doppler dataset's Doppler scaling
        factor is out of the allowed range

Modified 2010 August 25 by CM:
    Move TINYCALFACT definition to const.h

Modified 2010 August 10 by CM:
    Implement the "radfitmin" and "radobsmin" parameters: these are the
        pixel values that map to black for all fit and obs pgm images
        that are output for delay-Doppler frames

Modified 2010 March 20 by CM:
    For the "write" action for lightcurve datasets, include magnitude
        uncertainties as a new column in output files fit_MM.dat

Modified 2009 August 9 by CM:
    For the "write" action with the "listfit" parameter turned on, replace
        a ".rdf" or ".fits" or ".fit" suffix with ".fitdat"

Modified 2009 April 3 by CM:
    When the root node calls chi2 with list_breakdown = 1 (e.g., for the
        "write" action), print a warning if any plane-of-sky fit image is
        too small to "contain" all nonzero pixels in the POS sky rendering
        or if the model is too wide in delay-Doppler space for any
        (delay-)Doppler fit frame to be correctly constructed
    For MPI_Recv calls, mpi_par[0] is no longer equal to the MPI action,
        since the message tag argument already serves that purpose (as of
        2008 April 10) -- so the other mpi_par elements are renumbered

Modified 2008 June 29 by CM:
    For the "write" and "orbit" actions, zero out fit pixels/bins in
        delay-Doppler, Doppler, and plane-of-sky frames for which those
        pixels/bins have been zeroed out in the pixel mask

Modified 2008 April 10 by CM:
    Use message tag argument to MPI_Recv to identify the MPI action

Modified 2007 September 13 by CM:
    Implement "write_chi2fit0" parameter: for the "write" and "orbit"
        actions, output chi2 for delay-Doppler, Doppler, and plane-of-sky
        pixels/bins with zero model power; do this both for each individual
        frame, for all delay-Doppler data taken together, for all Doppler
        data taken together, and for all plane-of-sky data taken together

Modified 2007 August 18 by CM:
    Rename MPI_TAG to MPI_TAG_1 to avoid name conflict with mpich headers

Modified 2007 August 10 by CM:
    Create chi2_deldop, chi2_doppler, chi2_poset, and chi2_lghtcrv
        routines so that the code isn't one giant switch statement
    When a calfact value is negative, reset it to a tiny positive value
        rather than to zero in order to protect against division by zero
    Implement the "radfitmax" and "radobsmax" parameters for applying the
        same brightness scale to all fit and obs pgm images output for
        delay-Doppler frames

Modified 2007 August 4 by CM:
    Carry out "write" action steps for the "orbit" action as well

Modified 2007 January 6 by CM:
    For the "write" action for lightcurve datasets, output rotation phases
        to files fit_MM.dat and calc_MM.dat.

Modified 2006 October 1 by CM:
    For lightcurve datasets, chi-square is now computed in intensity space
        rather than in magnitude space.  (File output is still in
        magnitudes.)

Modified 2006 June 21 by CM:
    In delay-Doppler section, changed delres to del_per_pixel and dopres to
        dop_per_pixel
    For POS renderings and plane-of-sky fit frames, changed res to
        km_per_pixel
    When the root node calls chi2 with list_breakdown = 1, print a warning
        if the model extends beyond the boundaries of the POS frames, if
        any photometric parameter has an illegal value, or if any ellipsoid
        diameter is tiny or negative.  (This change will cause such
        warnings to be displayed for the "write" action.)

Modified 2006 June 18 by CM:
    Allow each delay-Doppler frame within a dataset to have different
        dimensions after vignetting
    Allow each Doppler frame within a dataset to have different
        dimensions after vignetting
    Allow plane-of-sky frames to be rectangular rather than square
    For the "write" action, adjust output for delay-Doppler, Doppler, and
        plane-of-sky frames to allow for masked-out pixels

Modified 2005 June 27 by CM:
    Renamed "round" function to "iround" to avoid conflict

Modified 2005 April 25 by CM:
    For the "write" action, compute the one-sigma percentage uncertainty
        on chi2 correctly; the expression used until now, 100*sqrt(2/dof),
        is only valid when all data are weighted equally.

Modified 2005 March 17 by CM:
    For the "fit" action with parallel processing, check that root receives
        the responses to the correct broadcast
    Compute degrees of freedom in routine read_dat rather than here, so
        that it only needs to be done once per fit rather than repeatedly
        (dof for individual data types is still computed here)
    Allow weights and degrees of freedom to be floating-point rather than
        integer; but if they are integer after all, print integer-rounded
        values to the screen rather than floating-point
    For the "write" action with "speckle" turned on, write Doppler file
        fit_MM_NN.dat with spectral values normalized to the input sdev
        value, not to the sdev value increased for self-noise.

Modified 2005 March 6 by CM:
    For the "write" action, write calibration factors for plane-of-sky
        datasets to disk if the "list_posetcal" parameter is turned on

Modified 2005 March 2 by CM:
    Rename some "sdev" and "var" values to be "oneovervar" (1/variance)
    For the "write" action, adjust arguments of revised "resampim" routine
    For the "write" action, rotate plane-of-sky fit/obs/res frames so that
        north is upward, unless poset_scaling = NONE

Modified 2005 February 22 by CM:
    For the "write" action, fix bug (incorrect array dimensions) in
        scaling fit vs. obs pgm image brightness for plane-of-sky datasets
    For the "write" action, add the new "image_rebin" argument to function
        resampim to handle output images which have much coarser resolution
        than the model POS frames from which they are constructed
        (i.e., which are greatly undersampled).  This situation probably
        won't arise often: The real importance of "image_rebin" is for
        dealing with plane-of-sky fit frames in routine calc_fits.

Modified 2005 January 25 by CM:
    Take care of uninitialized variables

Modified 2005 January 20 by CM:
    For the "write" action, implement the bilinear and bicubic
        interpolation options for the "dd_scaling" parameter
    For the "write" action, write model, data, and residual pgm images
        for POS datasets
    Add pgm file output (obs/fit/res) for POS datasets
    Correct the expression for chi-square for POS datasets
        (the calibration factor was being ignored)
    If the calibration factor for a POS frame is a floating parameter,
        don't allow it to be negative: instead set it to zero.  Also
        display a warning, unless this is a fit and we're not at the end
        of an iteration.

Modified 2005 January 12 by CM:
    For the "fit" action with parallel processing, revise the code so
        that "list_breakdown" will work: For each dataset which is handled
        by a branch node rather than by root, root broadcasts a request for
        that branch node to run chi2 for just that one dataset and to send
        the results to root.  This is necessary because otherwise each
        branch node calls chi2 just once to process *all* of its datasets
        in one shot, without keeping results for different data types
        (Doppler, delay-Doppler, etc.) separate.

Modified 2004 December 1 by CM:
    For the "write" action, adjust the "listfit" option so that
        the "listfit_path" directory can be created on the fly
        if necessary

Modified 2004 November 30 by CM:
    For the "write" action, implement the "listfit" option to write
        out the model "data" files for all delay-Doppler frames

Modified 2004 July 26 by CM:
    If the calibration factor for a Doppler or delay-Doppler frame
        is a floating parameter, don't allow it to be negative: instead
        set it to zero.  Also display a warning, unless this is a fit
        and we're not at the end of an iteration.

Modified 2004 June 27 by CM:
    Fixed bug: For 'dd_scaling' = 'block' the number of pixels per side
        for the resampled delay-Doppler obs/fit/residual images wasn't
        being defined

Modified 2004 April 3 by CM:
    Add the "list_breakdown" argument so that we can list chi2 by
        data type (Doppler, delay-Doppler, POS, lightcurves) for
        both the "fit" and the "write" action as desired

Modified 2004 March 20 by CM:
    For the "write" action, move final summary screen display from
        the main program (shape.c) to here
    For the "write" action, display summaries for each form of data
        (Doppler, delay-Doppler, POS, lightcurves) taken separately,
        in addition to the overall summary

Modified 2004 February 29 by CM:
    For lightcurve output, replace JD244 variable with jdoffset parameter
    For the "write" action, move all screen and file output which
        relies on updated calibration factors to this routine

Modified 2003 May 10 by CM:
    Account for contributions to chi-square from model (delay-)Doppler
        power which lies outside the data frame

Modified 2003 April 24 by CM:
    Display chi-square for Doppler datasets when action = "write"
    Implement the new "weight" parameter
 *****************************************************************************************/

#include "head.h"

double chi2_deldop( struct par_t *par, struct deldop_t *deldop, int list_breakdown,
		int s, double *chi2_all_deldop, double *chi2_fit0_deldop,
		double *dof_fit0_deldop);
double chi2_doppler( struct par_t *par, struct doppler_t *doppler, int list_breakdown,
		int s, double *chi2_all_doppler, double *chi2_fit0_doppler,
		double *dof_fit0_doppler);
double chi2_poset( struct par_t *par, struct poset_t *poset, int list_breakdown,
		int s, double *chi2_all_poset, double *chi2_fit0_poset,
		double *dof_fit0_poset);
double chi2_lghtcrv( struct par_t *par, struct lghtcrv_t *lghtcrv, int list_breakdown,
		int s, double *chi2_all_lghtcrv);


double chi2( struct par_t *par, struct dat_t *dat, int list_breakdown)
{
	char dofstring[MAXLEN], dof0string[MAXLEN];
	int s, set_min, set_max, print_breakdown;
	double chi2_all_doppler, chi2_all_deldop, chi2_all_poset, chi2_all_lghtcrv,
	chi2_fit0_doppler, chi2_fit0_deldop, chi2_fit0_poset, chi2_branch,
	dof_fit0_doppler, dof_fit0_deldop, dof_fit0_poset;

	/*  Initialize variables that accumulate chi-square values  */

	dat->chi2 = chi2_all_deldop = chi2_all_doppler = chi2_all_poset
			= chi2_all_lghtcrv = chi2_fit0_deldop = chi2_fit0_doppler
			= chi2_fit0_poset = 0.0;
	dof_fit0_deldop = dof_fit0_doppler = dof_fit0_poset = 0.0;

	/*  If we're parallel processing with list_breakdown = 1, AND this is a
      branch node calling the chi2 routine, we only process a single dataset
      on this call.  In such cases mpi_par[1] has already been assigned by
      the root node via the "MPI_CHI2BREAKDOWN" MPI action: see the
      MPI_Bcast call further down in this routine.

      If we're not parallel processing, or if list_breakdown = 0, or if this
      is the root node, we loop through ALL datasets on this call to chi2.    */

	set_min = 0;
	set_max = dat->nsets - 1;

	/*  Loop through all datasets, carry out chi-square computations,
      and provide screen and image output                            */

	for (s=set_min; s<=set_max; s++) {
		switch (dat->set[s].type) {
		case DELAY:
			dat->set[s].chi2 = chi2_deldop( par, &dat->set[s].desc.deldop,
					list_breakdown, s, &chi2_all_deldop,
					&chi2_fit0_deldop, &dof_fit0_deldop);
			break;
		case DOPPLER:
			dat->set[s].chi2 = chi2_doppler( par, &dat->set[s].desc.doppler,
					list_breakdown, s, &chi2_all_doppler,
					&chi2_fit0_doppler, &dof_fit0_doppler);
			break;
		case POS:
			dat->set[s].chi2 = chi2_poset( par, &dat->set[s].desc.poset,
					list_breakdown, s, &chi2_all_poset,
					&chi2_fit0_poset, &dof_fit0_poset);
			break;
		case LGHTCRV:
			dat->set[s].chi2 = chi2_lghtcrv( par, &dat->set[s].desc.lghtcrv,
					list_breakdown, s, &chi2_all_lghtcrv);
			break;
		default:
			bailout("chi2.c: can't handle this type yet\n");
		}

		dat->chi2 += dat->set[s].chi2;

	}  /* end for loop over datasets */

	/*  If specified, display the breakdown of chi-square information
      for each separate form of data (delay-Doppler, Doppler, POS,
      lightcurve) and for all datasets taken together.  Display
      degrees of freedom as integer if possible.                      */

	if (list_breakdown) {
		printf("#\n");
		print_breakdown = (dat->dof_deldop > SMALLVAL || dat->dof_doppler  > SMALLVAL
				|| dat->dof_poset    > SMALLVAL
				|| dat->dof_lghtcrv  > SMALLVAL);
		if (print_breakdown) {
			if (dat->dof_deldop > SMALLVAL) {
				intifpossible( dofstring, MAXLEN, dat->dof_deldop, SMALLVAL, "%f");
				printf("delay   chi2 = %e for %s dof (reduced chi2 = %f)\n",
						chi2_all_deldop, dofstring, chi2_all_deldop/dat->dof_deldop);
				if (par->write_chi2fit0) {
					intifpossible( dof0string, MAXLEN, dof_fit0_deldop, SMALLVAL, "%f");
					printf("              (%e outside model for %s dof)\n",
							chi2_fit0_deldop, dof0string);
				}
			}
			if (dat->dof_doppler > SMALLVAL) {
				intifpossible( dofstring, MAXLEN, dat->dof_doppler, SMALLVAL, "%f");
				printf("Doppler chi2 = %e for %s dof (reduced chi2 = %f)\n",
						chi2_all_doppler, dofstring, chi2_all_doppler/dat->dof_doppler);
				if (par->write_chi2fit0) {
					intifpossible( dof0string, MAXLEN, dof_fit0_doppler, SMALLVAL, "%f");
					printf("              (%e outside model for %s dof)\n",
							chi2_fit0_doppler, dof0string);
				}
			}
			if (dat->dof_poset > SMALLVAL) {
				intifpossible( dofstring, MAXLEN, dat->dof_poset, SMALLVAL, "%f");
				printf("POS     chi2 = %e for %s dof (reduced chi2 = %f)\n",
						chi2_all_poset, dofstring, chi2_all_poset/dat->dof_poset);
				if (par->write_chi2fit0) {
					intifpossible( dof0string, MAXLEN, dof_fit0_poset, SMALLVAL, "%f");
					printf("              (%e outside model for %s dof)\n",
							chi2_fit0_poset, dof0string);
				}
			}
			if (dat->dof_lghtcrv > SMALLVAL) {
				intifpossible( dofstring, MAXLEN, dat->dof_lghtcrv, SMALLVAL, "%f");
				printf("lghtcrv chi2 = %e for %s dof (reduced chi2 = %f)\n",
						chi2_all_lghtcrv, dofstring, chi2_all_lghtcrv/dat->dof_lghtcrv);
			}
			intifpossible( dofstring, MAXLEN, dat->dof, SMALLVAL, "%f");
			printf("ALLDATA chi2 = %e for %s dof (reduced chi2 = %f)",
					dat->chi2, dofstring, dat->chi2/dat->dof);
		} else {
			intifpossible( dofstring, MAXLEN, dat->dof, SMALLVAL, "%f");
			printf("        chi2 = %e for %s dof (reduced chi2 = %f)",
					dat->chi2, dofstring, dat->chi2/dat->dof);
		}
		if (par->baddiam)
			printf("  (BAD DIAMS)");
		if (par->badphoto)
			printf("  (BAD PHOTO)");
		if (par->posbnd)
			printf("  (BAD POS)");
		if (par->badposet)
			printf("  (BAD POSET)");
		if (par->badradar)
			printf("  (BAD RADAR)");
		if (par->baddopscale)
			printf("  (BAD DOPSCALE)");
		printf("\n");
		if (print_breakdown && par->write_chi2fit0) {
			intifpossible( dof0string, MAXLEN,
					dof_fit0_deldop + dof_fit0_doppler + dof_fit0_poset,
					SMALLVAL, "%f");
			printf("              (%e outside model for %s dof)\n",
					chi2_fit0_deldop + chi2_fit0_doppler + chi2_fit0_poset, dof0string);
		}
		printf("#\n");
		fflush(stdout);
	}

	/*  For the "write" and "orbit" actions, compute the one-sigma
      percentage uncertainty on chi-square; the expectation value is
      the number of degrees of freedom (= sum over all frames and
      lightcurves of weight*n_points), and the variance is the sum
      of 2*(weight^2)*n_points                                        */

	if (par->action == WRITE || par->action == ORBIT) {
		printf("# chi2 one-sigma uncertainty is %4.1f percent\n",
				100*sqrt(dat->chi2_variance)/dat->dof);
		fflush(stdout);
	}

	return dat->chi2;
}

double chi2_deldop( struct par_t *par, struct deldop_t *deldop, int list_breakdown,
		int s, double *chi2_all_deldop, double *chi2_fit0_deldop,
		double *dof_fit0_deldop)
{
	FILE *fp;
	char name[MAXLEN], fitname[MAXLEN], obsname[MAXLEN], tempstring[MAXLEN],
	weightstring[MAXLEN], dofstring[MAXLEN], dof0string[MAXLEN];
	int f, i, j, ndel, ndop, n, n_pix;
	double chi2_set, err, err_fit0, o2, m2, om, o2_fit0, del0, ddel, dop0, ddop,
	calval, fit255, obs255, fit255_use, obs255_use, weight, dof, dof_fit0,
	thresh_fit0;
	double **obs, **fit, **res, **oneovervar, **resamp_fit, **resamp_obs,
	**resamp_res;

	/* Initialize variables to avoid compilation warnings  */
	n = 0;
	del0 = ddel = dop0 = ddop = 0.0;

	/* Initialize chi-square for dataset  */
	chi2_set = 0.0;

	/* Loop through all frames for this dataset  */

	for (f=0; f<deldop->nframes; f++) {
		ndel = deldop->frame[f].ndel;
		ndop = deldop->frame[f].ndop;
		obs = deldop->frame[f].obs;
		fit = deldop->frame[f].fit;
		oneovervar = deldop->frame[f].oneovervar;  /* 1/variance */
		weight = deldop->frame[f].weight;
		dof = deldop->frame[f].dof;

		/* Initialize contributions to chi-square to values that account for
		 * overflow (if any) beyond the limits of the data frame.  These
		 * contributions were computed by routine pos2deldop.              */
		o2 = deldop->frame[f].overflow_o2;
		m2 = deldop->frame[f].overflow_m2;
		om = 0.0;

		/*  Now add the contributions from power
        within the limits of the data frame.  */
		for (i=1; i<=ndel; i++)
			for (j=1; j<=ndop; j++) {
				o2 += obs[i][j]*obs[i][j]*oneovervar[i][j];
				m2 += fit[i][j]*fit[i][j]*oneovervar[i][j];
				om += fit[i][j]*obs[i][j]*oneovervar[i][j];
			}

		/*  If this frame's calibration factor is allowed to float,
        set it to minimize chi-square, the sum over all pixels of
        { (obs - calfact*fit)^2 / variance }.                       */

		if (deldop->frame[f].cal.state == 'f') {
			if (om > 0.0) {
				deldop->frame[f].cal.val = om/m2;
			} else {
				deldop->frame[f].cal.val = TINYCALFACT;
				if (par->action != FIT || list_breakdown)
					printf("WARNING: set %2d frame %2d had negative calfact reset to %10.4e\n",
							s, f, deldop->frame[f].cal.val);
			}
		}

		/*  Compute chi-square for this frame  */
		calval = deldop->frame[f].cal.val;
		err = weight*(o2 - 2*calval*om + calval*calval*m2);
		deldop->frame[f].chi2 = err;

		chi2_set += err;
		if (list_breakdown)
			*chi2_all_deldop += err;

		/*  Compute the chi-square contributions and number of degrees of freedom
        due to pixels whose model signal is less than or equal to
        'chi2fit0_thresh' standard deviations of the noise in the data frame   */

		o2_fit0 = 0.0;
		dof_fit0 = 0.0;
		err_fit0 = 0.0;
		thresh_fit0 = par->chi2fit0_thresh * deldop->frame[f].sdev;
		if (list_breakdown && par->write_chi2fit0) {
			for (i=1; i<=ndel; i++)
				for (j=1; j<=ndop; j++)
					if (calval*fit[i][j] <= thresh_fit0) {
						o2_fit0 += obs[i][j]*obs[i][j]*oneovervar[i][j];
						if (oneovervar[i][j] > 0.0)
							dof_fit0 += weight;
					}
			err_fit0 = weight*o2_fit0;
			*chi2_fit0_deldop += err_fit0;
			*dof_fit0_deldop += dof_fit0;
		}

		/*  For the "write" and "orbit" actions, display chi-square
        and create files for the data, fits, and residuals; some
        of these files must be written in this routine because
        they rely on updated values of the calibration factor.    */

		if (par->action == WRITE || par->action == ORBIT) {

			/*  Zero out obs and fit pixels that have been zeroed out in a pixel mask  */

			if (deldop->frame[f].pixels_weighted)
				for (i=1; i<=ndel; i++)
					for (j=1; j<=ndop; j++)
						if (oneovervar[i][j] == 0.0) {
							obs[i][j] = 0.0;
							fit[i][j] = 0.0;
						}

			/*  Display weight and degrees of freedom as integers if possible  */

			intifpossible( weightstring, MAXLEN, weight, SMALLVAL, "%f");
			intifpossible( dofstring, MAXLEN, dof, SMALLVAL, "%f");
			if (dof == weight*ndel*ndop) {
				printf("chi2 of set %2d frame %2d is %f    for %s x (%d x %d) = %s dof",
						s, f, err, weightstring, ndel, ndop, dofstring);
			} else {
				n_pix = iround( dof/weight);
				printf("chi2 of set %2d frame %2d is %f    for %s x (%d pixels) = %s dof",
						s, f, err, weightstring, n_pix, dofstring);
			}
			printf("    (red. chi2 = %.2f)\n", err/dof);
			if (par->write_chi2fit0) {
				intifpossible( dof0string, MAXLEN, dof_fit0, SMALLVAL, "%f");
				printf("    (outside model, chi2 = %f for %s dof)\n", err_fit0, dof0string);
			}
			fflush(stdout);

			/*  Write the model "data" to a text file if desired  */

			if (par->listfit) {

				/*  Create the "listfit_path" directory if necessary; note that
            createdir will return -1 if the directory already exists,
            0 if it can't create it, or 1 if it successfully creates it  */

				if (!createdir(par->listfit_path)) {
					printf("Unable to create 'listfit_path' directory\n");
					bailout("chi2.c: program halted\n");
				}

				/*  Create the output filename: the same as the name of the
            actual datafile, but with a new path, with any ".dat" suffix
            removed, and with the ".fitdat" suffix added                  */

				changepath(tempstring, MAXLEN, deldop->frame[f].name, par->listfit_path);
				if ( !(addsuffix(name, MAXLEN, tempstring, ".rdf",  ".fitdat") ||
						addsuffix(name, MAXLEN, tempstring, ".fits", ".fitdat") ||
						addsuffix(name, MAXLEN, tempstring, ".fit",  ".fitdat")    ) )
					addsuffix(name, MAXLEN, tempstring, ".dat",  ".fitdat");
				if (!strcmp(name, deldop->frame[f].name)) {
					printf("listfit would overwrite set %d frame %d datafile\n", s, f);
					bailout("chi2.c: Halt program instead\n");
				}

				/*  Write the normalized model pixel values to the file  */

				FOPEN( fp, name, "w");
				for (i=1; i<=ndel; i++)
					for (j=1; j<=ndop; j++)
						fprintf(fp, "%13.6e\n",
								calval * fit[i][j] / deldop->frame[f].sdev );
				fclose( fp);
			}

			/*  Resample the data and the fit if desired; then compute the
          maximum pixel value for the data and (cal*fit) frames, so
          that those two pgm images can be on the same scale if
          desired.  (fit255 and obs255 refer to the values which will
          map to 255 = bright white.)  Lastly, write the pgm files.    */

			if (deldop->nframes > 100) {
				sprintf( fitname, "fit_%02d_%03d.pgm", s, f);
				sprintf( obsname, "obs_%02d_%03d.pgm", s, f);
			} else {
				sprintf( fitname, "fit_%02d_%02d.pgm", s, f);
				sprintf( obsname, "obs_%02d_%02d.pgm", s, f);
			}
			fit255 = -1.0e20;
			obs255 = -1.0e20;
			if (par->dd_scaling == NONE) {
				for (i=1; i<=ndel; i++)
					for (j=1; j<=ndop; j++) {
						fit255 = MAX( fit255, fit[i][j]);
						obs255 = MAX( obs255, obs[i][j]);
					}
				fit255_use = (par->radfitmax == 0.0) ? fit255 : par->radfitmax / calval;
				obs255_use = (par->radobsmax == 0.0) ? obs255 : par->radobsmax;
				if (par->scalefitobs == SCALE_MAXFITOBS) {
					if (obs255_use > calval*fit255_use)
						fit255_use = obs255_use/calval;
					else
						obs255_use = calval*fit255_use;
				} else if (par->scalefitobs == SCALE_FIT) {
					obs255_use = calval*fit255_use;
				} else if (par->scalefitobs == SCALE_OBS) {
					fit255_use = obs255_use/calval;
				}
				if (par->radfitmax != 0.0)
					printf("# %s  (max pixel value = %10.4e, bright white >= %10.4e)\n",
							fitname, calval*fit255, calval*fit255_use);
				if (par->radobsmax != 0.0)
					printf("# %s  (max pixel value = %10.4e, bright white >= %10.4e)\n",
							obsname, obs255, obs255_use);
				wimaspgmsc( fit,
						1, ndel, 1, ndop, par->radfitmin, fit255_use,
						par->dd_clockwiserot, par->dd_xflip, par->dd_yflip,
						fitname);
				wimaspgmsc( obs,
						1, ndel, 1, ndop, par->radobsmin, obs255_use,
						par->dd_clockwiserot, par->dd_xflip, par->dd_yflip,
						obsname);
			} else {
				del0 = deldop->frame[f].delcom_vig
						+ deldop->frame[f].view[deldop->v0].deloff;
				/* ddel is pos_width / (km per delay bin) */
				ddel = (par->pos_width * KM2US) / deldop->del_per_pixel;
				dop0 = deldop->frame[f].dopcom_vig + deldop->frame[f].view[deldop->v0].dopoff;
				ddop = par->pos_width * deldop->frame[f].view[deldop->v0].km2Hz
						/ deldop->dop_per_pixel;
				n = deldop->frame[f].pos.n;
				resamp_fit = matrix( -n, n, -n, n);
				resampim( fit, 1, ndel, 1, ndop,
						resamp_fit, -n, n, -n, n,
						del0, ddel, dop0, ddop, 0.0,
						(int) par->dd_scaling, (int) par->image_rebin);
				resamp_obs = matrix( -n, n, -n, n);
				resampim( obs, 1, ndel, 1, ndop,
						resamp_obs, -n, n, -n, n,
						del0, ddel, dop0, ddop, 0.0,
						(int) par->dd_scaling, (int) par->image_rebin);
				for (i=-n; i<=n; i++)
					for (j=-n; j<=n; j++) {
						fit255 = MAX( fit255, resamp_fit[i][j]);
						obs255 = MAX( obs255, resamp_obs[i][j]);
					}
				fit255_use = (par->radfitmax == 0.0) ? fit255 : par->radfitmax / calval;
				obs255_use = (par->radobsmax == 0.0) ? obs255 : par->radobsmax;
				if (par->scalefitobs == SCALE_MAXFITOBS) {
					if (obs255_use > calval*fit255_use)
						fit255_use = obs255_use/calval;
					else
						obs255_use = calval*fit255_use;
				} else if (par->scalefitobs == SCALE_FIT) {
					obs255_use = calval*fit255_use;
				} else if (par->scalefitobs == SCALE_OBS) {
					fit255_use = obs255_use/calval;
				}
				if (par->radfitmax != 0.0)
					printf("# %s  (max pixel value = %10.4e, bright white >= %10.4e)\n",
							fitname, calval*fit255, calval*fit255_use);
				if (par->radobsmax != 0.0)
					printf("# %s  (max pixel value = %10.4e, bright white >= %10.4e)\n",
							obsname, obs255, obs255_use);
				wimaspgmsc( resamp_fit, -n, n, -n, n, par->radfitmin, fit255_use,
						par->dd_clockwiserot, par->dd_xflip, par->dd_yflip,
						fitname);
				wimaspgmsc( resamp_obs, -n, n, -n, n, par->radobsmin, obs255_use,
						par->dd_clockwiserot, par->dd_xflip, par->dd_yflip,
						obsname);
				free_matrix( resamp_fit, -n, n, -n, n);
				free_matrix( resamp_obs, -n, n, -n, n);
			}

			/*  Compute the array of normalized fit residuals  */

			res = matrix( 1, ndel, 1, ndop);
			for (i=1; i<=ndel; i++)
				for (j=1; j<=ndop; j++)
					res[i][j] = (obs[i][j] - calval*fit[i][j]) * sqrt(oneovervar[i][j]);

			/*  Write rounded residuals to disk if desired  */

			if (par->listres) {
				if (deldop->nframes > 100)
					sprintf( name, "res_%02d_%03d.lst", s, f);
				else
					sprintf( name, "res_%02d_%02d.lst", s, f);
				FOPEN( fp, name, "w");
				for (i=1; i<=ndel; i++) {
					fprintf( fp, "%4d:", i);
					for (j=1; j<=ndop; j++)
						fprintf( fp, "%5d", (int) floor(res[i][j] + 0.5));
					fprintf( fp, "\n");
				}
				fclose( fp);
			}

			/*  Write the floating-point absolute values of the residuals,
          first resampling the residual image if desired              */

			if (par->dd_resid > 0.0) {
				for (i=1; i<=ndel; i++)
					for (j=1; j<=ndop; j++)
						res[i][j] = fabs(res[i][j]);
				if (deldop->nframes > 100)
					sprintf( name, "res_%02d_%03d.pgm", s, f);
				else
					sprintf( name, "res_%02d_%02d.pgm", s, f);
				if (par->dd_scaling == NONE) {
					wimaspgmsc( res, 1, ndel, 1, ndop,
							0.0, par->dd_resid,
							par->dd_clockwiserot, par->dd_xflip, par->dd_yflip,
							name);
				} else {
					resamp_res = matrix( -n, n, -n, n);
					resampim( res, 1, ndel, 1, ndop,
							resamp_res, -n, n, -n, n,
							del0, ddel, dop0, ddop, 0.0,
							(int) par->dd_scaling, (int) par->image_rebin);
					wimaspgmsc( resamp_res, -n, n, -n, n, 0.0, par->dd_resid,
							par->dd_clockwiserot, par->dd_xflip, par->dd_yflip,
							name);
					free_matrix( resamp_res, -n, n, -n, n);
				}
			}
			free_matrix( res, 1, ndel, 1, ndop);

		}  /* end of "write" block */

	}  /* end for loop over frames */

	return chi2_set;
}

double chi2_doppler( struct par_t *par, struct doppler_t *doppler, int list_breakdown,
		int s, double *chi2_all_doppler, double *chi2_fit0_doppler,
		double *dof_fit0_doppler)
{
	FILE *fp;
	char name[MAXLEN], weightstring[MAXLEN], dofstring[MAXLEN], dof0string[MAXLEN];
	int f, j, ndop, n_bins;
	double chi2_set, err, err_fit0, o2, m2, om, o2_fit0, sc, calval, weight, dof,
	dof_fit0, thresh_fit0;
	double *obs, *fit, *oneovervar;

	/*  Initialize chi-square for dataset  */

	chi2_set = 0.0;

	/*  Loop through all frames for this dataset  */

	for (f=0; f<doppler->nframes; f++) {
		ndop = doppler->frame[f].ndop;
		obs = doppler->frame[f].obs;
		fit = doppler->frame[f].fit;
		oneovervar = doppler->frame[f].oneovervar;  /* 1/variance */
		weight = doppler->frame[f].weight;
		dof = doppler->frame[f].dof;

		/*  Initialize contributions to chi-square to values
        that account for overflow (if any) beyond the
        limits of the data frame.  These contributions
        were computed by routine pos2doppler.              */

		o2 = doppler->frame[f].overflow_o2;
		m2 = doppler->frame[f].overflow_m2;
		om = 0.0;

		/*  Now add the contributions from power
        within the limits of the data frame.  */

		for (j=1; j<=ndop; j++) {
			o2 += obs[j]*obs[j]*oneovervar[j];
			m2 += fit[j]*fit[j]*oneovervar[j];
			om += fit[j]*obs[j]*oneovervar[j];
		}

		/*  If this frame's calibration factor is allowed to float,
        set it to minimize chi-square, the sum over all bins of
        { (obs - calfact*fit)^2 / variance }.                     */

		if (doppler->frame[f].cal.state == 'f') {
			if (om > 0.0) {
				doppler->frame[f].cal.val = om/m2;
			} else {
				doppler->frame[f].cal.val = TINYCALFACT;
				if (par->action != FIT || list_breakdown)
					printf("WARNING: set %2d frame %2d had negative calfact reset to %10.4e\n",
							s, f, doppler->frame[f].cal.val);
			}
		}

		/*  Compute chi-square for this frame  */

		calval = doppler->frame[f].cal.val;
		err = weight*(o2 - 2*calval*om + calval*calval*m2);
		doppler->frame[f].chi2 = err;
		chi2_set += err;
		if (list_breakdown)
			*chi2_all_doppler += err;

		/*  Compute the chi-square contributions and number of degrees of freedom
        due to bins whose model signal is less than or equal to
        'chi2fit0_thresh' standard deviations of the noise in the data frame   */

		o2_fit0 = 0.0;
		dof_fit0 = 0.0;
		err_fit0 = 0.0;
		thresh_fit0 = par->chi2fit0_thresh * doppler->frame[f].sdev;
		if (list_breakdown && par->write_chi2fit0) {
			for (j=1; j<=ndop; j++)
				if (calval*fit[j] <= thresh_fit0) {
					o2_fit0 += obs[j]*obs[j]*oneovervar[j];
					if (oneovervar[j] > 0.0)
						dof_fit0 += weight;
				}
			err_fit0 = weight*o2_fit0;
			*chi2_fit0_doppler += err_fit0;
			*dof_fit0_doppler += dof_fit0;
		}

		/*  For the "write" and "orbit" actions, display chi-square
        and create a file for the data, fits, and residuals;
        this file must be written in this routine because it
        relies on the updated value of the calibration factor.   */

		if (par->action == WRITE || par->action == ORBIT) {

			/*  Zero out obs and fit bins that have been zeroed out in a pixel mask  */

			if (doppler->frame[f].pixels_weighted)
				for (j=1; j<=ndop; j++)
					if (oneovervar[j] == 0.0) {
						obs[j] = 0.0;
						fit[j] = 0.0;
					}

			/*  Display weight and degrees of freedom as integers if possible  */

			intifpossible( weightstring, MAXLEN, weight, SMALLVAL, "%f");
			intifpossible( dofstring, MAXLEN, dof, SMALLVAL, "%f");
			if (dof == weight*ndop) {
				printf("chi2 of set %2d frame %2d is %f    for %s x (1 x %d) = %s dof",
						s, f, err, weightstring, ndop, dofstring);
			} else {
				n_bins = iround( dof/weight);
				printf("chi2 of set %2d frame %2d is %f    for %s x (%d bins) = %s dof",
						s, f, err, weightstring, n_bins, dofstring);
			}
			printf("    (red. chi2 = %.2f)\n", err/dof);
			if (par->write_chi2fit0) {
				intifpossible( dof0string, MAXLEN, dof_fit0, SMALLVAL, "%f");
				printf("    (outside model, chi2 = %f for %s dof)\n", err_fit0, dof0string);
			}
			fflush(stdout);

			/*  Write normalized fits to disk  */

			if (doppler->nframes > 100)
				sprintf( name, "fit_%02d_%03d.dat", s, f);
			else
				sprintf( name, "fit_%02d_%02d.dat", s, f);
			FOPEN( fp, name, "w");
			for (j=1; j<=ndop; j++) {
				sc = 1/doppler->frame[f].sdev;
				fprintf( fp, "%d %f %f %f\n", j,
						obs[j]*sc, calval*fit[j]*sc, (obs[j] - calval*fit[j])*sc);
			}
			fclose( fp);
		}
	}
	return chi2_set;
}

double chi2_poset( struct par_t *par, struct poset_t *poset, int list_breakdown,
		int s, double *chi2_all_poset, double *chi2_fit0_poset,
		double *dof_fit0_poset)
{
	FILE *fp=NULL;
	char name[MAXLEN], weightstring[MAXLEN], dofstring[MAXLEN], dof0string[MAXLEN];
	int f, i, j, n_pix, n_pos, nrow, ncol;
	double chi2_set, err, err_fit0, o2, m2, om, o2_fit0, calval, fit255, obs255,
	xoff, yoff, resamp_fact, resamp_x0, resamp_y0, resamp_width,
	resamp_angle, weight, dof, dof_fit0, thresh_fit0;
	double **obs, **fit, **res, **oneovervar, **resamp_fit, **resamp_obs,
	**resamp_res;

	/*  Initialize chi-square for dataset  */

	chi2_set = 0.0;

	/*  For the "write" and "orbit" actions,
      open calibration factor file if desired  */

	if ((par->action == WRITE || par->action == ORBIT) && par->list_posetcal) {
		sprintf( name, "calfact_%02d.lst", s);
		FOPEN( fp, name, "w");
		fprintf(fp, "set frm       calfact\n");
	}

	/*  Loop through all frames for this dataset  */

	for (f=0; f<poset->nframes; f++) {
		ncol = poset->frame[f].ncol;
		nrow = poset->frame[f].nrow;
		obs = poset->frame[f].obs.b;
		fit = poset->frame[f].fit.b;
		oneovervar = poset->frame[f].oneovervar;  /* 1/variance */
		weight = poset->frame[f].weight;
		dof = poset->frame[f].dof;

		o2 = m2 = om = 0.0;
		for (i=1; i<=ncol; i++)
			for (j=1; j<=nrow; j++) {
				o2 += obs[i][j]*obs[i][j]*oneovervar[i][j];
				m2 += fit[i][j]*fit[i][j]*oneovervar[i][j];
				om += fit[i][j]*obs[i][j]*oneovervar[i][j];
			}

		/*  The calibration factor always floats for plane-of-sky frames:
        set it to minimize chi-square, the sum over all pixels of
        { (obs - calfact*fit)^2 / variance }.                          */

		if (om > 0.0) {
			poset->frame[f].cal.val = om/m2;
		} else {
			poset->frame[f].cal.val = TINYCALFACT;
			if (par->action != FIT || list_breakdown)
				printf("WARNING: set %2d frame %2d had negative calfact reset to %10.4e\n",
						s, f, poset->frame[f].cal.val);
		}

		/*  Compute chi-square for this frame  */

		calval = poset->frame[f].cal.val;
		err = weight*(o2 - 2*calval*om + calval*calval*m2);
		poset->frame[f].chi2 = err;
		chi2_set += err;
		if (list_breakdown)
			*chi2_all_poset += err;

		/*  Compute the chi-square contributions and number of degrees of freedom
        due to pixels whose model signal is less than or equal to
        'chi2fit0_thresh' standard deviations of the noise in the data frame   */

		o2_fit0 = 0.0;
		dof_fit0 = 0.0;
		err_fit0 = 0.0;
		thresh_fit0 = par->chi2fit0_thresh;  /* "sdev" = 1.0 for plane-of-sky data */
		if (list_breakdown && par->write_chi2fit0) {
			for (i=1; i<=ncol; i++)
				for (j=1; j<=nrow; j++)
					if (calval*fit[i][j] <= thresh_fit0) {
						o2_fit0 += obs[i][j]*obs[i][j]*oneovervar[i][j];
						if (oneovervar[i][j] > 0.0)
							dof_fit0 += weight;
					}
			err_fit0 = weight*o2_fit0;
			*chi2_fit0_poset += err_fit0;
			*dof_fit0_poset += dof_fit0;
		}

		/*  For the "write" and "orbit" actions, display chi-square
        and create files for the data, fits, and residuals; some
        of these files must be written in this routine because
        they rely on updated values of the calibration factor.    */

		if (par->action == WRITE || par->action == ORBIT) {

			/*  Zero out obs and fit pixels that have been zeroed out in a pixel mask  */

			if (poset->frame[f].pixels_weighted)
				for (i=1; i<=ncol; i++)
					for (j=1; j<=nrow; j++)
						if (oneovervar[i][j] == 0.0) {
							obs[i][j] = 0.0;
							fit[i][j] = 0.0;
						}

			/*  Display weight and degrees of freedom as integers if possible  */

			intifpossible( weightstring, MAXLEN, weight, SMALLVAL, "%f");
			intifpossible( dofstring, MAXLEN, dof, SMALLVAL, "%f");
			if (dof == weight*ncol*nrow) {
				printf("chi2 of set %2d frame %2d is %f    for %s x (%d x %d) = %s dof",
						s, f, err, weightstring, nrow, ncol, dofstring);
			} else {
				n_pix = iround( dof/weight);
				printf("chi2 of set %2d frame %2d is %f    for %s x (%d pixels) = %s dof",
						s, f, err, weightstring, n_pix, dofstring);
			}
			printf("    (red. chi2 = %.2f)\n", err/dof);
			if (par->write_chi2fit0) {
				intifpossible( dof0string, MAXLEN, dof_fit0, SMALLVAL, "%f");
				printf("    (outside model, chi2 = %f for %s dof)\n", err_fit0, dof0string);
			}
			fflush(stdout);

			/*  Write this frame's calibration factor to disk if desired  */

			if (par->list_posetcal)
				fprintf(fp, "%3d %3d %13.6e\n", s, f, calval);

			/*  Resample the data and the fit if desired; then compute the
          maximum pixel value for the data and (cal*fit) frames, so
          that those two pgm images can be on the same scale if
          desired.  (fit255 and obs255 refer to the values which will
          map to 255 = bright white.)  Lastly, write the pgm files.    */

			n_pos = poset->frame[f].pos.n;
			xoff = poset->frame[f].off[0].val;
			yoff = poset->frame[f].off[1].val;
			resamp_fact = poset->frame[f].pos.km_per_pixel
					/ poset->frame[f].fit.km_per_pixel;
			resamp_x0 = poset->frame[f].colcom_vig + xoff;
			resamp_y0 = poset->frame[f].rowcom_vig + yoff;
			resamp_width = 2*n_pos*resamp_fact;
			resamp_angle = poset->frame[f].northangle;
			fit255 = -1.0e20;
			obs255 = -1.0e20;
			if (par->poset_scaling == NONE) {
				for (i=1; i<=ncol; i++)
					for (j=1; j<=nrow; j++) {
						fit255 = MAX( fit255, fit[i][j]);
						obs255 = MAX( obs255, obs[i][j]);
					}
				if (par->scalefitobs == SCALE_MAXFITOBS) {
					if (obs255 > calval*fit255)
						fit255 = obs255/calval;
					else
						obs255 = calval*fit255;
				} else if (par->scalefitobs == SCALE_FIT) {
					obs255 = calval*fit255;
				} else if (par->scalefitobs == SCALE_OBS) {
					fit255 = obs255/calval;
				}
				if (poset->nframes > 100)
					sprintf( name, "fit_%02d_%03d.pgm", s, f);
				else
					sprintf( name, "fit_%02d_%02d.pgm", s, f);
				wimaspgmsc( fit,
						1, ncol, 1, nrow, 0.0, fit255, 0, 0, 0, name);
				if (poset->nframes > 100)
					sprintf( name, "obs_%02d_%03d.pgm", s, f);
				else
					sprintf( name, "obs_%02d_%02d.pgm", s, f);
				wimaspgmsc( obs,
						1, ncol, 1, nrow, 0.0, obs255, 0, 0, 0, name);
			} else {
				resamp_fit = matrix( -n_pos, n_pos, -n_pos, n_pos);
				resampim( fit, 1, ncol, 1, nrow,
						resamp_fit, -n_pos, n_pos, -n_pos, n_pos,
						resamp_x0, resamp_width, resamp_y0, resamp_width, resamp_angle,
						(int) par->poset_scaling, (int) par->image_rebin);
				resamp_obs = matrix( -n_pos, n_pos, -n_pos, n_pos);
				resampim( obs, 1, ncol, 1, nrow,
						resamp_obs, -n_pos, n_pos, -n_pos, n_pos,
						resamp_x0, resamp_width, resamp_y0, resamp_width, resamp_angle,
						(int) par->poset_scaling, (int) par->image_rebin);
				for (i=-n_pos; i<=n_pos; i++)
					for (j=-n_pos; j<=n_pos; j++) {
						fit255 = MAX( fit255, resamp_fit[i][j]);
						obs255 = MAX( obs255, resamp_obs[i][j]);
					}
				if (par->scalefitobs == SCALE_MAXFITOBS) {
					if (obs255 > calval*fit255)
						fit255 = obs255/calval;
					else
						obs255 = calval*fit255;
				} else if (par->scalefitobs == SCALE_FIT) {
					obs255 = calval*fit255;
				} else if (par->scalefitobs == SCALE_OBS) {
					fit255 = obs255/calval;
				}
				if (poset->nframes > 100)
					sprintf( name, "fit_%02d_%03d.pgm", s, f);
				else
					sprintf( name, "fit_%02d_%02d.pgm", s, f);
				wimaspgmsc( resamp_fit, -n_pos, n_pos, -n_pos, n_pos,
						0.0, fit255, 0, 0, 0, name);
				if (poset->nframes > 100)
					sprintf( name, "obs_%02d_%03d.pgm", s, f);
				else
					sprintf( name, "obs_%02d_%02d.pgm", s, f);
				wimaspgmsc( resamp_obs, -n_pos, n_pos, -n_pos, n_pos,
						0.0, obs255, 0, 0, 0, name);
				free_matrix( resamp_fit, -n_pos, n_pos, -n_pos, n_pos);
				free_matrix( resamp_obs, -n_pos, n_pos, -n_pos, n_pos);
			}

			/*  Compute the array of normalized fit residuals  */

			res = matrix( 1, ncol, 1, nrow);
			for (i=1; i<=ncol; i++)
				for (j=1; j<=nrow; j++)
					res[i][j] = (obs[i][j] - calval*fit[i][j]);

			/*  Write the floating-point absolute values of the residuals,
          first resampling the residual image if desired              */

			if (par->poset_resid > 0.0) {
				for (i=1; i<=ncol; i++)
					for (j=1; j<=nrow; j++)
						res[i][j] = fabs(res[i][j]);
				if (poset->nframes > 100)
					sprintf( name, "res_%02d_%03d.pgm", s, f);
				else
					sprintf( name, "res_%02d_%02d.pgm", s, f);
				if (par->poset_scaling == NONE) {
					wimaspgmsc( res, 1, ncol, 1, nrow,
							0.0, par->poset_resid, 0, 0, 0, name);
				} else {
					resamp_res = matrix( -n_pos, n_pos, -n_pos, n_pos);
					resampim( res, 1, ncol, 1, nrow,
							resamp_res, -n_pos, n_pos, -n_pos, n_pos,
							resamp_x0, resamp_width, resamp_y0, resamp_width, resamp_angle,
							(int) par->poset_scaling, (int) par->image_rebin);
					wimaspgmsc( resamp_res, -n_pos, n_pos, -n_pos, n_pos,
							0.0, par->poset_resid, 0, 0, 0, name);
					free_matrix( resamp_res, -n_pos, n_pos, -n_pos, n_pos);
				}
			}
			free_matrix( res, 1, ncol, 1, nrow);

		}  /* end of "write" block */

	}  /* end for loop over frames */

	/*  For the "write" and "orbit" actions, close calibration factor file  */

	if ((par->action == WRITE || par->action == ORBIT) && par->list_posetcal)
		fclose( fp);

	return chi2_set;
}

double chi2_lghtcrv( struct par_t *par, struct lghtcrv_t *lghtcrv, int list_breakdown,
		int s, double *chi2_all_lghtcrv)
{
	FILE *fp;
	char name[MAXLEN], weightstring[MAXLEN], dofstring[MAXLEN];
	int i, n, ncalc;
	double chi2_set, err, o2, m2, om, calval, weight, dof, obsmag, sunmag, fitmag, obsmagerr;

	n = lghtcrv->n;
	ncalc = lghtcrv->ncalc;

	/*  Compute contributions to chi-square  */
	o2 = m2 = om = 0.0;
	for (i=1; i<=n; i++) {

		o2 += lghtcrv->obs[i] * lghtcrv->obs[i] * lghtcrv->oneovervar[i];
		m2 += lghtcrv->fit[i] * lghtcrv->fit[i] * lghtcrv->oneovervar[i];
		om += lghtcrv->fit[i] * lghtcrv->obs[i] * lghtcrv->oneovervar[i];
	}
	/* If this lightcurve's calibration factor is allowed to float,
      set it to minimize chi-square, the sum over all points of
      { (obs - calfact*fit)^2 / variance }.                         */

	if (lghtcrv->cal.state == 'f') {
		if (om > 0.0) {
			lghtcrv->cal.val = om/m2;
		} else {
			lghtcrv->cal.val = TINYCALFACT;
			if (par->action != FIT || list_breakdown)
				printf("WARNING: set %2d          had negative calfact reset to %10.4e\n",
						s, lghtcrv->cal.val);
		}
	}

	/*  Compute chi-square for dataset  */

	calval = lghtcrv->cal.val;
	weight = lghtcrv->weight;
	dof = lghtcrv->dof;
	err = weight*(o2 - 2*calval*om + calval*calval*m2);
	chi2_set = err;

	if (list_breakdown)
		*chi2_all_lghtcrv += err;

	/*  For the "write" and "orbit" actions, display chi-square
      and write files with data and fits; the files must be
      written in this routine because they rely on updated
      values for the calibration factors.                      */

	if ((*par).action == WRITE || (*par).action == ORBIT) {

	    /*  Display weight and degrees of freedom as integers if possible  */

	    intifpossible( weightstring, MAXLEN, weight, SMALLVAL, "%f");
	    intifpossible( dofstring, MAXLEN, dof, SMALLVAL, "%f");
	    printf("chi2 of set %2d          is %f    for %s x (%d points) = %s dof",
	           s, err, weightstring, n, dofstring);
	    printf("    (red. chi2 = %.2f)\n", err/dof);
	    fflush(stdout);

	    /*  Output the calculated model lightcurve magnitudes  */

	    sunmag = (*par).sun_appmag[ (*lghtcrv).ioptlaw ];

	    sprintf( name, "calc_%02d.dat", s);
	    FOPEN( fp, name, "w");
	    for (i=1; i<=ncalc; i++) {
	      fitmag = sunmag - 2.5*log10( calval * (*lghtcrv).y[i]);
	      fprintf( fp, "%f %f %10.6f\n", (*lghtcrv).x[i] - (*par).jdoffset,
	               fitmag, (*lghtcrv).rotphase_calc[i]);
	    }
	    fclose( fp);

	    /*  Output the model lightcurve magnitudes at the observation epochs;
	        the corresponding model intensities were obtained (in routine
	        calc_fits) from the calculated model intensities via cubic
	        spline interpolation (and similarly for rotation phases)           */

	    sprintf( name, "fit_%02d.dat", s);
	    FOPEN( fp, name, "w");
	    for (i=1; i<=n; i++) {
	      obsmag = sunmag - 2.5*log10( (*lghtcrv).obs[i]);
	      fitmag = sunmag - 2.5*log10( calval * (*lghtcrv).fit[i]);
	      obsmagerr = 1/(0.4 * LN10 * sqrt((*lghtcrv).oneovervar[i]) * (*lghtcrv).obs[i]);
	      fprintf( fp, "%f %f %f %10.6f %6.4f\n",
	               (*lghtcrv).t[i][(*lghtcrv).v0] - (*par).jdoffset, obsmag, fitmag,
	               (*lghtcrv).rotphase_obs[i], obsmagerr);
	    }
	    fclose( fp);
	  }

	  return chi2_set;
	}
