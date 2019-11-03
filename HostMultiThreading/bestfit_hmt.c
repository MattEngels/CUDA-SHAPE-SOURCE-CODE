/* This version of bestfit involves CPU programming only - no GPUs are involved.
 * Host-multithreading uses the pthreads library to implement multiple CPU
 * threads responsible for different datasets, in much the same was as the old
 * MPI program mode did. */

#include "../shape/head.h"
static double *hotparam;
static struct par_t *spar;
static struct mod_t *smod;
static struct dat_t *sdat;

static int newsize, newshape, newspin, newphoto, newdelcor, newdopscale, newxyoff,
showvals=0, vary_delcor0_size, vary_delcor0_shapespin, vary_dopscale_spin,
vary_dopscale_sizeshape, vary_alb_size, vary_alb_shapespin, vary_hapke,
call_vary_params, check_posbnd, check_badposet, check_badradar;
static double deldop_zmax, deldop_zmax_save, cos_subradarlat, cos_subradarlat_save,
rad_xsec, rad_xsec_save, opt_brightness, opt_brightness_save, baddiam_factor,
badphoto_factor, posbnd_factor, badposet_factor, badradar_factor,
baddopscale_factor;
double objective_hmt(double x, pthread_t *hmt_thread);

double bestfit_hmt(struct par_t *par, struct mod_t *mod, struct dat_t *dat)
{
	char hostname[MAXLEN], dofstring[MAXLEN];
	int iter=0, p, cntr, first_fitpar, b, partype, keep_iterating, ilaw;
	long pid_long;
	pid_t pid;
	double beginerr, enderr=0.0, ax, bx, cx, obja, objb, objc, xmin,
	final_chi2, final_redchi2, dummyval, dummyval2, dummyval3, dummyval4,
	delta_delcor0, dopscale_factor, radalb_factor, optalb_factor;

	/* Get the hostname of host machine and the PID  */
	(void) gethostname(hostname, MAXLEN-1);
	pid = getpid();
	pid_long = (long) pid;  /* Assumes pid_t fits in a long */
	printf("#\n# Single-processor CUDA fit (pid %ld on %s)\n", pid_long, hostname);
	fflush(stdout);

	/* Create the host threads - HMT_threads total but that includes the
	 * currently running thread, so we create one less */
	pthread_t *hmt_thread;
//	hmt_thread = malloc(sizeof(pthread_t)*HMT_threads);
	hmt_thread = malloc(sizeof(pthread_t)*(HMT_threads-1));

	/* Initialize static global pointers used by objective(x) below to be
	 * compatible with "Numerical Recipes in C" routines       */
	spar = par;
	smod = mod;
	sdat = dat;

	printf("\n\n###   HOST MULTI-THREADING MODE (%i THREADS)   ### \n\n\n", HMT_threads);

	/* Initialize static global parameters  */
	newsize = newshape = newspin = newphoto = newdelcor = newdopscale = newxyoff = 1;
	deldop_zmax = deldop_zmax_save = 0.0;
	cos_subradarlat = cos_subradarlat_save = 0.0;
	rad_xsec = rad_xsec_save = 0.0;
	opt_brightness = opt_brightness_save = 0.0;
	vary_delcor0_size = (par->vary_delcor0 != VARY_NONE);
	vary_delcor0_shapespin = (par->vary_delcor0 == VARY_ALL);
	vary_dopscale_spin = (par->vary_dopscale != VARY_NONE);
	vary_dopscale_sizeshape = (par->vary_dopscale == VARY_ALL);
	vary_alb_size = (par->vary_radalb != VARY_NONE || par->vary_optalb != VARY_NONE);
	vary_alb_shapespin = (par->vary_radalb == VARY_ALL || par->vary_optalb == VARY_ALL);
	vary_hapke = 0;
	if (par->vary_optalb != VARY_NONE)
		for (ilaw=0; ilaw<mod->photo.noptlaws; ilaw++)
			if (mod->photo.opttype[ilaw] == HAPKE || mod->photo.opttype[ilaw] == HARMHAPKE
					|| mod->photo.opttype[ilaw] == INHOHAPKE)
				vary_hapke = 1;
	call_vary_params = (par->vary_delcor0 != VARY_NONE || par->vary_dopscale != VARY_NONE
			|| par->vary_radalb != VARY_NONE
			|| par->vary_optalb != VARY_NONE);

	/* Initialize local parameters  */
	delta_delcor0 = 0.0;
	dopscale_factor = radalb_factor = optalb_factor = 1.0;

	/* Compute deldop_zmax_save, cos_subradarlat_save, rad_xsec_save, and
	 * opt_brightness_save for the initial model  */
	if (call_vary_params)
	{
		realize_mod( par, mod);
		realize_spin( par, mod, dat);
		realize_photo( par, mod, 1.0, 1.0, 0);  /* set R_save to R */
		/* realize_delcor and realize_dopscale were called by read_dat */
		vary_params_hmt(par, mod, dat, &deldop_zmax_save, &rad_xsec_save,
				&opt_brightness_save, &cos_subradarlat_save, hmt_thread);
		printf("\n\nvary_params call finished. \n");
	}

	printf("rad_xsec: %f\n", rad_xsec_save);
	printf("deldop_zmax: %f\n", (float)deldop_zmax_save);

	/* Point hotparam to a dummy variable (dummyval) rather than to a model pa-
	 * rameter; then call objective(0.0) to set dummy variable = 0.0, realize
	 * the initial model, calculate the fits, return initial model's objective
	 * function as enderr.                          */
	hotparam = &dummyval;
	enderr = objective_hmt(0.0, hmt_thread);
	printf("#\n# searching for best fit ...\n");
	printf("%4d %8.6f to begin", 0, enderr);

	if (par->baddiam)		printf("  (BAD DIAMS)");
	if (par->badphoto)		printf("  (BAD PHOTO)");
	if (par->posbnd)		printf("  (BAD POS)");
	if (par->badposet)		printf("  (BAD POSET)");
	if (par->badradar)		printf("  (BAD RADAR)");
	if (par->baddopscale)	printf("  (BAD DOPSCALE)");
	printf("\n");
	fflush(stdout);

	/* Display the region within each delay-Doppler or Doppler frame that, ac-
	 * cording to initial model, has nonzero power. A warning is displayed if
	 * any region extends beyond the data limits: the vignetting is too tight,
	 * or else some model parameter (such as a delay correction polynomial co-
	 * efficient) is seriously in error.   */
	show_deldoplim( dat);

	/* Set the starting fit parameter for the first iteration only  */
	first_fitpar = par->first_fitpar;
	if (first_fitpar < 0 || first_fitpar >= par->nfpar) {
		printf("ERROR: need 0 <= first_fitpar < nparams (%d)\n", par->nfpar);
		bailout("bestfit.c\n");
	}

	/* Iteratively adjust model; for each iteration, step through all free pa-
	 * rameters, adjusting one parameter at a time so as to minimize the objec-
	 * tive function at each step. Stop when fractional decrease in the objec-
	 * tive function from one iteration to the next is less than term_prec.   */

	do {
		showvals = 1;        /* show reduced chi-square and penalties at beginning */
		beginerr = enderr;
		printf("# iteration %d %f", ++iter, beginerr);
		if (par->baddiam)			printf("  (BAD DIAMS)");
		if (par->badphoto)			printf("  (BAD PHOTO)");
		if (par->posbnd)			printf("  (BAD POS)");
		if (par->badposet)			printf("  (BAD POSET)");
		if (par->badradar)			printf("  (BAD RADAR)");
		if (par->baddopscale)		printf("  (BAD DOPSCALE)");
		printf("\n");
		if (par->objfunc_start > 0.0 && iter == 1) {
			printf("#\n");
			printf("# (at end of iteration 1 will treat objfunc_start = %8.6f as initial value)\n",
					par->objfunc_start);
		}
		fflush(stdout);

		/*  Show breakdown of chi-square by data type (by 3rd argument = 0)    */
		chi2( par, dat, 1);

		/*  Loop through the free parameters  */
		cntr = first_fitpar % par->npar_update;
		for (p=first_fitpar; p<par->nfpar; p++) {

			/*  Adjust only parameter p on this try  */
			hotparam = par->fpntr[p];
			partype = par->fpartype[p];     /* parameter type */
			newsize = newshape = newspin = newphoto = newdelcor = newdopscale
					= newxyoff = 0;
			if 		(partype == SIZEPAR)		newsize = 1;
			else if (partype == SHAPEPAR)		newshape = 1;
			else if (partype == SPINPAR)		newspin = 1;
			else if (partype == PHOTOPAR)		newphoto = 1;
			else if (partype == DELCORPAR)		newdelcor = 1;
			else if (partype == DOPSCALEPAR)	newdopscale = 1;
			else if (partype == XYOFFPAR)		newxyoff = 1;

			/*  If this is a size parameter AND model extends beyond POS frame
			 * AND the "avoid_badpos" parameter is turned on, shrink model by
			 * 5% at a time until it fits within the POS frame.
			 * We must start with the redundant model evaluation for the un-
			 * changed value of the size parameter, in case the first call to
			 * objective displays reduced chi-square and the penalty functions.  */
			if (par->avoid_badpos && partype == SIZEPAR) {
				while (par->posbnd) {
					objective_hmt(*hotparam, hmt_thread);
					if (par->posbnd)
						*hotparam *= 0.95;
				}
			}

			/* Use Numerical Recipes routine mnbrak to bracket a minimum in the
			 * objective function (reduced chi-square plus penalties) objec-
			 * tive(x), where x is the value of parameter p.  As initial trial
			 * parameter values, use ax (unadjusted value) and bx, that value
			 * incremented by the appropriate step size (length_step,spin_step,
			 * etc.). mnbrak returns 3 parameter values, with bx between ax
			 * and cx; note that ax and bx are changed from their input values.
			 * It also returns the 3 corresponding objective(x) values, where
			 * objb is less than obja and objc.  Hence there is at least one
			 * local minimum (but not necessarily *any* global minimum)
			 * somewhere between ax and cx.          */
			ax = *hotparam;
			bx = ax + par->fparstep[p];
			mnbrak_hmt( &ax, &bx, &cx, &obja, &objb, &objc,
					objective_hmt, hmt_thread);

			/* Before homing in on local minimum, initialize flags that will
			 * tell us if model extended beyond POS frame (sky rendering) for
			 * any trial parameter value(s), if it extended beyond any POS ima-
			 * ges, and if it was too wide in delay-Doppler space         */
			check_posbnd = 0;
			check_badposet = 0;
			check_badradar = 0;

			/* Now use Numerical Recipes function brent to find local minimum -
			 * that is, to find xmin, the best value of x, to within the
			 * *fractional* tolerance specified for parameter p (length_tol,
			 * spin_tol, etc.). brent's return value is the minimized objective
			 * function, objective(xmin). If more than one local minimum bet-
			 * ween ax and cx, brent might not find the best one. brent_abs is
			 * a modified version of brent that has an absolute fitting tole-
			 * rance as one of its arguments, in addition to the existing
			 * fractional tolerance.                                      */
			enderr = brent_abs_hmt( ax, bx, cx, objective_hmt,
					par->fpartol[p], par->fparabstol[p], &xmin, hmt_thread);

			/* Realize whichever part(s) of the model has changed.
			 *
			 * The code here is somewhat opaque because more than one part of
			 * the model may have changed - if the "vary_delcor0" "vary_radalb"
			 * and/or "vary_optalb" parameter is being used to permit joint pa-
			 * rameter adjustments. Before calling the vary_params routine, the
			 * size/shape and spin states must be realized (realize_mod and
			 * realize_spin); if albedos are being varied jointly with other
			 * parameters, the photometric state must also be realized
			 * (realize_photo); and in either case the 0th-order delay correc-
			 * tion polynomial coefficients must be reset to their saved
			 * values via the appropriate call to realize_delcor.          */
			(*hotparam) = xmin;
			if (newsize || newshape)
				realize_mod( par, mod);
			if (newspin)
				realize_spin( par, mod, dat);
			if ((newsize && vary_alb_size) || ((newshape ||
					newspin) && vary_alb_shapespin))
				realize_photo( par, mod, 1.0, 1.0, 1);  /* set R to R_save */
			if ((newsize && vary_delcor0_size) || ((newshape || newspin)
					&& vary_delcor0_shapespin))
				realize_delcor( dat, 0.0, 1);  /* set delcor0 to delcor0_save */
			if ((newspin && vary_dopscale_spin) || ((newsize || newshape)
					&& vary_dopscale_sizeshape))
				realize_dopscale( par, dat, 1.0, 1);  /* set dopscale to dopscale_save */
			if (call_vary_params) {

				/* Call vary_params to get the adjustments to 0th-order delay
				 * correction polynomial coefficients, to Doppler scaling fac-
				 * tors, and to radar and optical albedos                  */
				vary_params_hmt(par, mod, dat, &deldop_zmax, &rad_xsec,
						&opt_brightness, &cos_subradarlat, hmt_thread);
				delta_delcor0 = (deldop_zmax - deldop_zmax_save)*KM2US;
				if (cos_subradarlat != 0.0)
					dopscale_factor = cos_subradarlat_save/cos_subradarlat;
				if (rad_xsec != 0.0)
					radalb_factor = rad_xsec_save/rad_xsec;
				if (opt_brightness != 0.0)
					optalb_factor = opt_brightness_save/opt_brightness;
			}
			if ((newsize && vary_alb_size) || ((newshape || newspin) &&
					vary_alb_shapespin)) {
				realize_photo( par, mod, radalb_factor, optalb_factor, 2);  /* reset R, then R_save */

				/* Must update opt_brightness_save for Hapke optical scattering
				 * law, since single-scattering albedo w isn't just an overall
				 * scaling factor  */
				if (vary_hapke)
					vary_params_hmt(par, mod, dat, &dummyval2, &dummyval3,
							&opt_brightness_save, &dummyval4, hmt_thread);
			} else if (newphoto) {
				rad_xsec_save = rad_xsec;
				opt_brightness_save = opt_brightness;
				realize_photo( par, mod, 1.0, 1.0, 0);  /* set R_save to R */
			}
			if ((newsize && vary_delcor0_size) || ((newshape || newspin) &&
					vary_delcor0_shapespin)) {
				deldop_zmax_save = deldop_zmax;
				realize_delcor( dat, delta_delcor0, 2);  /* reset delcor0, then delcor0_save */
			} else if (newdelcor) {
				realize_delcor( dat, 0.0, 0);  /* set delcor0_save to delcor0 */
			}
			if ((newspin && vary_dopscale_spin) || ((newsize || newshape) &&
					vary_dopscale_sizeshape)) {
				cos_subradarlat_save = cos_subradarlat;
				realize_dopscale( par, dat, dopscale_factor, 2);  /* reset dopscale, then dopscale_save */
			} else if (newdopscale) {
				realize_dopscale( par, dat, 1.0, 0);  /* set dopscale_save to dopscale */
			}
			if (newxyoff)
				realize_xyoff( dat);

			/* If the model extended beyond POS frame (sky rendering) for any
			 * trial parameter value(s), if it extended beyond any plane-of-
			 * sky fit frames, or if it was too wide in delay-Doppler space,
			 * evaluate model for best-fit parameter value to check if these
			 * problems persist - that is, to update "posbnd" "badposet" and
			 * "badradar" parameters for updated model.
			 * (This needn't be done for "baddiam" "badphoto" flags: if we've
			 * just finished adjusting an ellipsoid dimension or photometric
			 * parameter, realize_mod or realize_photo was called in code block
			 * above in order to realize the changed portion of model, and that
			 * call updated corresponding flag. Also we needn't worry about the
			 * "baddopscale" flag, since realize_dopscale was called above if
			 * Doppler scaling factors were changed.) The call to objective
			 * (*hotparam) first sets *hotparam (the parameter that we just
			 * adjusted) equal to itself (i.e., no change) and then calls
			 * calc_fits to evaluate the model for all datasets.          */
			if (check_posbnd || check_badposet || check_badradar)
				objective_hmt(*hotparam, hmt_thread);

			/* Display the objective function after each parameter adjustment.  */
			printf("%4d %8.6f %d", p, enderr, iround(par->fpartype[p]));
			if (par->baddiam)		printf("  (BAD DIAMS)");
			if (par->badphoto)		printf("  (BAD PHOTO)");
			if (par->posbnd)		printf("  (BAD POS)");
			if (par->badposet)		printf("  (BAD POSET)");
			if (par->badradar)		printf("  (BAD RADAR)");
			if (par->baddopscale)	printf("  (BAD DOPSCALE)");
			printf("\n");
			fflush(stdout);

			/* Display reduced chi-square and individual penalty values after
			 * every 20th parameter adjustment. Setting showvals to 1 here
			 * means that these things will be displayed next time objective(x)
			 * is evaluated - at start of NEXT parameter adjustment.  Specifi-
			 * cally, they will be displayed when routine mnbrak evaluates
			 * objective(x) for *unadjusted* parameter value ax (see comment
			 * above).
			 * Also rewrite model and obs files after every 20th parameter
			 * adjustment. Most of obs file doesn't change, but some floating
			 * parameters (i.e. delay correction polynomial coefficients) do.  */
			if (++cntr >= par->npar_update) {
				cntr = 0;
				showvals = 1;
				calc_fits_hmt(par, mod, dat, hmt_thread);
				chi2( par, dat, 0);
				write_mod( par, mod);
				write_dat( par, dat);
			}
		}

		/* End of this iteration: Write model and data to disk, and display the
		 * region within each delay-Doppler or Doppler frame for which model
		 * power is nonzero.                                               */
		if (cntr != 0) {
			calc_fits_hmt(par, mod, dat, hmt_thread);
			chi2( par, dat, 0);
			write_mod( par, mod);
			write_dat( par, dat);
		}
		show_deldoplim( dat);

		/* Check if we should start a new iteration  */
		if (iter == par->term_maxiter) {
			/* Just completed last iteration permitted by "term_maxiter" para-
			 * meter, so stop iterating; note that since iter is 1-based, this
			 * test is always false if "term_maxiter" = 0 (its default value)  */
			keep_iterating = 0;

		} else if (first_fitpar > 0) {
			/* Just completed partial iteration (possible for iteration 1): if
			 * "objfunc_start" parameter was given, check if fractional decrea-
			 * se in objective function *relative to objfunc_start* during the
			 * just-completed iteration was larger than term_prec, thus
			 * justifying a new iteration; if it wasn't specified, definitely
			 * proceed to a new iteration.                            */
			if (par->objfunc_start > 0.0)
				keep_iterating = ((par->objfunc_start - enderr)/enderr >= par->term_prec);
			else
				keep_iterating = 1;
			first_fitpar = 0;     /* for all iterations after the first iteration */

		} else if (par->term_badmodel &&
				(par->posbnd || par->badphoto || par->baddiam
						|| par->badposet || par->badradar
						|| par->baddopscale) ) {

			/* Just completed a full iteration, stop iterating because "term_
			 * badmodel" parameter is turned on and model has a fatal flaw: it
			 * extends beyond POS frame OR it one or more illegal photometric
			 * parameters OR it has one or more tiny or negative ellipsoid dia-
			 * meters OR it has plane-of-sky fit frames too small to "contain"
			 * model OR it is too wide in delay-Doppler space for (delay-)
			 * Doppler fit frames to be correctly constructed OR it has out-of-
			 * range values for one or more Doppler scaling factors    */
			keep_iterating = 0;

		}
		else {
			/* Just completed a full iteration and the model has no fatal flaws
			 * (or else the "term_badmodel" parameter is turned off): keep
			 * iterating if fractional decrease objective function during the
			 * just-completed iteration was greater than term_prec         */
			keep_iterating = ((beginerr - enderr)/enderr >= par->term_prec);
		}

	} while (keep_iterating);

	/* Show final values of reduced chi-square, individual penalty functions,
	 * and the objective function  */
	final_chi2 = chi2( par, dat, 1);
	final_redchi2 = final_chi2/dat->dof;
	printf("# search completed\n");
	if (par->pen.n > 0 || par->baddiam || par->badphoto || par->posbnd
			|| par->badposet || par->badradar
			|| par->baddopscale) {
		printf("#\n");
		printf("# %15s %e\n", "reduced chi2", final_redchi2);
		if (par->pen.n > 0) {
			par->showstate = 1;
			penalties( par, mod, dat);
			par->showstate = 0;
		}
		if (par->baddiam)
			printf("# objective func multiplied by %.1f: illegal ellipsoid diameters\n",
					baddiam_factor);
		if (par->badphoto)
			printf("# objective func multiplied by %.1f: illegal photometric parameters\n",
					badphoto_factor);
		if (par->posbnd)
			printf("# objective func multiplied by %.1f: model extends beyond POS frame\n",
					posbnd_factor);
		if (par->badposet)
			printf("# objective func multiplied by %.1f: "
					"model extends beyond plane-of-sky fit image\n",
					badposet_factor);
		if (par->badradar)
			printf("# objective func multiplied by %.1f: "
					"model is too wide in delay-Doppler space to construct fit image\n",
					badradar_factor);
		if (par->baddopscale)
			printf("# objective func multiplied by %.1f: illegal Doppler scaling factors\n",
					baddopscale_factor);
		printf("# ----------------------------\n");
		printf("# %15s %e\n", "objective func", enderr);
		printf("#\n");
	}
	intifpossible( dofstring, MAXLEN, dat->dof, SMALLVAL, "%f");
	printf("# final chi2 = %e for %s dof (reduced chi2 = %f)\n",
			final_chi2, dofstring, final_redchi2);
	printf("#\n");
	printf("\nIterations total: %i\n", iter);
	printf("CPU fit enderr: %g\n", enderr);
	fflush(stdout);
	free(hmt_thread);
	return enderr;
}


/*  objective(x) is the objective function, with x the value of the one
    model parameter that is being adjusted at the moment by bestfit.
    Other parameters on which objective depends must be placed in static
    variables at the top of this file, for compatibility with Numerical
    Recipes routines mnbrak and brent (which search for minima of a
    function of *one* variable).

    objective(x) also displays reduced chi-square and the individual
    penalty values if bestfit has set showvals = 1.  It then resets
    showvals to 0 after displaying these quantities.                      */

double objective_hmt(double x, pthread_t *hmt_thread)
{
	int b;
	double err, pens, delta_delcor0, dopscale_factor, radalb_factor,
	optalb_factor,branch_posbnd_logfactor, branch_badposet_logfactor,
	branch_badradar_logfactor, branch_baddopscale_logfactor;

	/*  Initialize local parameters  */
	delta_delcor0 = 0.0;
	dopscale_factor = radalb_factor = optalb_factor = 1.0;

	/*  Assign new trial value to the model parameter being adjusted  */
	(*hotparam) = x;

	/* Realize whichever part(s) of the model have changed, then calculate root's
	 * contribution to chi-square.
	 * The code here is somewhat opaque because more than one part of the model
	 * may have changed - if the "vary_delcor0" "vary_dopscale" "vary_radalb" and
	 * /or "vary_optalb" parameter is being used to permit joint parameter ad-
	 * justments. Before calling the vary_params routine, the size/shape and spin
	 * states must be realized (realize_mod and realize_spin); if albedos are
	 * being varied jointly with other parameters, the photometric state must
	 * also be realized (realize_photo); and in either case the 0th-order delay
	 * correction polynomial coefficients and the Doppler scaling factors must be
	 * reset to their saved values via the appropriate calls to realize_delcor
	 * and realize_dopscale, respectively.
	 *
	 * DO NOT set 3rd argument of chi2 (list_breakdown) to 1 here (MPI reasons) */

	if (newsize || newshape)
		realize_mod( spar, smod);
	if (newspin)
		realize_spin( spar, smod, sdat);
	if ((newsize && vary_alb_size) || ((newshape || newspin) && vary_alb_shapespin))
		realize_photo( spar, smod, 1.0, 1.0, 1);  /* set R to R_save */
	if ((newsize && vary_delcor0_size) || ((newshape || newspin) && vary_delcor0_shapespin))
		realize_delcor( sdat, 0.0, 1);  /* set delcor0 to delcor0_save */
	if ((newspin && vary_dopscale_spin) || ((newsize || newshape) && vary_dopscale_sizeshape))
		realize_dopscale( spar, sdat, 1.0, 1);  /* set dopscale to dopscale_save */
	if (call_vary_params) {

		/* Call vary_params to get the trial adjustments to 0th-order delay correc-
		 * tion polynomial coefficients, to Doppler scaling factors,and to radar
		 * and optical albedos, then send them to the branch nodes  */
		vary_params_hmt(spar, smod, sdat, &deldop_zmax, &rad_xsec, &opt_brightness,
				&cos_subradarlat, hmt_thread);

		delta_delcor0 = (deldop_zmax - deldop_zmax_save)*KM2US;
		if (cos_subradarlat != 0.0)
			dopscale_factor = cos_subradarlat_save/cos_subradarlat;
		if (rad_xsec != 0.0)
			radalb_factor = rad_xsec_save/rad_xsec;
		if (opt_brightness != 0.0)
			optalb_factor = opt_brightness_save/opt_brightness;
	}

	if ((newsize && vary_alb_size) || ((newshape || newspin) && vary_alb_shapespin))
		realize_photo( spar, smod, radalb_factor, optalb_factor, 1);  /* adjust R */
	else if (newphoto)
		realize_photo( spar, smod, 1.0, 1.0, 0);  /* set R_save to R */
	if ((newsize && vary_delcor0_size) || ((newshape || newspin) && vary_delcor0_shapespin))
		realize_delcor( sdat, delta_delcor0, 1);  /* adjust delcor0 */
	else if (newdelcor)
		realize_delcor( sdat, 0.0, 0);  /* set delcor0_save to delcor0 */
	if ((newspin && vary_dopscale_spin) || ((newsize || newshape) && vary_dopscale_sizeshape))
		realize_dopscale( spar, sdat, dopscale_factor, 1);  /* adjust dopscale */
	else if (newdopscale)
		realize_dopscale( spar, sdat, 1.0, 0);  /* set dopscale_save to dopscale */
	if (newxyoff)
		realize_xyoff( sdat);

	calc_fits_hmt(spar, smod, sdat, hmt_thread);
	err = chi2( spar, sdat, 0);

	/* Divide chi-square by DOF to get reduced chi-square.    */
	err /= sdat->dof;

	/* If bestfit has set showvals = 1, display reduced chi-square. Then set
	 * spar->showstate = 1, so that when function penalties is called later,
	 * it "knows" that it should display the individual penalty values.
	 * Reset showstate to 0 if showvals = 0.  */
	if (showvals) {
		printf("# %15s %e\n", "reduced chi2", err);
		spar->showstate = 1;
	}
	else
		spar->showstate = 0;

	/* Compute penalties and add to reduced chi-square. Individual penalty values
	 * will be displayed if we set spar->showstate = 1 a few lines back.        */
	pens = penalties( spar, smod, sdat);
	err += pens;

	/* Double the objective function if there's an ellipsoid component with tiny
	 * or negative diameter, if any optical photometric parameters have invalid
	 * values, if any portion of the model lies outside specified POS window or
	 * outside any plane-of-sky fit image, or if model is too wide in delay-Dopp-
	 * ler space for any (delay-)Doppler fit image to be correctly constructed.
	 * This effectively rules out any models with any of these flaws.         */
	if (spar->baddiam) {
		baddiam_factor = spar->bad_objfactor * exp(spar->baddiam_logfactor);
		err *= baddiam_factor;
		if (showvals)
			printf("# objective func multiplied by %.1f: illegal ellipsoid diameters\n",
					baddiam_factor);
	}
	if (spar->badphoto) {
		badphoto_factor = spar->bad_objfactor * exp(spar->badphoto_logfactor);
		err *= badphoto_factor;
		if (showvals)
			printf("# objective func multiplied by %.1f: illegal photometric parameters\n",
					badphoto_factor);
	}
	if (spar->posbnd) {
		check_posbnd = 1;     /* tells bestfit about this problem */
		posbnd_factor = spar->bad_objfactor * exp(spar->posbnd_logfactor);
		err *= posbnd_factor;
		if (showvals)
			printf("# objective func multiplied by %.1f: model extends beyond POS frame\n",
					posbnd_factor);
	}
	if (spar->badposet) {
		check_badposet = 1;     /* tells bestfit about this problem */
		badposet_factor = spar->bad_objfactor * exp(spar->badposet_logfactor);
		err *= badposet_factor;
		if (showvals)
			printf("# objective func multiplied by %.1f: plane-of-sky fit frame too small\n",
					badposet_factor);
	}
	if (spar->badradar) {
		check_badradar = 1;     /* tells bestfit about this problem */
		badradar_factor = spar->bad_objfactor * exp(spar->badradar_logfactor);
		err *= badradar_factor;
		if (showvals)
			printf("# objective func multiplied by %.1f: model too wide in delay-Doppler space\n",
					badradar_factor);
	}
	if (spar->baddopscale) {
		baddopscale_factor = spar->bad_objfactor * exp(spar->baddopscale_logfactor);
		err *= baddopscale_factor;
		if (showvals)
			printf("# objective func multiplied by %.1f: illegal Doppler scaling factors\n",
					baddopscale_factor);
	}

	/* Reset showvals to 0 if it had been 1 (i.e., turn off display of reduced
	 * chi-square and the individual penalty values).  */
	if (showvals)
		fflush( stdout);
	showvals = 0;
	return err;
}
