/*****************************************************************************************
                                                                                bestfit.c
Iterate over all floating parameters, at each step adjusting just one parameter x in order
to minimize objective(x), the objective function (reduced chi-square plus penalties).
Continue until the fractional reduction in objective(x) due to a full pass through the
parameter list is less than term_prec.  Return the final value of the objective function.
__________________________________________________________________________________________
Modified 2016 July 7 by Matt Engels:
	Adapted for use in shape-cuda.
------------------------------------------------------------------------------------------
Modified 2014 February 19 by CM:
    Allow for multiple optical scattering laws when setting the "vary_hapke" flag
Modified 2013 July 14 by CM:
    Implement the "term_maxiter" parameter
Modified 2012 July 5 by MCN and CM:
    Use the gethostname function rather than the HOST environment variable to get root's
        hostname
    List root's PID in addition to the hostname
    List the PID for each branch node, not just the hostname
Modified 2012 June 13 by CM:
    Implement "objfunc_start" parameter
Modified 2012 March 23 by CM:
    Implement Doppler scaling -- more particularly, simultaneous adjustment of shape/spin
        parameters and Doppler scale factors via the "vary_dopscale" parameter
Modified 2010 April 12 by CM:
    Bug fix: When fitting a size, shape, or spin parameter with the
         "vary_delcor0" parameter being used, call realize_delcor to reset
         the 0th-order delay correction polynomial coefficients to their
         saved values before calling vary_params.  (For infinitely fine
         model resolution and delay-Doppler resolution this wouldn't
         matter but in practice it does.)
Modified 2009 November 15 by CM:
    Fix printf statement with too many arguments
Modified 2009 July 5 by CM:
    Add "npar_update" parameter rather than hard-wiring an update (rewrite
        mod and obs files and display reduced chi2 and penalty functions)
        every 20th parameter adjustment
Modified 2009 April 3 by CM:
    If the model has illegal properties (e.g., negative ellipsoid diameters)
        then, for each type of problem, multiply the objective function not
        only by the "bad_objfactor" parameter but also by an additional
        factor that increases as the problem gets worse.  The
        "baddiam_logfactor" "badphoto_logfactor" "posbnd_logfactor"
        "badposet_logfactor" and "badradar_logfactor" parameters are the
        logarithms of the additional factors for the five possible problem
        types; the calc_fits routine computes the logarithms rather than the
        factors themselves so as to avoid floating-point overflow.
    Revise MPI_CALC so that root receives the "posbnd_logfactor" parameter
        from each branch node rather than the "posbnd" parameter:
        posbnd_logfactor > 0.0 if the model extends beyond the POS frame
        for any of the branch node's datasets.  If root sees that this
        value is > 0.0, it will set its "posbnd" flag and will increase the
        objective function accordingly.
    Revise MPI_CALC so that root receives the "badposet_logfactor"
        parameter from each branch node: badposet_logfactor > 0.0 if the
        model extends beyond the fit frame for any of the branch node's
        plane-of-sky datasets.  If root sees that this value is > 0.0, it
        will set its "badposet" flag and will increase the objective
        function accordingly.
    Revise MPI_CALC so that root receives the "badradar_logfactor"
        parameter from each branch node: badradar_logfactor > 0.0 if the
        model is too wide in delay-Doppler space for the program to
        construct some or all (delay-)Doppler fit frames.  If root sees
        that this value is > 0.0, it will set its "badradar" flag and will
        increase the objective function accordingly.
    For MPI_Recv calls, mpi_par[0] is no longer equal to the MPI action,
        since the message tag argument already serves that purpose (as of
        2008 April 10) -- so the other mpi_par elements are renumbered
Modified 2008 August 10 by CM:
    Never terminate the fit at the end of a partial iteration -- that is,
        after the first iteration of a fit where first_fitpar > 0
Modified 2008 July 11 by CM:
    Display the hostname even for single-processor fits
Modified 2008 April 10 by CM:
    For parallel-processing fits, display the hostname for each node
    Use message tag argument to MPI_Recv to identify the MPI action
Modified 2007 August 29 by CM:
    Implement the "avoid_badpos" parameter: if this parameter is turned on
        and the model extends beyond the POS frame and it is time to fit a
        size parameter, start by shrinking that size parameter until the
        model fits within the POS frame
    Implement the "bad_objfactor" parameter in routine objective: multiply
        the objective function by this factor for illegal photometric
        parameters, for tiny or negative ellipsoid diameters, and for
        models that extend beyond the plane-of-sky frame.  (Previously
        this factor was fixed at 2.0.)
    Rename MPI_TAG to MPI_TAG_1 to avoid name conflict with mpich headers
Modified 2007 August 16 by CM:
    Implement the "term_badmodel" parameter: If this parameter is turned on
        and, at the end of any fit iteration, the model ever extends beyond
        the POS frame OR has any illegal photometric parameters OR has any
        tiny or negative ellipsoid diameters, the fit is terminated.
Modified 2007 August 10 by CM:
    Eliminate unused variables
Modified 2006 December 20 by CM:
    Revise MPI_CALC so that root receives the "posbnd" parameter from each
        branch node, so that the objective function can be doubled if the
        model extends beyond the plane-of-sky frame for any datasets
    If the model extends beyond the plane-of-sky frame for any trial value
        of a parameter, evaluate the model for the best-fit parameter value
        to check whether or not it extends beyond the POS frame
Modified 2006 October 1 by CM:
    Add two new arguments to realize_delcor
    Add three new arguments to realize_photo
    Implement "vary_delcor0" "vary_radalb" and "vary_optalb" parameters
    Implement SIZEPAR parameters via the "newsize" variable
Modified 2005 June 27 by CM:
    Renamed "round" function to "iround" to avoid conflict
Modified 2005 March 17 by CM:
    For parallel processing, check that root is receiving the responses
        to the correct broadcast
    Root no longer needs to compute degrees of freedom or to receive
        dof values from branch nodes: Now they are computed in read_dat
    Degrees of freedom can now be floating-point rather than integer
Modified 2005 February 28 by CM:
    Add screen warnings if objective function has been doubled due to
        (a) tiny or negative ellipsoid diameters
        (b) illegal photometric parameters
        (c) model extending beyond the model POS frame
    Initialize the three parameters (baddiam, badphoto, posbnd) that
        flag these three problems in other routines (realize_mod,
        realize_photo, calc_fits) rather than in objective(x), so that
        these three parameters can be used for actions other than "fit"
    Rename DATAPAR to be DELCORPAR
    Add XYOFFPAR and implement the new realize_xyoff routine
Modified 2005 February 22 by CM:
    Move branch nodes' signoff statements from shape.c to here, so that
        they can appear in order
Modified 2005 February 13 by CM:
    Rename objective function "f(x)" to be "objective(x)"
    Only broadcast to branch nodes if there are any branch nodes
        (i.e., if mpi_nproc > 1)
    Broadcast the new MPI_DUMMYPAR signal to branch nodes before evaluating
        objective(0.0), the objective function for the existing model;
        this tells each branch node to point hotparam to a dummy variable
        rather than to a model parameter, so that the dummy variable will
        be set to 0.0 and the model will be unchanged.
    Broadcast the new MPI_CALFACT signal to branch nodes to get updated
        calibration factors before rewriting the obs file
    Root now realizes the model after setting a parameter to its best value
    Make sure that root and branch nodes update the model (i.e., that they
        call the calc_fits and chi2 routines) before rewriting the mod and
        obs files and before calling routine show_deldoplim
    Avoid unnecessary model realizations for root by allowing newshape,
        newspin, newphoto, and newdelcor to be 0, not always 1 as before
    Move MPI_DONE broadcast to here from shape.c
Modified 2005 January 25 by CM:
    Eliminated unused variable
Modified 2005 January 10 by CM:
    When fitting using parallel processing, ping all of the branch nodes
        and inform the user that they're active
Modified 2004 October 29 by CM:
    Add "first_fitpar" parameter so that a fit can be started (or resumed)
        at some parameter (counting from 0) other than the first parameter
Modified 2004 October 10 by CM:
    Fix chi-square display at start of each iteration and at the
        end of the fit by calling realize_mod, realize_spin, realize_photo,
        realize_delcor, and calc_fits before calling chi2
Modified 2004 August 13 by CM:
    Call modified minimum search routine brent_abs rather than brent
        so that absolute fitting tolerances can be specified
Modified 2004 May 21 by CM:
    Display the final values of the individual penalty functions
Modified 2004 April 3 by CM:
    Add the "list_breakdown" argument to routine chi2 so that we can
        display the chi2 breakdown by data type (Doppler, delay-Doppler,
        POS, lightcurves) at the start of each fit iteration and at
        the end of the fit
Modified 2004 February 26 by CM:
    realize_photo now takes two arguments rather than one
Modified 2003 April 26 by CM:
    Added "show_deldoplim" call at the end of each fit iteration,
        to check for overly tight data vignetting
Modified 2003 April 23 by CM:
    Implemented '=' state for delay correction polynomial coefficients
        via the "realize_delcor" routine
Modified 2003 April 17 by CM:
    Added "baddiam" parameter to function f so that the objective
        function is doubled if an ellipsoid component has a tiny or
        negative diameter
Modified 2003 April 2 by CM:
    In function f (which computes reduced-chi-squared-plus-penalties),
        moved call to "penalties" from before spar->showstate is set
        to after.
    Values of reduced chi-squared and of the various penalties are
        printed to the screen after every 20th parameter adjustment.
        To be precise, they're printed at the very first call to f when
        adjusting parameter 21, 41, 61, etc.  This call is made within
        function bestfit by minimum-bracketing function mnbrak;
        it corresponds to the *unadjusted* value of parameter 21 (or 41
        or ...), which is what we want.
    Until now, the individual penalty values were being printed on the
 *second* call to f, also made by mnbrak but with parameter 21
        incremented by the relevant initial step size (e.g., length_step).
        Hence these printed values were irrelevant and misleadingly large.
        Moving the call to "penalties" later in the code fixes the problem.
 *****************************************************************************************/
extern "C" {
#include "../shape/head.h"
}
//static __device__ double *hotparam;
static __device__ int dmax_frames;
static struct par_t *spar, *sdev_par, *sdev_par1;
static struct mod_t *smod, *smod1, *sdev_mod, *sdev_mod1;
static struct dat_t *sdat, *sdev_dat, *sdev_dat1;

static int newsize, newshape, newspin, newphoto, newdelcor, newdopscale, newxyoff,
showvals=0, vary_delcor0_size, vary_delcor0_shapespin, vary_dopscale_spin,
vary_dopscale_sizeshape, vary_alb_size, vary_alb_shapespin, vary_hapke,
call_vary_params, check_posbnd, check_badposet, check_badradar;
static double deldop_zmax, deldop_zmax_save, cos_subradarlat, cos_subradarlat_save,
rad_xsec, rad_xsec_save, opt_brightness, opt_brightness_save, baddiam_factor,
badphoto_factor, posbnd_factor, badposet_factor, badradar_factor,
baddopscale_factor;
static unsigned char type;
static double hotparamval;

__host__ double objective_gpu(double x, struct vertices_t **verts,
		unsigned char *htype, unsigned char *dtype, int *nframes, int *nviews,
		int *lc_n, int nsets, int nf, cudaStream_t *bf_stream,
		double **fit_overflow);

__device__ double bf_hotparamval, bf_dummyval=0.0, *hotparam;
__device__ int bf_partype;

__global__ void bf_get_flags_krnl(struct par_t *dpar, int *flags) {
	/* Single-threaded kernel */
	if (threadIdx.x == 0) {
		flags[0] = (int)dpar->baddiam;
		flags[1] = (int)dpar->badphoto;
		flags[2] = (int)dpar->posbnd;
		flags[3] = (int)dpar->badposet;
		flags[4] = (int)dpar->badradar;
		flags[5] = (int)dpar->baddopscale;
	}
}
__global__ void ocs_get_flags_krnl(struct par_t *dpar, int *flags,
		double *dlogfactors) {
	/* Single-threaded kernel */
	if (threadIdx.x == 0) {
		flags[0] = (int)dpar->baddiam;
		flags[1] = (int)dpar->badphoto;
		flags[2] = (int)dpar->posbnd;
		flags[3] = (int)dpar->badposet;
		flags[4] = (int)dpar->badradar;
		flags[5] = (int)dpar->baddopscale;

		dlogfactors[0] = dpar->bad_objfactor;
		dlogfactors[1] = dpar->baddiam_logfactor;
		dlogfactors[2] = dpar->badphoto_logfactor;
		dlogfactors[3] = dpar->posbnd_logfactor;
		dlogfactors[4] = dpar->badposet_logfactor;
		dlogfactors[5] = dpar->badradar_logfactor;
		dlogfactors[6] = dpar->baddopscale_logfactor;
	}
}
__global__ void bf_set_hotparam_initial_krnl() {
	/* Single-threaded kernel */
	if (threadIdx.x == 0)
		hotparam = &bf_dummyval;
}
__global__ void bf_set_hotparam_pntr_krnl(double **fpntr,
		int *fpartype, int p) {
	/* Single-threaded kernel */
	if (threadIdx.x == 0) {
		hotparam = fpntr[p];	/* This is pointing at a device variable */
		bf_partype = fpartype[p];  /* parameter type */
	}
}
__global__ void bf_get_hotparam_val_krnl() {
	/* Single threaded kernel */
	if (threadIdx.x == 0) {
		bf_hotparamval = *hotparam;
	}
}
__global__ void bf_mult_hotparam_val_krnl(double factor) {
	/* Single-threaded kernel */
	if (threadIdx.x == 0)
		*hotparam *= factor;
}
__global__ void bf_set_hotparam_val_krnl(double newvalue) {
	/* Single-threaded kernel */
	if (threadIdx.x == 0) {
		*hotparam = newvalue;
		bf_hotparamval = newvalue;
	}
}
__global__ void set_verts_shortcut_krnl(struct mod_t *dmod,
		struct vertices_t **verts, int max_frames) {
	/* Single-threaded kernel */
	if (threadIdx.x == 0) {
		verts[0] = &dmod->shape.comp[0].real;
		dmax_frames = max_frames;
	}
}

__host__ double bestfit_gpu(struct par_t *dpar, struct mod_t *dmod,
		struct dat_t *ddat, struct par_t *par, struct mod_t *mod,
		struct dat_t *dat)
{
	char hostname[MAXLEN], dofstring[MAXLEN];
	int i, iter=0, p, cntr, first_fitpar, partype, keep_iterating=1, ilaw,
			nf, term_maxiter;
	long pid_long;
	pid_t pid;
	double beginerr, enderr, ax, bx, cx, obja, objb, objc, xmin, final_chi2,
		final_redchi2, dummyval2, dummyval3, dummyval4, delta_delcor0,
		dopscale_factor, radalb_factor, optalb_factor, *hfparstep, *hfpartol,
		*hfparabstol, objfunc_start, term_prec;
	unsigned char *htype, *dtype, action, avoid_badpos, term_badmodel;
	int nsets, *nframes, *lc_n, *nviews, nfpar, *hfpartype, npar_update, max_frames=0,
			max_streams=0, *flags, *hflags, delay_flag=0;
	struct vertices_t **verts;
	dim3 THD;

	gpuErrchk(cudaSetDevice(GPU0));
	/* This section collects parameters used for CUDA kernel launches throughout
	 * the program.  The cudaStreams created here are used/re-used for the
	 * lifetime of one program run */
	nsets = dat->nsets;
	nfpar = par->nfpar;
	nf = mod->shape.comp[0].real.nf;
	action = par->action;
	npar_update = par->npar_update;
	avoid_badpos = par->avoid_badpos;
	objfunc_start = par->objfunc_start;
	term_prec = par->term_prec;
	term_badmodel = par->term_badmodel;
	type = mod->shape.comp[0].type;
	htype 	= (unsigned char *) malloc(nsets*sizeof(unsigned char));
	nframes = (int *) malloc(nsets*sizeof(int));
	lc_n	= (int *) malloc(nsets*sizeof(int));
	nviews 	= (int *) malloc(nsets*sizeof(int));
	gpuErrchk(cudaMalloc((void**)&dtype, sizeof(unsigned char)*nsets));
	gpuErrchk(cudaMalloc((void**)&verts, sizeof(struct vertices_t*)*2));

	for (int s=0; s<nsets; s++) {
		htype[s] = dat->set[s].type;
		switch (htype[s]) {
		case DELAY:
			nframes[s] = dat->set[s].desc.deldop.nframes;
			nviews[s]  = dat->set[s].desc.deldop.nviews;
			lc_n[s]    = 0;
			delay_flag = 1;
			break;
		case DOPPLER:
			nframes[s] = dat->set[s].desc.doppler.nframes;
			nviews[s]  = dat->set[s].desc.doppler.nviews;
			lc_n[s]    = 0;
			break;
		case POS:
			nframes[s] = dat->set[s].desc.poset.nframes;
			nviews[s]  = dat->set[s].desc.poset.nviews;
			lc_n[s]    = 0;
			break;
		case LGHTCRV:
			nframes[s] = dat->set[s].desc.lghtcrv.ncalc;
			nviews[s]  = dat->set[s].desc.lghtcrv.nviews;
			lc_n[s]    = dat->set[s].desc.lghtcrv.n;
			break;
		}
		if (nframes[s]>max_frames)	max_frames = nframes[s];
	}
	gpuErrchk(cudaMemcpy(dtype, htype, sizeof(unsigned char)*nsets,
			cudaMemcpyHostToDevice));

	/* The following check is necessary to ensure the dVdIdCOM reduction has
	 * enough streams to operate properly	 */
	if (max_frames < 13) max_streams = 13;
	else	max_streams = max_frames;

	/* Create streams for gpu0 (the only gpu in single-GPU mode)
	 * Also allocate the fit overflow arrays - since we process streamed frames for one set
	 * at a time, we can get away with max_streams multiples and re-use them set after set */
	cudaStream_t bf_stream[max_streams];
	double **fit_overflow;
	if (delay_flag)
		cudaCalloc1((void**)&fit_overflow, sizeof(double**), max_streams);

	for (int f=0; f<max_streams; f++) {
		gpuErrchk(cudaStreamCreate(&bf_stream[f]));

		/* Allocate double pointers (outer loop) and offset addressing */
		if (delay_flag)
			cudaCalloc1((void**)&fit_overflow[f], sizeof(double), MAXOVERFLOW*MAXOVERFLOW);
	}

	/*..........................End section..................................*/

	/* Get the hostname of host machine and the PID */
	(void) gethostname(hostname, MAXLEN-1);
	pid = getpid();
	pid_long = (long) pid;  /* Assumes pid_t fits in a long */
	printf("#\n# Regular GPU CUDA fit (pid %ld on %s)\n", pid_long, hostname);
	fflush(stdout);

	/* Allocate memory for pointers, steps, and tolerances on bothhost and
	 * device. fpntr remains a cudaMallocManaged allocation because it is a
	 * double pointer.  */
	cudaCalloc1((void**)&sdev_par, sizeof(struct par_t), 1);
	gpuErrchk(cudaMemcpy(sdev_par, &par, sizeof(struct par_t), cudaMemcpyHostToDevice));
	cudaCalloc1((void**)&sdev_mod, sizeof(struct mod_t), 1);
	gpuErrchk(cudaMemcpy(sdev_mod, &mod, sizeof(struct mod_t), cudaMemcpyHostToDevice));
	cudaCalloc1((void**)&sdev_dat, sizeof(struct dat_t), 1);
	gpuErrchk(cudaMemcpy(sdev_dat, &dat, sizeof(struct dat_t), cudaMemcpyHostToDevice));
	cudaCalloc1((void**)&flags, sizeof(int), 7);
	cudaCalloc1((void**)&fparstep, sizeof(double), nfpar);
	cudaCalloc1((void**)&fpartol, sizeof(double), nfpar);
	cudaCalloc1((void**)&fparabstol, sizeof(double), nfpar);
	cudaCalloc1((void**)&fpartype, sizeof(int), nfpar);
	cudaCalloc1((void**)&fpntr, sizeof(double*), nfpar);
	hfparstep 	 = (double *) malloc(nfpar*sizeof(double));
	hfpartol	 = (double *) malloc(nfpar*sizeof(double));
	hfparabstol  = (double *) malloc(nfpar*sizeof(double));
	hfpartype 	 = (int *) 	  malloc(nfpar*sizeof(int));
	hflags 		 = (int *) malloc(7*sizeof(int));

	for (i=0; i<nfpar; i++)
		cudaCalloc1((void**)&fpntr[i], sizeof(double), 1);

	/* Set vertices shortcut and also set max_frames (the maximum number of
	 * frames for any one set) to device so that objective_gpu
	 * can retrieve it later */
	set_verts_shortcut_krnl<<<1,1>>>(dmod, verts, max_frames);
	checkErrorAfterKernelLaunch("set_verts_shortcut_krnl");

	/* Initialize static global pointers used by objective(x) below
      to be compatible with "Numerical Recipes in C" routines       */
	spar = par;			smod = mod;			sdat = dat;
	sdev_par = dpar;	sdev_mod = dmod;	sdev_dat = ddat;

	/*  Initialize static global parameters  */
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

	/*  Initialize local parameters  */
	delta_delcor0 = 0.0;
	dopscale_factor = radalb_factor = optalb_factor = 1.0;

	/* The following call sets up the parameter lists allocated above and copy
	 * the device contents to host copies */
	mkparlist_gpu(dpar, dmod,	ddat, fparstep, fpartol, fparabstol, fpartype,
			fpntr, nfpar, nsets);
	gpuErrchk(cudaMemcpy(hfparstep, fparstep, sizeof(double)*nfpar, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(hfpartol, fpartol, sizeof(double)*nfpar, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(hfparabstol, fparabstol, sizeof(double)*nfpar, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(hfpartype,	fpartype, sizeof(int)*nfpar, cudaMemcpyDeviceToHost));

	/* Compute deldop_zmax_save, cos_subradarlat_save, rad_xsec_save, and
	 * opt_brightness_save for the initial model  */
	if (call_vary_params)
	{
		realize_mod_gpu(dpar, dmod, type, nf, bf_stream);
		realize_spin_gpu(dpar, dmod, ddat, htype, nframes, nviews,
				nsets, bf_stream);
		realize_photo_gpu(dpar, dmod, 1.0, 1.0, 0, nf);  /* set R_save to R */
		vary_params_gpu(dpar, dmod, ddat, action, &deldop_zmax_save,
				&rad_xsec_save, &opt_brightness_save, &cos_subradarlat_save,
				nframes, lc_n, nviews, verts, htype, dtype, nf, nsets,
				bf_stream, max_frames, fit_overflow);
	}
	printf("rad_xsec: %f\n", rad_xsec_save);
	printf("deldop_zmax: %f\n", deldop_zmax_save);

	/* Point hotparam to a dummy variable (dummyval) rather than to a model pa-
	 * rameter; then call objective(0.0) to set dummy variable = 0.0, realize
	 * the initial model, calculate the fits, return initial model's objective
	 * function as enderr.                          */
	bf_set_hotparam_initial_krnl<<<1,1>>>();
	checkErrorAfterKernelLaunch("bf_set_hotparam_initial_krnl");

	enderr = objective_gpu(0.0, verts, htype, dtype, nframes,
				nviews, lc_n, nsets, nf, bf_stream, fit_overflow);

	printf("#\n# searching for best fit ...\n");
	printf("%4d %8.6f to begin", 0, enderr);

	/* Launch single-thread kernel to retrieve flags in dev_par */
	/*		flags[0] = dpar->baddiam;
			flags[1] = dpar->badphoto;
			flags[2] = dpar->posbnd;
			flags[3] = dpar->badposet;
			flags[4] = dpar->badradar;
			flags[5] = dpar->baddopscale;*/

	bf_get_flags_krnl<<<1,1>>>(dpar, flags);
	checkErrorAfterKernelLaunch("bf_get_flags_krnl");
	gpuErrchk(cudaMemcpy(hflags, flags, sizeof(int)*7,
			cudaMemcpyDeviceToHost));

	/* Now act on the flags just retrieved from dev_par */
	if (hflags[0]==1)		printf("  (BAD DIAMS)");
	if (hflags[1]==1)		printf("  (BAD PHOTO)");
	if (hflags[2]==1)		printf("  (BAD POS)");
	if (hflags[3]==1)		printf("  (BAD POSET)");
	if (hflags[4]==1)		printf("  (BAD RADAR)");
	if (hflags[5]==1)		printf("  (BAD DOPSCALE)");		printf("\n");
	fflush(stdout);

	/* Display the region within each delay-Doppler or Doppler frame that, ac-
	 * cording to initial model, has nonzero power. A warning is displayed if
	 * any region extends beyond the data limits: the vignetting is too tight,
	 * or else some model parameter (such as a delay correction polynomial co-
	 * efficient) is seriously in error.   */
	show_deldoplim_gpu(ddat, htype, nsets, nframes, max_frames);

	/* Set the starting fit parameter for the first iteration only  */
	first_fitpar = par->first_fitpar;
	term_maxiter = par->term_maxiter;
	if (first_fitpar < 0 || first_fitpar >= nfpar) {
		printf("ERROR: need 0 <= first_fitpar < nparams (%d)\n", nfpar);
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

		/* Launch single-thread kernel to retrieve flags in dev_par */
		bf_get_flags_krnl<<<1,1>>>(dpar, flags);
		checkErrorAfterKernelLaunch("bf_get_flags_krnl");
		gpuErrchk(cudaMemcpy(hflags, flags, sizeof(int)*7,
				cudaMemcpyDeviceToHost));

		/* Now act on the flags just retrieved from dev_par */
		if (hflags[0]==1)		printf("  (BAD DIAMS)");
		if (hflags[1]==1)		printf("  (BAD PHOTO)");
		if (hflags[2]==1)		printf("  (BAD POS)");
		if (hflags[3]==1)		printf("  (BAD POSET)");
		if (hflags[4]==1)		printf("  (BAD RADAR)");
		if (hflags[5]==1)		printf("  (BAD DOPSCALE)");		printf("\n");
		fflush(stdout);

		/* Show breakdown of chi-square by data type    */
		chi2_gpu(dpar, ddat, htype, dtype, nframes, lc_n, 1,
				nsets, bf_stream, max_frames);

		/*  Loop through the free parameters  */
		cntr = first_fitpar % npar_update;
		for (p=first_fitpar; p<nfpar; p++) {
			/*  Adjust only parameter p on this try  */
			bf_set_hotparam_pntr_krnl<<<1,1>>>(fpntr, fpartype, p);
			checkErrorAfterKernelLaunch("bf_set_hotparam_pntr_krnl");
			gpuErrchk(cudaMemcpyFromSymbol(&partype, bf_partype, sizeof(int),
					0, cudaMemcpyDeviceToHost));

			newsize = newshape = newspin = newphoto = newdelcor = newdopscale
					= newxyoff = 0;
			if 		(partype == SIZEPAR)		newsize	 	= 1;
			else if (partype == SHAPEPAR)		newshape 	= 1;
			else if (partype == SPINPAR)		newspin 	= 1;
			else if (partype == PHOTOPAR)		newphoto 	= 1;
			else if (partype == DELCORPAR)		newdelcor 	= 1;
			else if (partype == DOPSCALEPAR)	newdopscale	= 1;
			else if (partype == XYOFFPAR)		newxyoff 	= 1;

			/* If this is a size parameter AND model extends beyond POS frame
			 * AND the "avoid_badpos" parameter is turned on, shrink model by
			 * 5% at a time until it fits within the POS frame.
			 * We must start with the redundant model evaluation for the un-
			 * changed value of the size parameter, in case the first call to
			 * objective displays reduced chi-square and the penalty functions.  */
			if (avoid_badpos && partype == SIZEPAR) {
				bf_get_flags_krnl<<<1,1>>>(dpar, flags);
				checkErrorAfterKernelLaunch("bf_get_flags_krnl");
				gpuErrchk(cudaMemcpy(hflags, flags, sizeof(int)*7,
						cudaMemcpyDeviceToHost));

				/* Get value of (*hotparam) */
				bf_get_hotparam_val_krnl<<<1,1>>>();
				checkErrorAfterKernelLaunch("bf_get_hotparam_val_krnl");
				gpuErrchk(cudaMemcpyFromSymbol(&hotparamval, bf_hotparamval,
						sizeof(double),	0, cudaMemcpyDeviceToHost));

				while (hflags[2]==1) {
					objective_gpu(hotparamval, verts, htype, dtype,
							nframes, nviews, lc_n, nsets, nf, bf_stream, fit_overflow);

					bf_get_flags_krnl<<<1,1>>>(dpar, flags);
					checkErrorAfterKernelLaunch("bf_get_flags_krnl");
					gpuErrchk(cudaMemcpy(hflags, flags, sizeof(int)*7,
							cudaMemcpyDeviceToHost));

					if (hflags[2]==1) {
						/* Set the value pointed to by hotparam to 0.95 of its
						 * previous value */
						bf_mult_hotparam_val_krnl<<<1,1>>>(0.95);
						checkErrorAfterKernelLaunch("bf_mult_hotparam_val_krnl");
					}
				}
			}
			/* Get value of (*hotparam) so that mnbrak can use it*/
			bf_get_hotparam_val_krnl<<<1,1>>>();
			checkErrorAfterKernelLaunch("bf_get_hotparam_val_krnl");

			gpuErrchk(cudaMemcpyFromSymbol(&hotparamval, bf_hotparamval,
					sizeof(double),	0, cudaMemcpyDeviceToHost));

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
			ax = hotparamval;
			bx = ax + hfparstep[p]; /* par usage us fine here */
			mnbrak_gpu(&ax, &bx, &cx, &obja, &objb, &objc,
					objective_gpu, verts, htype, dtype, nframes,
					nviews, lc_n, nsets, nf, bf_stream, fit_overflow);

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
			enderr = brent_abs_gpu(ax, bx, cx, objective_gpu, hfpartol[p],
					hfparabstol[p], &xmin, verts, htype, dtype, nframes, nviews, lc_n,
					nsets, nf, bf_stream, fit_overflow);

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
			/* Set the value pointed to by hotparam to 0.95 of its
			 * previous value (*hotparam) = xmin; */
			bf_set_hotparam_val_krnl<<<1,1>>>(xmin);
			checkErrorAfterKernelLaunch("bf_set_hotparam_val_krnl");
			gpuErrchk(cudaMemcpyFromSymbol(&hotparamval, bf_hotparamval,
					sizeof(double),	0, cudaMemcpyDeviceToHost));

			if (newsize || newshape)
				realize_mod_gpu(dpar, dmod, type, nf, bf_stream);
			if (newspin) {
				realize_spin_gpu(dpar, dmod, ddat, htype, nframes,
						nviews, nsets, bf_stream);
			}
			if ((newsize && vary_alb_size) || ((newshape ||
					newspin) && vary_alb_shapespin))
				realize_photo_gpu(dpar, dmod, 1.0, 1.0, 1, nf);  /* set R to R_save */
			if ((newsize && vary_delcor0_size) || ((newshape || newspin)
					&& vary_delcor0_shapespin)) {
				realize_delcor_gpu(ddat, 0.0, 1, nsets, nframes);  /* set delcor0 to delcor0_save */
			}
			if ((newspin && vary_dopscale_spin) || ((newsize || newshape)
					&& vary_dopscale_sizeshape))
				realize_dopscale_gpu(dpar, ddat, 1.0, 1, nsets, dtype);  /* set dopscale to dopscale_save */
			if (call_vary_params) {
				/* Call vary_params to get the adjustments to 0th-order delay
				 * correction polynomial coefficients, to Doppler scaling fac-
				 * tors, and to radar and optical albedos                  */

				vary_params_gpu(dpar,dmod,ddat,11,&deldop_zmax,
						&rad_xsec, &opt_brightness, &cos_subradarlat,
						nframes, lc_n, nviews, verts, htype, dtype, nf, nsets,
						bf_stream, max_frames, fit_overflow);

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
				realize_photo_gpu(dpar, dmod, radalb_factor, optalb_factor, 2, nf);  /* reset R, then R_save */

				/* Must update opt_brightness_save for Hapke optical scattering
				 * law, since single-scattering albedo w isn't just an overall
				 * scaling factor  */
				if (vary_hapke)
					vary_params_gpu(dpar,dmod,ddat,12,&dummyval2,
							&dummyval3,&opt_brightness,&dummyval4,
							nframes, lc_n, nviews, verts, htype, dtype, nf, nsets,
							bf_stream, max_frames, fit_overflow);

			} else if (newphoto) {
				rad_xsec_save = rad_xsec;
				opt_brightness_save = opt_brightness;
				realize_photo_gpu(dpar, dmod, 1.0, 1.0, 0, nf);  /* set R_save to R */
			}
			if ((newsize && vary_delcor0_size) || ((newshape || newspin) &&
					vary_delcor0_shapespin)) {
				deldop_zmax_save = deldop_zmax;
				realize_delcor_gpu(ddat, delta_delcor0, 2, nsets, nframes);  /* reset delcor0, then delcor0_save */
			} else if (newdelcor)
				realize_delcor_gpu(ddat, 0.0, 0, nsets, nframes);  /* set delcor0_save to delcor0 */

			if ((newspin && vary_dopscale_spin) || ((newsize || newshape) &&
					vary_dopscale_sizeshape)) {
				cos_subradarlat_save = cos_subradarlat;
				realize_dopscale_gpu(dpar, ddat, dopscale_factor, 2, nsets, dtype);  /* reset dopscale, then dopscale_save */
			} else if (newdopscale) {
				realize_dopscale_gpu(dpar, ddat, 1.0, 0, nsets, dtype);  /* set dopscale_save to dopscale */
			}
			if (newxyoff)
				realize_xyoff_gpu(ddat, nsets, dtype);

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
				objective_gpu(hotparamval, verts, htype, dtype,
						nframes, nviews, lc_n, nsets, nf, bf_stream, fit_overflow);
			/* Launch single-thread kernel to retrieve flags in dev_par */
			bf_get_flags_krnl<<<1,1>>>(dpar, flags);
			checkErrorAfterKernelLaunch("bf_get_flags_krnl");
			gpuErrchk(cudaMemcpy(hflags, flags, sizeof(int)*7,
					cudaMemcpyDeviceToHost));
			/* Display the objective function after each parameter adjustment.  */
			printf("%4d %8.6f %d", p, enderr, iround(par->fpartype[p]));
			if (hflags[0]==1)		printf("  (BAD DIAMS)");
			if (hflags[1]==1)		printf("  (BAD PHOTO)");
			if (hflags[2]==1)		printf("  (BAD POS)");
			if (hflags[3]==1)		printf("  (BAD POSET)");
			if (hflags[4]==1)		printf("  (BAD RADAR)");
			if (hflags[5]==1)		printf("  (BAD DOPSCALE)");
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
			if (++cntr >= npar_update) {
				cntr = 0;
				showvals = 1;
				calc_fits_gpu(dpar, dmod, ddat, verts, nviews,
						nframes, lc_n, htype, nsets, nf, bf_stream, max_frames, fit_overflow);
				chi2_gpu(dpar, ddat, htype, dtype, nframes,
						lc_n, 0, nsets, bf_stream, max_frames);

				write_mod( dpar, dmod);
				write_dat( dpar, ddat);
			}

		}  // End fitpar loop

		/* End of this iteration: Write model and data to disk, and display the
		 * region within each delay-Doppler or Doppler frame for which model
		 * power is nonzero.                                               */
		if (cntr != 0) {
			calc_fits_gpu(dpar, dmod, ddat, verts, nviews,
					nframes, lc_n, htype, nsets, nf, bf_stream, max_frames, fit_overflow);
			chi2_gpu(dpar, ddat, htype, dtype, nframes,
					lc_n, 0, nsets, bf_stream, max_frames);

			write_mod( dpar, dmod);
			write_dat( dpar, ddat);
		}
		show_deldoplim_gpu(ddat, htype, nsets, nframes, max_frames);

		/* Check if we should start a new iteration  */
		if (iter == term_maxiter) {
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
			if (objfunc_start > 0.0)
				keep_iterating = ((objfunc_start - enderr)/enderr >= term_prec);
			else
				keep_iterating = 1;
			first_fitpar = 0;     /* for all iterations after the first iteration */

		} else if (term_badmodel && (hflags[0] || hflags[1] || hflags[2] ||
				hflags[3] || hflags[4] || hflags[5]) ) {

			/* Just completed a full iteration, stop iterating because "term_
			 * badmodel" parameter is turned on and model has a fatal flaw: it
			 * extends beyond POS frame OR it one or more illegal photometric
			 * parameters OR it has one or more tiny or negative ellipsoid dia-
			 * meters OR it has plane-of-sky fit frames too small to "contain"
			 * model OR it is too wide in delay-Doppler space for (delay-)
			 * Doppler fit frames to be correctly constructed OR it has out-of-
			 * range values for one or more Doppler scaling factors    */
			keep_iterating = 0;

		} else {
			/* Just completed a full iteration and the model has no fatal flaws
			 * (or else the "term_badmodel" parameter is turned off): keep
			 * iterating if fractional decrease objective function during the
			 * just-completed iteration was greater than term_prec         */
			keep_iterating = ((beginerr - enderr)/enderr >= term_prec);
		}

	} while (keep_iterating);

		/* Show final values of reduced chi-square, individual penalty functions,
		 * and the objective function  */
		final_chi2 = chi2_gpu(dpar, ddat, htype, dtype, nframes,
				lc_n, 1, nsets, bf_stream, max_frames);

		final_redchi2 = final_chi2/dat->dof;
		printf("# search completed\n");

		/* Launch single-thread kernel to get these final flags from dev->par:
		 * pen.n, baddiam, badphoto, posbnd, badposet, badradar, baddopscale */
		/* Launch single-thread kernel to retrieve flags in dev_par */
		bf_get_flags_krnl<<<1,1>>>(dpar, flags);
		checkErrorAfterKernelLaunch("bf_get_flags_krnl");
		gpuErrchk(cudaMemcpy(hflags, flags, sizeof(int)*7,
				cudaMemcpyDeviceToHost));

		if (par->pen.n > 0 || hflags[0] || hflags[1] || hflags[2]	|| hflags[3] ||
				hflags[4] || hflags[5]) {
			printf("#\n");
			printf("# %15s %e\n", "reduced chi2", final_redchi2);
			if (par->pen.n > 0) {
				par->showstate = 1;
			penalties_gpu(dpar, dmod, ddat, par->showstate);
			par->showstate = 0;
		}
		if (hflags[0]==1)
			printf("# objective func multiplied by %.1f: illegal ellipsoid diameters\n",
					baddiam_factor);
		if (hflags[1]==1)
			printf("# objective func multiplied by %.1f: illegal photometric parameters\n",
					badphoto_factor);
		if (hflags[2]==1)
			printf("# objective func multiplied by %.1f: model extends beyond POS frame\n",
					posbnd_factor);
		if (hflags[3]==1)
			printf("# objective func multiplied by %.1f: "
					"model extends beyond plane-of-sky fit image\n",
					badposet_factor);
		if (hflags[4]==1)
			printf("# objective func multiplied by %.1f: "
					"model is too wide in delay-Doppler space to construct fit image\n",
					badradar_factor);
		if (hflags[5]==1)
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
	printf("GPU fit enderr: %3.8g\n", enderr);
	fflush(stdout);

	/* Destroy the streams */
	cudaSetDevice(GPU0);
	for (int f=0; f<max_frames; f++)
		cudaStreamDestroy(bf_stream[f]);

	free(hflags);
	free(htype);
	free(nframes);
	free(lc_n);
	free(nviews);
	free(hfparstep);
	free(hfpartol);
	free(hfparabstol);
	free(hfpartype);
	cudaFree(sdev_par);
	cudaFree(sdev_mod);
	cudaFree(sdev_dat);
	cudaFree(fparstep);
	cudaFree(fpartol);
	cudaFree(fparabstol);
	cudaFree(fpartype);
	cudaFree(fpntr);
	cudaFree(flags);
	cudaFree(dtype);
	cudaFree(verts);
	cudaFree(fit_overflow);
	cudaDeviceReset();
	return enderr;
}

/* objective_gpu is a version of objective_cuda that takes an extra
 * argument - the cudaStreams created in bestfit_cuda2. The goal is to
 * reduce overhead from stream creation/destruction to a minimum by having
 * just one set number of streams per program run.  */
__host__ double objective_gpu(
		double x,
		struct vertices_t **verts,
		unsigned char *htype,
		unsigned char *dtype,
		int *nframes,
		int *nviews,
		int *lc_n,
		int nsets,
		int nf,
		cudaStream_t *bf_stream,
		double **fit_overflow)
{
	double err, pens, delta_delcor0, dopscale_factor, radalb_factor,
		optalb_factor, *dlogfactors, *hlogfactors;
//	unsigned char *dflags, *hflags;
	int *dflags, *hflags;
	int max_frames;

	gpuErrchk(cudaMalloc((void**)&dflags, sizeof(int)*6));
	gpuErrchk(cudaMalloc((void**)&dlogfactors, sizeof(double)*7));
	hflags 	 	= (int *) malloc(6*sizeof(int));
	hlogfactors	= (double *) malloc(7*sizeof(double));

	/* Initialize local parameters  */
	delta_delcor0 = 0.0;
	dopscale_factor = radalb_factor = optalb_factor = 1.0;

	/* Assign new trial value to the model parameter being adjusted  */
	bf_set_hotparam_val_krnl<<<1,1>>>(x);	//(*hotparam) = x;
	checkErrorAfterKernelLaunch("bf_set_hotparam_val_krnl (in objective_cuda)");
	gpuErrchk(cudaMemcpyFromSymbol(&max_frames, dmax_frames,
			sizeof(int),	0, cudaMemcpyDeviceToHost));

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
	 * and realize_dopscale, respectively.*/

	if (newsize || newshape)
		realize_mod_gpu(sdev_par, sdev_mod, type, nf, bf_stream);
	if (newspin)
		realize_spin_gpu(sdev_par, sdev_mod, sdev_dat, htype, nframes,
				nviews, nsets, bf_stream);

	if ((newsize && vary_alb_size) || ((newshape || newspin) && vary_alb_shapespin))
		realize_photo_gpu(sdev_par, sdev_mod, 1.0, 1.0, 1, nf);  /* set R to R_save */
	if ((newsize && vary_delcor0_size) || ((newshape || newspin) && vary_delcor0_shapespin)) {
		realize_delcor_gpu(sdev_dat, 0.0, 1, nsets, nframes);  /* set delcor0 to delcor0_save */
	}

	if ((newspin && vary_dopscale_spin) || ((newsize || newshape) && vary_dopscale_sizeshape))
		realize_dopscale_gpu(sdev_par, sdev_dat, 1.0, 1, nsets, dtype);  /* set dopscale to dopscale_save */
	if (call_vary_params) {
		/* Call vary_params to get the trial adjustments to 0th-order delay correc-
		 * tion polynomial coefficients, to Doppler scaling factors,and to radar
		 * and optical albedos, then send them to the branch nodes  */
		vary_params_gpu(sdev_par, sdev_mod, sdev_dat, spar->action,
				&deldop_zmax, &rad_xsec, &opt_brightness, &cos_subradarlat,
				nframes, lc_n, nviews, verts, htype, dtype, nf, nsets,
				bf_stream, max_frames, fit_overflow);

		delta_delcor0 = (deldop_zmax - deldop_zmax_save)*KM2US;
//		printf("deldop_zmax, %f, deldop_zmax_save, %f, delta_delcor0, %f\n",
//						deldop_zmax, deldop_zmax_save, delta_delcor0);
		if (cos_subradarlat != 0.0)
			dopscale_factor = cos_subradarlat_save/cos_subradarlat;
		if (rad_xsec != 0.0)
			radalb_factor = rad_xsec_save/rad_xsec;
		if (opt_brightness != 0.0)
			optalb_factor = opt_brightness_save/opt_brightness;
	}

	if ((newsize && vary_alb_size) || ((newshape || newspin) && vary_alb_shapespin))
		realize_photo_gpu(sdev_par, sdev_mod, radalb_factor, optalb_factor, 1, nf);  /* adjust R */
	else if (newphoto)
		realize_photo_gpu(sdev_par, sdev_mod, 1.0, 1.0, 0, nf);  /* set R_save to R */
	if ((newsize && vary_delcor0_size) || ((newshape || newspin) && vary_delcor0_shapespin)) {
		realize_delcor_gpu(sdev_dat, delta_delcor0, 1, nsets, nframes);  /* adjust delcor0 */
	}
	else if (newdelcor) {
		realize_delcor_gpu(sdev_dat, 0.0, 0, nsets, nframes);  /* set delcor0_save to delcor0 */
	}
	if ((newspin && vary_dopscale_spin) || ((newsize || newshape) && vary_dopscale_sizeshape))
		realize_dopscale_gpu(sdev_par, sdev_dat, dopscale_factor, 1, nsets, dtype);  /* adjust dopscale */
	else if (newdopscale)
		realize_dopscale_gpu(sdev_par, sdev_dat, 1.0, 0, nsets, dtype);  /* set dopscale_save to dopscale */
	if (newxyoff)
		realize_xyoff_gpu(sdev_dat, nsets, dtype);

	calc_fits_gpu(sdev_par, sdev_mod, sdev_dat, verts, nviews, nframes, lc_n,
			htype, nsets, nf, bf_stream, max_frames, fit_overflow);
	err = chi2_gpu(sdev_par, sdev_dat, htype, dtype, nframes, lc_n, 0, nsets,
			bf_stream, max_frames);

	/* Divide chi-square by DOF to get reduced chi-square.    */
	err /= sdat->dof;

	/* If bestfit has set showvals = 1, display reduced chi-square. Then set
	 * spar->showstate = 1, so that when function penalties is called later,
	 * it "knows" that it should display the individual penalty values.
	 * Reset showstate to 0 if showvals = 0.  */
//	showvals=1;
	if (showvals) {
		printf("# %15s %e\n", "reduced chi2", err);
		spar->showstate = 1;
	}
	else
		spar->showstate = 0;

//	printf("showvals: %i\n", showvals);
//	printf("spar->showstate: %i\n", spar->showstate);
	/* Compute penalties and add to reduced chi-square. Individual penalty values
	 * will be displayed if we set spar->showstate = 1 a few lines back.        */
	pens = penalties_gpu(sdev_par, sdev_mod, sdev_dat, spar->showstate);
//	printf("err: %3.6f\n", err);
//	printf("pens: %3.6f\n", pens);
//	printf("%3.8g, %3.8g, %3.8g\n", x, err, pens);
//	printf("%3.8g, %3.8g\n", err, pens);
	err += pens;
	/* Double the objective function if there's an ellipsoid component with tiny
	 * or negative diameter, if any optical photometric parameters have invalid
	 * values, if any portion of the model lies outside specified POS window or
	 * outside any plane-of-sky fit image, or if model is too wide in delay-Dopp-
	 * ler space for any (delay-)Doppler fit image to be correctly constructed.
	 * This effectively rules out any models with any of these flaws.         */
	/* NOTE: TO-DO: baddiam may need to come from elsewhere other than spar.
	 * However, bestfit gets called only once and spar/smod/sdat gets copied
	 * only once.
	 * flags[0] = dpar->baddiam;
		flags[1] = dpar->badphoto;
		flags[2] = dpar->posbnd;
		flags[3] = dpar->badposet;
		flags[4] = dpar->badradar;
		flags[5] = dpar->baddopscale;
		dlogfactors[0] = dpar->bad_objfactor;
		dlogfactors[1] = dpar->baddiam_logfactor;
		dlogfactors[2] = dpar->badphoto_logfactor;
		dlogfactors[3] = dpar->posbnd_logfactor;
		dlogfactors[4] = dpar->badposet_logfactor;
		dlogfactors[5] = dpar->badradar_logfactor;
		dlogfactors[6] = dpar->baddopscale_logfactor;
	 */
	ocs_get_flags_krnl<<<1,1>>>(sdev_par, dflags, dlogfactors);
	checkErrorAfterKernelLaunch("bf_get_flags_krnl");
	gpuErrchk(cudaMemcpy(hflags, dflags, sizeof(int)*6,
			cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(hlogfactors, dlogfactors, sizeof(double)*7,
			cudaMemcpyDeviceToHost));

	if (hflags[0]==1) {
		baddiam_factor = hlogfactors[0] * exp(hlogfactors[1]);
		err *= baddiam_factor;
		if (showvals)
			printf("# objective func multiplied by %.1f: illegal ellipsoid diameters\n",
					baddiam_factor);
	}
	if (hflags[1]==1) {
		badphoto_factor = hlogfactors[0] * exp(hlogfactors[2]);
		err *= badphoto_factor;
		if (showvals)
			printf("# objective func multiplied by %.1f: illegal photometric parameters\n",
					badphoto_factor);
	}
	if (hflags[2]==1) {
		check_posbnd = 1;     /* tells bestfit about this problem */
		posbnd_factor = hlogfactors[0] * exp(hlogfactors[3]);
		err *= posbnd_factor;
		if (showvals)
			printf("# objective func multiplied by %.1f: model extends beyond POS frame\n",
					posbnd_factor);
	}
	if (hflags[3]==1) {
		check_badposet = 1;     /* tells bestfit about this problem */
		badposet_factor = hlogfactors[0] * exp(hlogfactors[4]);
		err *= badposet_factor;
		if (showvals)
			printf("# objective func multiplied by %.1f: plane-of-sky fit frame too small\n",
					badposet_factor);
	}
	if (hflags[4]==1) {
		check_badradar = 1;     /* tells bestfit about this problem */
		badradar_factor = hlogfactors[0] * exp(hlogfactors[5]);
		err *= badradar_factor;
		if (showvals)
			printf("# objective func multiplied by %.1f: model too wide in delay-Doppler space\n",
					badradar_factor);
	}
	if (hflags[5]==1) {
		baddopscale_factor = hlogfactors[0] * exp(hlogfactors[6]);
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

	free(hflags);
	free(hlogfactors);
	cudaFree(dflags);
	cudaFree(dlogfactors);
	//printf("(GPU MODE) err (return value): %g\n", err);
	return err;
}
