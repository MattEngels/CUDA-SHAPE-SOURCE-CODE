/*****************************************************************************************
                                                                             delcorinit.c

For each delay correction polynomial, set the coefficients to reasonable initial values by
aligning the centroid of the model's power with the data centroid.  The coefficients are
estimated via singular value decomposition, with each frame weighted by the number of
delay-Doppler pixels (or Doppler bins) times the explicit weight given in the obs file.

The link between Doppler offsets and delay drift is accounted for: Doppler is proportional
to the negative time derivative of delay.  Hence each delay-Doppler frame provides two
constraints on the coefficients -- matching the data vs. fit centroids in delay and also
in Doppler -- while each Doppler frame provides one constraint.

Constant coefficients (state = 'c') are never changed.  Datasets whose coefficients have
state = '=' contribute to corrections to the appropriate preceding datasets.

Noisy data can yield an absurd data centroid -- especially if the power summed over the
frame is very small and/or negative, in which case the centroid may lie far outside the
frame limits.  For this reason each frame whose data centroid will be computed rather than
read from a file (see below) must have sufficiently high signal-to-noise to be used for
coefficient estimation.  The user-specified "delcorthresh" parameter represents this
threshold (default = 2.0).

Another approach, useful for weak or poorly normalized data, is to hand-specify the data
centroids (presumably after visually inspecting the data).  It's possible to do this for
just some frames, leaving the centroids of the others to be computed automatically.  The
"delcor_read" parameter must be set to "yes" and the "delcor_file" parameter must be the
name of the input file containing the centroids.  This file must have an entry for each
frame of each Doppler or delay-Doppler dataset, even those frames for which automatic
centroid computation will be used, and even those frames with zero weight.  An entry line
for a delay-Doppler frame contains five values: dataset number and frame number (both
0-based); a read/compute flag which is 1 if the data centroid is to be hand-specified and
0 if it is to be computed; and the delay row number and Doppler column number of the
hand-specified centroid (both floating-point and 1-based).  The last two values must be
included even if the centroid is automatically computed rather than hand-specified, but
they are not used in this case, so they can be set to 0.  An entry line for a Doppler
frame has only four values: dataset and frame numbers, read/compute flag, and Doppler bin
number of the hand-specified centroid (floating-point and 1-based).

If "delcor_read" is omitted or is set to "no" then data centroids for all frames are
computed rather than hand-specified.

If "delcor_verbose" is set to "yes" then the data centroids used for all frames, whether
computed or read from disk, are printed to the screen.  The printed values are 1-based
delay row and Doppler column numbers before any vignetting has been applied.

Fit centroids are computed by routines deldopcentroid and dopcentroid, which use various
quantities already computed by routines pos2deldop and pos2doppler, respectively.  (Those
routines in turn are called by the calc_fits routine).  Initially I tried to do without
fit centroids, instead shifting the midpoint of the rectangular fit limits (given by
frame->idellim and frame->idoplim) to match the data centroid.  That simple approach
didn't work very well: The fit's delay midpoint is strongly biased away from us by very
weak power at the model's limbs.

This routine was written for the new "delcorinit" action, which could be useful to get the
delay correction polynomials into the right ballpark before starting to perform fits for a
new target.

Modified 2015 June 10 by CM:
    Implement smearing for the "fit" and "write" actions

Modified 2014 July 4 by CM:
    Bug fix: the "deldop_noisesum" routine was computing the summed noise (slightly)
         incorrectly for images with stride > 1

Modified 2014 February 25 by CM:
    Implement "delcor_verbose" parameter
    Bug fix: in "deldop_constraints" and "doppler_constraints" routines, "read_centroid"
        should be integer, not double

Modified 2009 July 3 by CM:
    Improved error message when delcor_file entries are out of sequence

Modified 2008 August 19 by CM:
    Correct hand-specified (delay-)Doppler centroid values for vignetting

Modified 2007 August 4 by CM:
    Increased MAXCONSTRAINTS

Modified 2006 October 1 by CM:
    Add two new arguments to realize_delcor

Modified 2006 June 21 by CM:
    In deldop_constraints, changed delres to del_per_pixel and dopres to
        dop_per_pixel
    In doppler_constraints, changed dopres to dop_per_bin

Modified 2006 June 18 by CM:
    Allow each delay-Doppler frame within a dataset to have different
        dimensions after vignetting
    Allow each Doppler frame within a dataset to have different
        dimensions after vignetting

Modified 2006 March 10 by CM:
    Shrink the deldopcentroid and dopcentroid routines by using the
        information already computed by the pos2deldop and pos2doppler
        routines (which have been modified accordingly).  Note that this
        scheme relies on the fact that calc_fits is called before
        delcorinit is called.

Modified 2005 July 20 by CM:
    Compute and display mean and rms residuals in delay and Doppler
    Fixed bug in doppler_constraints: needed to add the fixed correction
        terms back to the fit centroid after subtracting the
        old correction
    Don't use any Doppler constraint if the only coefficient to be fit
        is the zeroth-order coefficient
    Add "facet" argument to radlaw routine

Modified 2005 March 10 by CM:
    Weights (listed in the obs file) are now floating-point, not integer,
        so change "weightfactor" variable accordingly

Modified 2005 March 1 by CM:
    Renamed "var" to "oneovervar" (1/variance) in a couple of places

Modified 2005 January 25 by CM:
    Removed unused variables

Modified 2004 November 10 by CM:
    Correct the 2004 October 6 correction: Don't try to change a polynomial
        if the number of *constraints* provided by the usable frames
        (two constraints per delay-Doppler frame, one per Doppler frame)
        is less than the number of floating polynomial coefficients

Modified 2004 October 25 by CM:
    Only fit the floating coefficients, instead of fitting all coefficients
        (as if they all were floating) and then changing only the floating
        coefficients

Modified 2004 October 6 by CM:
    Don't try to change a polynomial if the number of usable frames
        is less than the number of polynomial coefficients

Modified 2004 July 31 by CM:
    Add "delcor_read" and "delcor_file" parameters to permit user to
        hand-specify data centroids for some or all frames
    Add call to realize_delcor routine in order to assign coefficients
        which have state = '='

Modified 2004 Feb 19 by CM:
    Only use frames whose signal-to-noise ratio is sufficiently strong
    Major overhaul: Use singular value decomposition to estimate all
                    coefficients, rather than using a simple-minded
                    algorithm to estimate just the first two
                    coefficients (with all others set to zero)

Written 2003 May 14 by CM
 *****************************************************************************************/

#include "head.h"

#define MAXCONSTRAINTS 10000
#define MAXCOEFFS 20
#define SVDTOL 1.0e-5

static int *is_delay_constraint_dummy;
static double **design_dummy, *datavec_dummy, *sigma_dummy;

void deldop_constraints( struct par_t *par, struct mod_t *mod,
		struct deldop_t *deldop, int s, FILE *fp, int *fit_coeff,
		int *nframes_all, int *nframes_use, int *nconstraints);
void doppler_constraints( struct par_t *par, struct mod_t *mod,
		struct doppler_t *doppler, int s, FILE *fp, int *fit_coeff,
		int *nframes_all, int *nframes_use, int *nconstraints);
void estimate_coeffs( struct dat_t *dat, int s_delcor, int type_delcor,
		int n_delcor, int *fit_coeff, int nframes_all, int nframes_use,
		int nconstraints);
double deldop_noisesum( int ndel, int ndop, int codemethod, int spb, int stride);
void svdfit_delcor( double **design, double y[], double sig[], int ndata, double a[],
		int ma, double **u, double **v, double w[], double *chisq);
void deldopcentroid( struct deldop_t *deldop, int frm,
		double *del_mean, double *dop_mean);
void dopcentroid( struct doppler_t *doppler, int frm, double *dop_mean);


void delcorinit( struct par_t *par, struct mod_t *mod, struct dat_t *dat)
{
	FILE *fp = 0;
	int s_delcor, type_delcor, n_delcor, s, n, nframes_all, nframes_use,
	nconstraints, i;
	int *fit_coeff;
	double t0_delcor, t0;

	/*  If we will be reading in data centroids for some frames, open the input file  */

	if (par->delcor_read)
		FOPEN( fp, par->delcor_file, "r");

	/*  Initialize a matrix and two vectors which are "seen" by several
      subroutines; they are used to hold elements of the design matrix,
      data vector, and standard deviation vector (needed to get delay
      correction polynomial coefficients via singular value decomposition)
      for a given dataset or group of datasets.  Once all frames in the
      dataset(s) have been inspected, and it is clear how many usable
      constraints we can place on the coefficients, the elements will be
      copied into a new matrix and vectors with the correct dimensions.

      A third vector is Boolean: 1 if a constraint is a delay constraint,
      0 if it is a Doppler constraint.                                     */

	design_dummy = matrix( 1, MAXCONSTRAINTS, 1, MAXCOEFFS);
	datavec_dummy = vector( 1, MAXCONSTRAINTS);
	sigma_dummy = vector( 1, MAXCONSTRAINTS);
	is_delay_constraint_dummy = ivector( 1, MAXCONSTRAINTS);

	/*
      If a dataset has delay correction polynomial coefficients with
      state = '=', go backwards in the obs file until we reach a
      delay-Doppler or Doppler dataset whose polynomial coefficients
      have states 'f' and/or 'c' rather than '='.

      s_delcor is the number of the dataset we find.
      type_delcor tells whether that dataset is delay-Doppler or Doppler.
      n_delcor is the order of that dataset's polynomial (# coeffs - 1).
      t0_delcor is the reference epoch for the polynomial.
	 */

	s_delcor = -1;
	type_delcor = -1;
	n_delcor = 0;
	t0_delcor = -1.0;

	nframes_all = 0;    /* frames governed by a given set of delay coefficients    */
	nframes_use = 0;    /* frames sufficiently strong for computing coefficients   */
	nconstraints = 0;   /* combined # of delay constraints and Doppler constraints */

	fit_coeff = ivector( 0, n_delcor);  /* 1 if coefficient state = 'f', 0 if 'c' */

	for (s=0; s<dat->nsets; s++) {

		if (dat->set[s].type == DELAY) {

			n = dat->set[s].desc.deldop.delcor.n;
			t0 = dat->set[s].desc.deldop.delcor.t0;

			if (dat->set[s].desc.deldop.delcor.a[0].state != '=') {

				/*  Finish up the polynomial we've been working on
              before starting to do the next one              */

				estimate_coeffs( dat, s_delcor, type_delcor, n_delcor, fit_coeff,
						nframes_all, nframes_use, nconstraints);
				free_ivector( fit_coeff, 0, n_delcor);

				/*  Now start the next one  */

				s_delcor = s;
				type_delcor = DELAY;
				n_delcor = n;
				if (n_delcor+1 > MAXCOEFFS)
					bailout("must increase MAXCOEFFS in delcorinit.c\n");
				t0_delcor = t0;
				nframes_all = 0;
				nframes_use = 0;
				nconstraints = 0;
				fit_coeff = ivector( 0, n_delcor);
				for (i=0; i<=n_delcor; i++)
					fit_coeff[i] = (dat->set[s_delcor].desc.deldop.delcor.a[i].state == 'f');
				deldop_constraints( par, mod, &dat->set[s].desc.deldop, s, fp, fit_coeff,
						&nframes_all, &nframes_use, &nconstraints);
			}
			else if (s_delcor < 0)
				bailout("can't use \"=\" state for the first delay polynomial\n");
			else if (n != n_delcor)
				bailout("delay polynomials must have same degree if state = \"=\"\n");
			else if (fabs(t0 - t0_delcor) > HALFSECOND)
				bailout("delay polynomials must have same t0 if state = \"=\"\n");
			else
				deldop_constraints( par, mod, &dat->set[s].desc.deldop, s, fp, fit_coeff,
						&nframes_all, &nframes_use, &nconstraints);

		} else if (dat->set[s].type == DOPPLER) {

			n = dat->set[s].desc.doppler.delcor.n;
			t0 = dat->set[s].desc.doppler.delcor.t0;

			if (dat->set[s].desc.doppler.delcor.a[0].state != '=') {

				/*  Finish up the polynomial we've been working on
              before starting to do the next one              */

				estimate_coeffs( dat, s_delcor, type_delcor, n_delcor, fit_coeff,
						nframes_all, nframes_use, nconstraints);
				free_ivector( fit_coeff, 0, n_delcor);

				/*  Now start the next one  */

				s_delcor = s;
				type_delcor = DOPPLER;
				n_delcor = n;
				if (n_delcor+1 > MAXCOEFFS)
					bailout("must increase MAXCOEFFS in delcorinit.c\n");
				t0_delcor = t0;
				nframes_all = 0;
				nframes_use = 0;
				nconstraints = 0;
				fit_coeff = ivector( 0, n_delcor);
				for (i=0; i<=n_delcor; i++)
					fit_coeff[i] = (dat->set[s_delcor].desc.doppler.delcor.a[i].state == 'f');
				doppler_constraints( par, mod, &dat->set[s].desc.doppler, s, fp, fit_coeff,
						&nframes_all, &nframes_use, &nconstraints);
			}
			else if (s_delcor < 0)
				bailout("can't use \"=\" state for the first delay polynomial\n");
			else if (n != n_delcor)
				bailout("delay polynomials must have same degree if state = \"=\"\n");
			else if (fabs(t0 - t0_delcor) > HALFSECOND)
				bailout("delay polynomials must have same t0 if state = \"=\"\n");
			else
				doppler_constraints( par, mod, &dat->set[s].desc.doppler, s, fp, fit_coeff,
						&nframes_all, &nframes_use, &nconstraints);

		}
	}

	/*  Finish up the last polynomial  */

	estimate_coeffs( dat, s_delcor, type_delcor, n_delcor, fit_coeff,
			nframes_all, nframes_use, nconstraints);
	free_ivector( fit_coeff, 0, n_delcor);

	/*  Go back through all datasets and assign coefficients with state = '='  */

	realize_delcor( dat, 0.0, 0);

	/*  Free up space no longer needed  */

	free_matrix( design_dummy, 1, MAXCONSTRAINTS, 1, MAXCOEFFS);
	free_vector( datavec_dummy, 1, MAXCONSTRAINTS);
	free_vector( sigma_dummy, 1, MAXCONSTRAINTS);
	free_ivector( is_delay_constraint_dummy, 1, MAXCONSTRAINTS);

	/*  If we read in data centroids for some frames, close the input file  */

	if (par->delcor_read)
		fclose( fp);

}


/*  deldop_constraints determines a single delay-Doppler dataset's
    contributions to the delay correction polynomial coefficients   */

void deldop_constraints( struct par_t *par, struct mod_t *mod,
		struct deldop_t *deldop, int s, FILE *fp, int *fit_coeff,
		int *nframes_all, int *nframes_use, int *nconstraints)
{
	int f, i, j, k, ncoeffs, ndel, ndop, s_read, f_read, nframes_set,
	use_frame, n_new_constraints, read_centroid;
	double obsdel_mean, obsdop_mean, obs_sum, fitdel_mean, fitdop_mean,
	obs_sum_threshold, obs_weight, dopfactor, delta_t, tmp, fixed_terms,
	weightfactor;

	/*  Initialize variables to avoid compilation warnings  */

	obsdel_mean = obsdop_mean = 0.0;

	/*  Initialize other variables  */

	nframes_set = deldop->nframes;
	ncoeffs = deldop->delcor.n + 1;

	/*  Multiply delay drift (usec/day) by dopfactor
      to get Doppler offset (Hz)                    */

	dopfactor = -deldop->Ftx / 86400.0;

	/*  Find out how many new delay and Doppler constraints are being
      placed on the coefficients: 0 if all coefficients are fixed;
      1 if only the zeroth-order coefficient will be fit; 2 otherwise  */

	n_new_constraints = (fit_coeff[0]) ? 1 : 0;
	for (i=1; i<=deldop->delcor.n; i++)
		if (fit_coeff[i])
			n_new_constraints = 2;

	/*  Loop through each frame in this dataset  */

	for (f=0; f<deldop->nframes; f++) {

		/*  Initialize frame-specific variables  */

		ndel = deldop->frame[f].ndel;
		ndop = deldop->frame[f].ndop;
		obs_sum_threshold = par->delcorthresh
				* deldop_noisesum( ndel, ndop, deldop->codemethod,
						deldop->spb, deldop->stride  );

		/*  Read in the data centroid if desired  */

		if (par->delcor_read) {
			s_read = getint( fp);
			f_read = getint( fp);
			if (s_read != s || f_read != f) {
				printf("%s lists set %d frame %d where it should list delay-Doppler set %d frame %d\n",
						par->delcor_file, s_read, f_read, s, f);
				bailout("delcor_file entries are out of sequence\n");
			}
			read_centroid = getint( fp);
			obsdel_mean = getdouble( fp) - deldop->frame[f].idelvig[0] + 1;
			obsdop_mean = getdouble( fp) - deldop->frame[f].idopvig[0] + 1;
		} else {
			read_centroid = 0;
		}

		/*  If we're not getting this frame's data centroid from a disk file, compute
        it, using pixel strength (in standard deviations) as the weighting factor  */

		if (!read_centroid) {
			obsdel_mean = 0.0;
			obsdop_mean = 0.0;
			obs_sum = 0.0;
			for (i=1; i<=ndel; i++)
				for (j=1; j<=ndop; j++) {
					obs_weight = deldop->frame[f].obs[i][j]
					                                     * sqrt(deldop->frame[f].oneovervar[i][j]);
					obsdel_mean += i*obs_weight;
					obsdop_mean += j*obs_weight;
					obs_sum += obs_weight;
				}

			if (obs_sum != 0.0) {
				obsdel_mean /= obs_sum;
				obsdop_mean /= obs_sum;
			} else {
				obsdel_mean = obsdop_mean = -999.9;
			}

			/*  Proceed only if the sum of all pixel values is large enough
            (that is, if this frame's signal-to-noise is strong enough)  */

			use_frame = (obs_sum >= obs_sum_threshold);
		} else {
			use_frame = 1;
		}

		/*  Print data centroid (before any vignetting) to the screen if desired  */

		if (par->delcor_verbose) {
			printf("%2d %3d  %d  %.1f  %.1f",
					s, f, read_centroid,
					obsdel_mean + deldop->frame[f].idelvig[0] - 1,
					obsdop_mean + deldop->frame[f].idopvig[0] - 1);
			if (!use_frame)
				printf("  unusable");
			printf("\n");
		}

		/*  Skip this frame if its weight is zero
        or if there are no delay coefficients to fit  */

		weightfactor = ndel * ndop * deldop->frame[f].weight;

		if (weightfactor > 0.0 && n_new_constraints > 0) {

			*nframes_all += 1;

			if (use_frame) {

				*nframes_use += 1;
				if (*nconstraints+n_new_constraints > MAXCONSTRAINTS)
					bailout("must increase MAXCONSTRAINTS in delcorinit.c\n");

				/*  Get the fit centroid for this frame
            (according to the OLD correction polynomial)  */

				deldopcentroid( deldop, f, &fitdel_mean, &fitdop_mean);

				/*  Compute (frame epoch - delay polynomial reference epoch)  */

				delta_t = deldop->frame[f].view[deldop->v0].t - deldop->delcor.t0;

				/*  Compute this frame's contributions to the design matrix
            (one delay constraint and perhaps one Doppler constraint),
            to the data vector (delay and Doppler data-minus-fit
            centroid mismatches in usec and Hz), and to the vector of
            "standard deviations" ( = 1 / sqrt(weight) ).

            More precisely, design_dummy[i][k] is the factor which
            multiplies the k-th delay or Doppler coefficient in the
            i-th constraint, and datavec_dummy[i] is the data centroid
            minus the fit centroid, where we have subtracted the old
            correction from the fit centroid but then added back the
            correction terms which are being held fixed.                */

				*nconstraints += 1;
				fixed_terms = 0.0;
				k = 0;
				for (j=1, tmp=1.0; j<=ncoeffs; j++, tmp*=delta_t) {
					if (fit_coeff[j-1])
						design_dummy[*nconstraints][++k] = tmp;
					else
						fixed_terms += tmp*deldop->delcor.a[j-1].val;
				}
				datavec_dummy[*nconstraints] =
						(obsdel_mean - (fitdel_mean - deldop->frame[f].view[deldop->v0].deloff
								+ fixed_terms))
								* deldop->del_per_pixel;
				sigma_dummy[*nconstraints] = 1.0 / sqrt(weightfactor);
				is_delay_constraint_dummy[*nconstraints] = 1;

				if (n_new_constraints == 2) {
					*nconstraints += 1;
					fixed_terms = 0.0;
					k = 0;
					if (fit_coeff[0])
						design_dummy[*nconstraints][++k] = 0.0;
					for (j=2, tmp=dopfactor; j<=ncoeffs; j++, tmp*=delta_t) {
						if (fit_coeff[j-1])
							design_dummy[*nconstraints][++k] = (j-1)*tmp;
						else
							fixed_terms += (j-1)*tmp*deldop->delcor.a[j-1].val;
					}
					datavec_dummy[*nconstraints] =
							(obsdop_mean - (fitdop_mean - deldop->frame[f].view[deldop->v0].dopoff
									+ fixed_terms))
									* deldop->dop_per_pixel;
					sigma_dummy[*nconstraints] = 1.0 / sqrt(weightfactor);
					is_delay_constraint_dummy[*nconstraints] = 0;
				}
			}
		}
	}
}


/*  doppler_constraints determines a single Doppler dataset's
    contributions to the delay correction polynomial coefficients  */

void doppler_constraints( struct par_t *par, struct mod_t *mod,
		struct doppler_t *doppler, int s, FILE *fp, int *fit_coeff,
		int *nframes_all, int *nframes_use, int *nconstraints)
{
	int f, i, j, k, ncoeffs, ndop, s_read, f_read, nframes_set, use_frame,
	n_new_constraints, read_centroid;
	double obsdop_mean, obs_sum, fitdop_mean, obs_sum_threshold,
	obs_weight, dopfactor, delta_t, tmp, fixed_terms, weightfactor;

	/*  Initialize a variable to avoid compilation warnings  */

	obsdop_mean = 0.0;

	/*  Initialize other variables  */

	nframes_set = doppler->nframes;
	ncoeffs = doppler->delcor.n + 1;

	/*  Multiply delay drift (usec/day) by dopfactor
      to get Doppler offset (Hz)                    */

	dopfactor = -doppler->Ftx / 86400.0;

	/*  Find out how many new Doppler constraints are being placed on
      the coefficients: 0 if all coefficients are fixed or if only
      the zeroth-order coefficient will be fit; 1 otherwise          */

	n_new_constraints = 0;
	for (i=1; i<=doppler->delcor.n; i++)
		if (fit_coeff[i])
			n_new_constraints = 1;

	/*  Loop through each frame in this dataset  */

	for (f=0; f<doppler->nframes; f++) {

		/*  Initialize frame-specific quantities  */

		ndop = doppler->frame[f].ndop;
		obs_sum_threshold = par->delcorthresh * sqrt(ndop);

		/*  Read in the data centroid if desired  */

		if (par->delcor_read) {
			s_read = getint( fp);
			f_read = getint( fp);
			if (s_read != s || f_read != f) {
				printf("%s lists set %d frame %d where it should list Doppler set %d frame %d\n",
						par->delcor_file, s_read, f_read, s, f);
				bailout("delcor_file entries are out of sequence\n");
			}
			read_centroid = getint( fp);
			obsdop_mean = getdouble( fp) - doppler->frame[f].idopvig[0] + 1;
		} else {
			read_centroid = 0;
		}

		/*  If we're not getting this frame's data centroid from a disk file, compute
        it, using bin strength (in standard deviations) as the weighting factor    */

		if (!read_centroid) {
			obsdop_mean = 0.0;
			obs_sum = 0.0;
			for (j=1; j<=ndop; j++) {
				obs_weight = doppler->frame[f].obs[j]
				                                   * sqrt(doppler->frame[f].oneovervar[j]);
				obsdop_mean += j*obs_weight;
				obs_sum += obs_weight;
			}

			if (obs_sum != 0.0)
				obsdop_mean /= obs_sum;
			else
				obsdop_mean = -999.9;

			/*  Proceed only if the sum of all bin values is large enough
            (that is, if this frame's signal-to-noise is strong enough)  */

			use_frame = (obs_sum >= obs_sum_threshold);
		} else {
			use_frame = 1;
		}

		/*  Print data centroid (before any vignetting) to the screen if desired  */

		if (par->delcor_verbose) {
			printf("%2d %3d  %d  %.1f",
					s, f, read_centroid,
					obsdop_mean + doppler->frame[f].idopvig[0] - 1);
			if (!use_frame)
				printf("  unusable");
			printf("\n");
		}

		/*  Skip this frame if its weight is zero
        or if there are no Doppler coefficients to fit  */

		weightfactor = ndop * doppler->frame[f].weight;

		if (weightfactor > 0.0 && n_new_constraints > 0) {

			*nframes_all += 1;

			if (use_frame) {

				*nframes_use += 1;
				if (*nconstraints+1 > MAXCONSTRAINTS)
					bailout("must increase MAXCONSTRAINTS in delcorinit.c\n");

				/*  Get the fit centroid for this frame
            (according to the OLD correction polynomial)  */

				dopcentroid( doppler, f, &fitdop_mean);

				/*  Compute (frame epoch - delay polynomial reference epoch)  */

				delta_t = doppler->frame[f].view[doppler->v0].t - doppler->delcor.t0;

				/*  Compute this frame's contributions to the design matrix
            (one Doppler constraint), to the data vector (Doppler
            data-minus-fit centroid mismatch in Hz), and to the
            vector of "standard deviations" ( = 1 / sqrt(weight) ).

            More precisely, design_dummy[i][k] is the factor which
            multiplies the k-th Doppler coefficient in the
            i-th constraint, and datavec_dummy[i] is the data centroid
            minus the fit centroid, where we have subtracted the old
            correction from the fit centroid but then added back the
            correction terms which are being held fixed.                */

				*nconstraints += 1;
				fixed_terms = 0.0;
				k = 0;
				if (fit_coeff[0])
					design_dummy[*nconstraints][++k] = 0.0;
				for (j=2, tmp=dopfactor; j<=ncoeffs; j++, tmp*=delta_t) {
					if (fit_coeff[j-1])
						design_dummy[*nconstraints][++k] = (j-1)*tmp;
					else
						fixed_terms += (j-1)*tmp*doppler->delcor.a[j-1].val;
				}
				datavec_dummy[*nconstraints] =
						(obsdop_mean - (fitdop_mean - doppler->frame[f].view[doppler->v0].dopoff
								+ fixed_terms))
								* doppler->dop_per_bin;
				sigma_dummy[*nconstraints] = 1.0 / sqrt(weightfactor);
				is_delay_constraint_dummy[*nconstraints] = 0;
			}
		}
	}
}


/*  estimate_coeffs uses singular value decomposition to
    estimate the coefficients for a delay correction polynomial  */

void estimate_coeffs( struct dat_t *dat, int s_delcor, int type_delcor,
		int n_delcor, int *fit_coeff, int nframes_all,
		int nframes_use, int nconstraints)
{
	int *is_delay_constraint;
	double **design, *datavec, *sigma, *coeffs, **u, **v, *w, chisq;
	int i, j, k, ncoeffs, ncoeffs_fit;
	double newpoly, resid, weightfactor, sumdelresid, sumdelresid2, sumdelweights,
	sumdopresid, sumdopresid2, sumdopweights, meanresid, rmsresid;

	ncoeffs = n_delcor + 1;
	ncoeffs_fit = 0;
	for (j=0; j<ncoeffs; j++) {
		if (fit_coeff[j])
			ncoeffs_fit++;
	}

	if (s_delcor < 0 || n_delcor < 0 || ncoeffs_fit == 0)
		return;

	printf("# set %2d has %2d of %2d frames usable for computing delay coefficients\n",
			s_delcor, nframes_use, nframes_all);

	/*  If there aren't enough frames with usable data centroids, the data
      won't provide enough constraints for estimating the delay correction
      polynomial coefficients, so just print a warning and leave this
      polynomial unchanged                                                  */

	if (nconstraints < ncoeffs_fit) {
		printf("WARNING: set %2d coefficients left unchanged\n", s_delcor);
		return;
	}

	/*  Create a design matrix, data vector, and standard deviation vector
      which have the correct dimensions, then copy the elements to them.  */

	design = matrix( 1, nconstraints, 1, ncoeffs_fit);
	datavec = vector( 1, nconstraints);
	sigma = vector( 1, nconstraints);
	is_delay_constraint = ivector( 1, nconstraints);
	for (i=1; i<=nconstraints; i++) {
		for (j=1; j<=ncoeffs_fit; j++)
			design[i][j] = design_dummy[i][j];
		datavec[i] = datavec_dummy[i];
		sigma[i] = sigma_dummy[i];
		is_delay_constraint[i] = is_delay_constraint_dummy[i];
	}

	/*  Create a vector to hold the best-fit coefficients.
      Also create two arrays and a vector needed by svdfit_delcor;
      on input they provide workspace, while on output they hold
      information which can be used to get covariances.             */

	coeffs = vector( 1, ncoeffs_fit);
	u = matrix( 1, nconstraints, 1, ncoeffs_fit);
	v = matrix( 1, ncoeffs_fit, 1, ncoeffs_fit);
	w = vector( 1, ncoeffs_fit);

	/*  Use singular value decomposition to get the best-fit
      coefficients for this delay correction polynomial.    */

	svdfit_delcor( design, datavec, sigma, nconstraints, coeffs, ncoeffs_fit,
			u, v, w, &chisq);

	/*  Change the polynomial  */

	if (n_delcor == 0)
		printf("WARNING: set %2d has a 0th-order polynomial, correction may fail\n",
				s_delcor);

	if (type_delcor == DELAY) {

		for (i=0, k=0; i<=n_delcor; i++) {
			if (fit_coeff[i])
				dat->set[s_delcor].desc.deldop.delcor.a[i].val = coeffs[++k];
			else
				printf("WARNING: set %2d has coeff %d held constant, correction may fail\n",
						s_delcor, i);
		}

	} else {

		for (i=0, k=0; i<=n_delcor; i++) {
			if (fit_coeff[i])
				dat->set[s_delcor].desc.doppler.delcor.a[i].val = coeffs[++k];
			else if (i > 0)
				printf("WARNING: set %2d has coeff %d held constant, correction may fail\n",
						s_delcor, i);
		}

	}

	/*  Compute and display the mean and rms residuals in delay and Doppler  */

	sumdelresid = 0.0;
	sumdelresid2 = 0.0;
	sumdelweights = 0.0;
	sumdopresid = 0.0;
	sumdopresid2 = 0.0;
	sumdopweights = 0.0;
	for (i=1; i<=nconstraints; i++) {
		newpoly = 0.0;
		for (j=1; j<=ncoeffs_fit; j++)
			newpoly += design[i][j]*coeffs[j];
		resid = datavec[i] - newpoly;
		weightfactor = 1.0/(sigma[i]*sigma[i]);
		if (is_delay_constraint[i]) {
			sumdelresid += weightfactor*resid;
			sumdelresid2 += weightfactor*resid*resid;
			sumdelweights += weightfactor;
		} else {
			sumdopresid += weightfactor*resid;
			sumdopresid2 += weightfactor*resid*resid;
			sumdopweights += weightfactor;
		}
	}
	printf("# set %2d mean, rms residuals:  ", s_delcor);
	if (sumdelweights > 0.0) {
		meanresid = sumdelresid/sumdelweights;
		rmsresid = sqrt( MAX( 0.0, sumdelresid2/sumdelweights - meanresid*meanresid));
		printf("%.3f, %.3f usec\n", meanresid, rmsresid);
		if (sumdopweights > 0.0)
			printf("#                              ");
	}
	if (sumdopweights > 0.0) {
		meanresid = sumdopresid/sumdopweights;
		rmsresid = sqrt( MAX( 0.0, sumdopresid2/sumdopweights - meanresid*meanresid));
		printf("%.3f, %.3f Hz\n", meanresid, rmsresid);
	}
	fflush(stdout);

	/*  Free up space used by arrays and vectors no longer needed  */

	free_matrix( design, 1, nconstraints, 1, ncoeffs_fit);
	free_vector( datavec, 1, nconstraints);
	free_vector( sigma, 1, nconstraints);
	free_ivector( is_delay_constraint, 1, nconstraints);
	free_vector( coeffs, 1, ncoeffs_fit);
	free_matrix( u, 1, nconstraints, 1, ncoeffs_fit);
	free_matrix( v, 1, ncoeffs_fit, 1, ncoeffs_fit);
	free_vector( w, 1, ncoeffs_fit);

}


/*  deldop_noisesum computes the rms summed noise for a full delay-Doppler frame,
    ndop Doppler columns x ndel delay rows, in units of the single-pixel standard
    deviation.  For 1-spb data this is simply sqrt(ndop*ndel), but for multiple spb
    we have correlated noise in nearby rows, resulting in the more complicated
    expression

        sqrt( ndop * [sum over k of (ndel - |k|)*corr(j)] )

    Here k counts how many rows separate two pixels within a given image column,
    while j = k*stride counts how many samples separate them; k and j can be
    negative, zero, or positive.  corr(j) is the correlation coefficient between
    noise values for rows that are j samples apart.

    In the code below, "var" and "covar" represent factors in the noise power
    variance and covariance -- to be precise, the factors that multiply the standard
    cw / 1-spb variance (k Tsys df)^2 / N_looks.  The expressions used here are
    asymptotically valid as fft length and (for short-code images) code length
    approach infinity.  Slight Doppler variations (for spb > 1) have been ignored on
    the assumption that Doppler frequencies of interest are << unaliased bandwidth.

    For the typical case where ndel >> spb and stride = 1, the rms summed noise is
    larger than the standard result sqrt(# pixels) * (rms single-pixel noise) by a
    factor of sqrt(sum over j of corr(j)).                                            */

double deldop_noisesum( int ndel, int ndop, int codemethod, int spb, int stride)
{
	int j, k, n_covar;
	double tmp, delaysum, var, covar, corr, summednoise;

	delaysum = ndel;  /* k = 0 contribution */

	if (codemethod == LONG_ORIG) {
		var = 1.0;
		n_covar = spb;
		for (j=stride; j<n_covar; j+=stride) {
			k = j/stride;
			if (k < ndel) {
				tmp = 1 - j/(1.0*spb);
				covar = tmp*tmp;
				corr = covar/var;
				delaysum += 2*(ndel - k)*corr;   /* includes terms with k < 0 */
			}
		}
	} else {
		tmp = (2 + 1.0/(spb*spb)) / 3;
		var = tmp*tmp;
		n_covar = 2*spb - 1;
		for (j=stride; j<n_covar; j+=stride) {
			k = j/stride;
			if (k < ndel) {
				tmp = (2*spb - j - 1) * (2*spb - j) * (2*spb - j + 1);
				if (j < spb)
					tmp -= 4 * (spb - j - 1) * (spb - j) * (spb - j + 1);
				tmp /= 6.0*spb*spb*spb;
				covar = tmp*tmp;
				corr = covar/var;
				delaysum += 2*(ndel - k)*corr;   /* includes terms with k < 0 */
			}
		}
	}

	summednoise = sqrt(ndop*delaysum);  /* in units of single-pixel sdev */
	return summednoise;
}


/*  svdfit_delcor is a modified version of the Numerical Recipes routine svdfit:
    it performs singular value decomposition, but instead of having the user
    provide a function which then computes the elements of the design matrix,
    the design matrix itself is provided as an argument.

    The "estimate_coeffs" routine needs to call this modified version in order
    to estimate the delay correction polynomial coefficients.  This application
    would require one function to compute design matrix elements for delay
    constraints (forcing each delay-Doppler frame's model centroid to match the
    data centroid in the delay dimension) and a different function for Doppler
    constraints (forcing each Doppler and delay-Doppler frame's model and data
    centroids to match in the Doppler dimension), so the NR version of svdfit
    can't be used.                                                               */

void svdfit_delcor( double **design, double y[], double sig[], int ndata, double a[],
		int ma, double **u, double **v, double w[], double *chisq)
{
	void svbksb(double **u, double w[], double **v, int m, int n, double b[],
			double x[]);
	void svdcmp(double **a, int m, int n, double w[], double **v);
	int j,i;
	double wmax,tmp,thresh,sum,*b;

	b = vector(1,ndata);
	for (i=1; i<=ndata; i++) {
		tmp = 1.0/sig[i];
		for (j=1; j<=ma; j++) u[i][j] = design[i][j]*tmp;
		b[i] = y[i]*tmp;
	}
	svdcmp(u,ndata,ma,w,v);
	wmax = 0.0;
	for (j=1; j<=ma; j++)
		if (w[j] > wmax) wmax = w[j];
	thresh = SVDTOL*wmax;
	for (j=1; j<=ma; j++)
		if (w[j] < thresh) w[j] = 0.0;
	svbksb(u,w,v,ndata,ma,b,a);
	*chisq = 0.0;
	for (i=1; i<=ndata; i++) {
		for (sum=0.0, j=1; j<=ma; j++) sum += a[j]*design[i][j];
		*chisq += (tmp = (y[i] - sum)/sig[i], tmp*tmp);
	}
	free_vector(b,1,ndata);
}


/*  deldopcentroid computes the fit centroid for a
    delay-Doppler frame, including any overflow region  */

void deldopcentroid( struct deldop_t *deldop, int frm,
		double *del_mean, double *dop_mean)
{
	int i, j;
	double xsec, delsum, dopsum;
	struct deldopfrm_t *frame;

	frame = &deldop->frame[frm];

	/*  Initialize sums to the overflow region values
      already computed by the pos2deldop routine     */

	xsec = frame->overflow_xsec;
	delsum = frame->overflow_xsec * frame->overflow_delmean;
	dopsum = frame->overflow_xsec * frame->overflow_dopmean;

	/*  Now add the contributions from the fit image itself  */

	for (i=1; i<=frame->ndel; i++)
		for (j=1; j<=frame->ndop; j++) {
			xsec += frame->fit[i][j];
			delsum += i*frame->fit[i][j];
			dopsum += j*frame->fit[i][j];
		}

	/*  Finish the computation to get the centroid  */

	*del_mean = delsum/xsec;
	*dop_mean = dopsum/xsec;
}


/*  dopcentroid computes the fit centroid for a
    Doppler frame, including any overflow region  */

void dopcentroid( struct doppler_t *doppler, int frm, double *dop_mean) 
{
	int j;
	double xsec, dopsum;
	struct dopfrm_t *frame;

	frame = &doppler->frame[frm];

	/*  Initialize sums to the overflow region values
      already computed by the pos2doppler routine    */

	xsec = frame->overflow_xsec;
	dopsum = frame->overflow_xsec * frame->overflow_dopmean;

	/*  Now add the contributions from the fit spectrum itself  */

	for (j=1; j<=frame->ndop; j++) {
		xsec += frame->fit[j];
		dopsum += j*frame->fit[j];
	}

	/*  Finish the computation to get the centroid  */

	*dop_mean = dopsum/xsec;
}

#undef MAXCONSTRAINTS
#undef MAXCOEFFS
#undef SVDTOL
