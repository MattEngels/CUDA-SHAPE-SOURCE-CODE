/*****************************************************************************************
                                                                           show_moments.c

Display the volume, center-of-mass (COM) displacement, and inertia tensor of a model, plus
the surface area, equivalent diameter, and extent in each of the three body coordinates.

For a one-component model, also display information on the principal axes and on the
dynamically equivalent equal-volume ellipsoid (DEEVE).  A significant (and fairly
tedious) part of this task is to determine the epoch(s) at which the positive side of the
longest principal axis lies in the plane of sky while approaching Earth.

For NPA rotators, compute and display the behavior of the Euler angles defined by
Samarasinha & A'Hearn (1991, Icarus, 93, 194-225), describing the changing orientation of
the principal axes as viewed in an inertial reference frame whose +Z-axis points along the
angular momentum vector.

In order to accomplish these things, invoke shape with action = "moments" and give the
name of a standalone ephemeris file as the "pa_ephfile" parameter.  This file is needed to
get distances (for light-travel-time corrections) and plane-of-sky motions (which
contribute to the target's instantaneous apparent spin vector).  Specify the file's format
as the "pa_ephformat" parameter:

    pa_ephformat = horizons    (format in obs files) (default)
    pa_ephformat = datataking  (format used at telescope)

Time gaps in the ephemeris file, however small or large, are handled via linear
interpolation by function ephem2mat.

The angle and spin offsets are set to zero, unless the "pa_angleoff" and/or "pa_omegaoff"
parameter *vectors* are specified (in degrees and degrees/day, respectively).  There
should seldom if ever be any need to specify them.

This routine will search for epochs over the entire time range of the ephemeris file 
(other than a fraction of a time increment at the end of the file), unless you limit the
range by specifying the "pa_startepoch" and/or "pa_stopepoch" parameters (in Julian days).
This could help to avoid a huge output listing for a rapid rotator.

Stepping through the ephemeris in constant time increments, show_moments can find at most
one epoch per increment; hence the erratic motion of an NPA rotator could lead us to miss
some epochs.  If for some odd reason it is essential to find EVERY epoch when an NPA
rotator's long principal axis lies in the plane of the sky while approaching us, recompile
using a very small value (in degrees) for the ANGLEINCR_DEG parameter defined below.  Be
prepared to wait a while for your output....


- - - - - - - - - - - - - - - -  A note on coordinate systems - - - - - - - - - - - - - -


Most of shape deals in four Cartesian coordinate systems:

1) "o" = observer coordinates:
             x westward in the plane of the sky  (-RA)
             y northward in the plane of the sky (+DEC)
             z from the target towards Earth     (-LOS)

2) "s" = source (or "sun") coordinates:
             like observer coordinates, except that we imagine observing the asteroid from
             the Sun rather than from Earth (used only for lightcurve datasets)

3) "e" = ecliptic coordinates:
             like observer coordinates, except that x and y are oriented in the plane of
             the sky along ecliptic rather than equatorial coordinates
             (x = -LAMBDA and y = +BETA)

4) "a" = body-fixed ("asteroid") coordinates:

             For ellipsoid models, these are the three symmetry axes.

             After switching to a harmonic or vertex model, body coordinates cease to have
             any special physical significance, except that
                a) for PA rotators, z will always be the spin axis;
                b) for NPA rotators, the spin state is evolved (via Euler's equations) on
                   the assumption that the three body-fixed axes coincide with the model's
                   principal axes;
                c) if applying significant "inertiadev" penalties, the three
                   body-coordinate axes will always be *close* to the model's principal
                   axes as computed on the assumption of uniform density.


In this module we also use a fifth system:

5) "p" = principal-axis coordinates

             These of course are also body-fixed, but for clarity the term "body
             coordinates" in comments will always refer to the "a" coordinate system
             outlined above.

             The "p" coordinate system is derived from the model's inertia tensor as
             computed under the assumption of uniform density.

             As mentioned above, the "inertiadev" penalty forces the "p" coordinate system
             to remain close to the "a" system; that is, the coordinate tranformation
             matrix ap is close to the identity matrix, and the corresponding Euler angles
             obey theta ~ 0 and psi ~ -phi.


- - - - - - - - - - - - - - - End note on coordinate systems - - - - - - - - - - - - - - -


Modified 2015 May 10 by CM:
    For NPA rotators, add computation of the Samarasinha & A'Hearn (1991) Euler angles as
        functions of time, describing the changing orientation of the principal axes
        relative to the inertial coordinate system defined by those authors

Modified 2014 February 10 by CM:
    Add "ilaw" argument to the radlaw and apply_photo routines to implement multiple radar
        and optical scattering laws (just set it to zero, thus using the first radar or
        optical scattering law)

Modified 2013 August 28 by CM:
    Display minimum and maximum side (edge) lengths

Modified 2013 June 25 by CM:
    Allow image scaling to be governed by the "optposmax" parameter when "pa_scatlaw" is
        set to "lambertian"
    When the "optposmax" or "radposmax" parameter is nonzero (for a optical/lambertian or
        radar scattering law, respectively), print the maximum pixel value of each image
        to the screen

Modified 2013 May 20 by CM:
    Implement ovoid components (forgot to upload this change until 2013 June 7)

Modified 2012 March 24 by CM:
    Initialize variable to avoid compilation warning

Modified 2012 March 5 by CM:
    Implement "pa_highlight" parameter to highlight selected facets in the output images
    Implement "pa_bodycoords" parameter to orient the six output images along body-fixed
        coordinates rather than along the principal axes

Modified 2011 August 12 by CM:
    Implement spin impulses

Modified 2011 August 10 by CM:
    Bug fix for axis orientations in POS images: each view was along the correct axis
        but the two axes in the plane of the image were sometimes switched.  Introduce
        the "pa_to_obs" routine to handle this, and give a screen warning if an axis
        has to be given a nonstandard (backwards) orientation in the images.

Modified 2009 July 3 by CM:
    Accidentally left out a line return after the epoch search output

Modified 2009 July 1 by CM:
    For multiple-component models, undo the preceding change and
        display the same calculations and output the same images as
        for one-component models; since the "realize_mod" routine now
        flags facets that lie interior to the model, these calculations
        (mean and median side lengths, surface area, volume, COM,
        inertia tensor, principal moments of inertia, principal axes)
        are now at least moderately accurate

Modified 2009 June 28 by CM:
    For multiple-component models, output six pgm images showing the
        the model from each end of the three body-fixed axes rather
        than no images at all

Modified 2009 April 3 by CM:
    For NPA rotators, make a small correction to the angular momentum
        computation: must assume that principal-axis coordinates are
        identical to body-fixed coordinates, since this is what the
        inteuler routine assumes when evolving the spin state

Modified 2009 January 7 by CM:
    Fix small bug in changes made 2008 December 15
        (no YORP contribution to *initial* angular momentum)

Modified 2008 December 15 by CM:
    For NPA rotators display the direction of the angular momentum vector
        (ecliptic coordinates), the three ratios (L^2 / 2*E*inertia)
        that indicate whether this is a long-axis or short-axis mode,
        and the periods and amplitudes of the motion as viewed in an
        inertial reference frame

Modified 2008 March 8 by CM:
    Display the angle between each principal axis and the
        corresponding body-fixed axis

Modified 2008 January 23 by CM:
    Correct bug in handling light-time correction for epoch at which
        the positive end of the longest principal axis is in the
        plane of the sky while approaching Earth.  Both the epoch
        of this event and the epoch at which it would be viewed on
        Earth are now displayed on the screen, regardless of the
        value of the perform_ltc parameter.

Modified 2007 August 10 by CM:
    Initialize uninitialized variables

Modified 2007 August 7 by CM:
    Fix bug introduced 2007 August 4: forgot to allocate memory for
        the pos.body and pos.comp matrices

Modified 2007 August 4 by CM:
    Add orbit_offset and body arguments to posvis routine and remove
        facet argument
    Add c (component) argument to radlaw routine
    Add body argument to apply_photo routine
    Add comp matrix to POS frames

Modified 2007 February 21 by CM:
    Fix an error in computing the DEEVE radii: fortunately this error
        was canceling itself out and hence didn't mess up any output

Modified 2006 October 1 by CM:
    Add "intensityfactor" argument to apply_photo

Modified 2006 September 1 by CM and MCN:
    When "mark_unseen" parameter is turned on, add checks that facet
        number pos.f[i][j] is nonnegative

Modified 2006 June 21 by CM:
    For POS renderings, change res to km_per_pixel

Modified 2005 October 28 by CM:
    Fix bug in write_pa_views when "mark_unseen" parameter is set:
        the yellow-shaded regions for all six views were identical.
        (Must assign POS pixel values/colors before calling posvis
        for the next image, since posvis changes the mapping from
        model facets to POS pixels.)

Modified 2005 September 2 by CM:
    Place all six PA views on the same brightness scale

Modified 2005 August 10 by CM:
    Fix bug in write_pa_views (for negative end of axis when
        "mark_unseen" parameter is set)

Modified 2005 August 1 by CM:
    Implement the "pa_scatlaw" parameter, which determines whether
        the six PA views will use the optical scattering law, the
        radar scattering law, or (default) the Lambert law

Modified 2005 June 23 by CM:
    If the "mark_unseen" parameter is turned on, use color to mark
        regions which are always hidden/shadowed from Earth's
        view.  In this case the output images are in ppm format
        rather than the usual pgm format.

Modified 2005 April 23 by CM:
    Slightly modify screen output

Modified 2005 March 19 by CM:
    Add ANGLETOL_DEG parameter and angletol variable:
        When determining the epochs at which the positive end of the
        long principal axis lies in the plane of the sky while
        approaching Earth, check that this axis is no more than
        angletol radians out of the plane of the sky at the epoch
        returned by bisection routine rtbis

Modified 2005 February 9 by CM:
    Use scientific notation to display some very large and small
        numbers

Modified 2004 September 22 by CM:
    Compute the mean and median side (edge) lengths

Modified 2004 May 28 by CM:
    Print epoch to the screen for the special case where we
        land directly on it without having to do bisection

Modified 2004 April 26 by CM:
    Don't compute the DEEVE diameters using the spin.inertia
        principal moments unless the model is for a NPA rotator:
        Use the uniform-density principal moments of inertia
        for PA rotators.

Modified 2004 March 27 by CM:
    Move code for diagonalizing inertia tensor to a separate
        routine (diag_inertia.c)

Modified 2004 March 23 by CM:
    For one-component models, output six pgm images showing
        the model from each end of the three principal axes

Modified 2004 February 23 by CM:
    Replaced LTC variable by perform_ltc parameter for
        computing 1-way light-time correction

Modified 2003 October 27 by CM:
    List DEEVE diameters computed using the actual (spin)
        principal moments, not just using the uniform-density
        moments

Modified 2003 April 21 by CM:
    List ratios of principal moments of inertia relative to the
        maximum principal moment rather than the minimum
        principal moment

Written 2003 April 21 by CM, based in part on ../wfobj/wfmom.c
*****************************************************************************************/

#include "head.h"

#define ANGLEINCR_DEG 5.0
#define ANGLETOL_DEG 0.5
#define EPOCHTOL 1.0e-7

static struct par_t *spar;
static struct mod_t *smod;
static struct ephem_t *sephem;
static double n_longpa[3], t, sn0_static;


/*  Define global variables that are available
    to functions phi_dot_LAM and phi_dot_SAM    */     

static double min_truemoment, int_truemoment, max_truemoment, angmom_mag,
              L2_over_2E, kc2;


double phi_dot_LAM(double tau);
double phi_dot_SAM(double tau);
double longpa_losproj(double t0);
double get_tau0( double sn0, double cn0, double dn0, double P_tau);
double delta_sn( double tau0);
void write_pa_views( struct par_t *par, struct mod_t *mod, double ap[3][3],
                     int longpa_index, int shortpa_index);
void pa_to_obs( double op[3][3], int axis, int negflag, int longpa_index, int shortpa_index);


void show_moments( struct par_t *par, struct mod_t *mod)
{
  const char *comptype[4]   = {"ellipsoid", "ovoid", "harmonic", "vertex"};
  const char *coordname[3]  = {"X", "Y", "Z"};
  const char *monthName[12] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};

  int ncomp, c, i, j, longpa_index, shortpa_index, n_iter, max_iter, n_zeros, year, mon,
      day, hour, min, sec, s, ns, ns_tot, v1, v2, year0, mon0, day0, hour0, min0, sec0,
      intpa_index, LAM, f1, f2, longsign, intsign, shortsign;
  double min_extent[3], max_extent[3], vol, com[3], inertia[3][3], a[3], pmoment[3],
         ap[3][3], phi, theta, psi, max_moment, scale, anglesave[3], spin_magnitude,
         epoch, epoch_incr, aval, bval, bestepoch, bestroot, DEEVE_diam[3],
         DEEVE_diam_actual[3], startepoch, stopepoch, disp[3], mean_sidelength,
         median_sidelength, angleincr, angletol, bestepoch0, intspin[3], angmom[3],
         angmom_hat[3], energy, ae[3][3], angmom_long, angmom_lat, X_hat_ecl[3], vecmag,
         Y_hat[3], X_hat[3], long_hat[3], theta_dot_vec[3], phi0, k2, K_cel, period_psi,
         psi_dot_min, psi_dot_max, upper_limit, phi_dot_avg, period_phi_avg, phi_dot_min,
         phi_dot_max, period_theta, theta_min, theta_max, ampl_psi, ampl_theta, tnorm, sn0,
         cn0, dn0, tau0, tau, tau_incr, tau_prev, sn, cn, dn, dist, oe[3][3], se[3][3],
         solar_phase, solar_azimuth, orbspin[3], epoch_ltc;
  double *sidelengths;
  struct ephem_t ephem;

  /*  Set static global pointers which are used by function longpa_losproj
      below to be compatible with Numerical Recipes in C routines.          */

  spar = par;
  smod = mod;
  sephem = &ephem;

  /*  Convert angle-related parameters from degrees to radians  */

  angleincr = ANGLEINCR_DEG*D2R;
  angletol = ANGLETOL_DEG*D2R;

  /*  Read in the standalone ephemeris file  */

  read_ephem( &(*par), &ephem);

  /*  To get the epoch increment for the search to be performed later,
      look at the spin vector given in the mod file, take the magnitude,
      and find the time needed for angleincr degrees of rotation at this rate  */

  spin_magnitude = sqrt(  (*mod).spin.omega[0].val*(*mod).spin.omega[0].val
                        + (*mod).spin.omega[1].val*(*mod).spin.omega[1].val
                        + (*mod).spin.omega[2].val*(*mod).spin.omega[2].val );
  epoch_incr = angleincr/spin_magnitude;            /* days */

  /*  Set the epoch range for the search.  If no range was specified
      in the parameter file, use the limits in the ephemeris file,
      skipping a bit at the start so that we can correct for one-way
      light-travel time.                                              */

  startepoch = ephem.pnt[0].t + DAYSPERAU*ephem.pnt[0].dist + 1.0e-6;
  startepoch = MAX( startepoch, (*par).pa_startepoch);
  stopepoch = MIN( ephem.pnt[ephem.n - 1].t, (*par).pa_stopepoch);
  max_iter = (int)( (stopepoch - 1.0e-6 - startepoch)/epoch_incr );
  stopepoch = startepoch + epoch_incr*max_iter;

  /*  Display the number of model components and the number of
      vertices and facets in each component's vertex realization  */

  ncomp = (*mod).shape.ncomp;
  printf("\n*********************************************************\n\n");
  if (ncomp == 1) {
      printf("%d-component %s model realized with %d vertices and %d facets\n\n",
             ncomp, comptype[(*mod).shape.comp[0].type - 1],
             (*mod).shape.comp[0].real.nv, (*mod).shape.comp[0].real.nf);
  } else {
      printf("%d-component model (calculations are only moderately accurate):\n\n",
             ncomp);
      for (c=0; c<ncomp; c++)
        printf("      %s component %d realized with %d vertices and %d facets\n",
               comptype[(*mod).shape.comp[c].type - 1], c,
               (*mod).shape.comp[c].real.nv, (*mod).shape.comp[c].real.nf);
      printf("\n\n");
  }

  /*  Display the model's maximum extent in each dimension  */

  for (j=0; j<=2; j++) {
    min_extent[j] = HUGENUMBER;
    max_extent[j] = -HUGENUMBER;
  }
  for (c=0; c<ncomp; c++)
    for (i=0; i<(*mod).shape.comp[c].real.nv; i++)
      for (j=0; j<=2; j++) {
        min_extent[j] = MIN( min_extent[j], (*mod).shape.comp[c].real.v[i].x[j]);
        max_extent[j] = MAX( max_extent[j], (*mod).shape.comp[c].real.v[i].x[j]);
      }
  for (j=0; j<=2; j++)
    printf("%s: D = %f km (%f km to %f km)\n", coordname[j],
           max_extent[j]-min_extent[j], min_extent[j], max_extent[j]);

  /*  Display the mininum, median, maximum, and mean side (edge) lengths  */

  ns_tot = 0;
  for (c=0; c<ncomp; c++) {
    ns = (*mod).shape.comp[c].real.ns;
    for (s=0; s<ns; s++) {
      f1 = (*mod).shape.comp[c].real.s[s].f[0];
      f2 = (*mod).shape.comp[c].real.s[s].f[1];
      if ((*mod).shape.comp[c].real.f[f1].act || (*mod).shape.comp[c].real.f[f2].act)
        ns_tot++;
    }
  }
  sidelengths = vector(1, ns_tot);
  i = 0;
  mean_sidelength = 0.0;
  for (c=0; c<ncomp; c++) {
    ns = (*mod).shape.comp[c].real.ns;
    for (s=0; s<ns; s++) {
      f1 = (*mod).shape.comp[c].real.s[s].f[0];
      f2 = (*mod).shape.comp[c].real.s[s].f[1];
      if ((*mod).shape.comp[c].real.f[f1].act || (*mod).shape.comp[c].real.f[f2].act) {
        v1 = (*mod).shape.comp[c].real.s[s].v[0];
        v2 = (*mod).shape.comp[c].real.s[s].v[1];
        for (j=0; j<=2; j++)
          disp[j] = (*mod).shape.comp[c].real.v[v2].x[j] -
                    (*mod).shape.comp[c].real.v[v1].x[j];
        sidelengths[++i] = vecnorm(disp);
        mean_sidelength += sidelengths[i];
      }
    }
  }
  mean_sidelength /= ns_tot;
  hpsort(ns_tot, sidelengths);
  if (2*(ns_tot/2) == ns_tot)
    median_sidelength = (sidelengths[ns_tot/2] + sidelengths[ns_tot/2 + 1])/2;
  else
    median_sidelength = sidelengths[(ns_tot+1)/2];
  printf("\nedge lengths: mean = %f km    min = %f km    median = %f km    max = %f km\n",
         mean_sidelength, sidelengths[1], median_sidelength, sidelengths[ns_tot]);
  fflush(stdout);
  free_vector(sidelengths, 1, ns_tot);

  /*  Display the volume, the COM displacement and inertia tensor
      (computed assuming uniform unit density), the surface area,
      and the equivalent diameter.                                 */

  vol = (*mod).shape.volume;
  for (i=0; i<=2; i++) {
    com[i] = (*mod).shape.com[i];
    for (j=0; j<=2; j++)
      inertia[i][j] = (*mod).shape.inertia[i][j];
  }
  printf("\narea    = %12.6e square km\n", (*mod).shape.area);
  printf("volume  = %12.6e cubic km\n", vol);
  printf("equiv D = %f km\n\n", 2*pow(3*vol/(4*PIE), 1/3.0));
  printf("com     = %f %f %f km\n\n", com[0], com[1], com[2]);
  printf("inertia = %13.6e %13.6e %13.6e\n", inertia[0][0], inertia[0][1], inertia[0][2]);
  printf("          %13.6e %13.6e %13.6e\n", inertia[0][1], inertia[1][1], inertia[1][2]);
  printf("          %13.6e %13.6e %13.6e\n", inertia[0][2], inertia[1][2], inertia[2][2]);
  fflush(stdout);

  /*  Display the properties of the principal axes
      and of the dynamically equivalent equal-volume ellipsoid (DEEVE)  */

  /*  Diagonalize the inertia tensor to get pmoments, the principal moments
      of inertia, and ap, the transformation matrix taking us from
      principal-axis to body-fixed coordinates                               */

  diag_inertia( inertia, pmoment, ap);

  /*  For each principal axis, display its direction cosines w.r.t. body
      coordinates, the moment of inertia about that axis, and the ratio of
      that moment to the maximum principal moment.                          */

  max_moment = MAX( pmoment[0], pmoment[1]);
  max_moment = MAX( max_moment, pmoment[2]);
  printf("\nDirection cosines (w.r.t. body coordinates) of principal axes are:\n\n");
  for (i=0; i<=2; i++) {
    printf("PA%d: %9.6f %9.6f %9.6f : ", i+1, ap[0][i], ap[1][i], ap[2][i]);
    printf("moment = %12.6e [%f]\n", pmoment[i], pmoment[i]/max_moment);
  }
  printf("\n(The 3x3 matrix at above left takes us from body -> PA coords;");
  printf("\n                its transpose takes us from PA -> body coords.)\n");

  /*  Compute the Euler angles corresponding to transformation matrix ap;
      also display the Euler angles for transformations in the opposite
      direction (body coordinates --> principal axis coordinates).           */

  mat2euler( ap, &phi, &theta, &psi);
  printf("\nEuler angles for PA -> body  =  %11.6f %11.6f %11.6f deg\n",
         phi*R2D, theta*R2D, psi*R2D);
  printf("Euler angles for body -> PA  =  %11.6f %11.6f %11.6f deg\n",
         -psi*R2D, -theta*R2D, -phi*R2D);

  /*  Compute the angle between each principal axis
      and the corresponding body-fixed axis          */

  printf("\nAngle between PA1 & x, PA2 & y, PA3 & z = %10.6f %10.6f %10.6f deg\n",
         acos(ap[0][0])*R2D, acos(ap[1][1])*R2D, acos(ap[2][2])*R2D);

  /*  Given a unit-density ellipsoid with volume V_ell and axis radii a, b, c
      along x, y, and z, respectively, the moment of inertia about the x-axis
      is (V_ell/5)*(b^2 + c^2), and similarly for the other two axes.  Hence
      the three parameters below are the three radii of an ellipsoid whose
      principal moments of inertia are the same as our model's.                 */

  a[0] = sqrt( (5/vol) * (-pmoment[0] + pmoment[1] + pmoment[2]) / 2 );
  a[1] = sqrt( (5/vol) * ( pmoment[0] - pmoment[1] + pmoment[2]) / 2 );
  a[2] = sqrt( (5/vol) * ( pmoment[0] + pmoment[1] - pmoment[2]) / 2 );

  /*  Take those inertia ellipsoid radii, multiply them by the cube root of
      ( V_model / V_ell ), and double them: These are the DEEVE diameters.   */

  scale = 2*pow( 3*vol/(4*PIE*a[0]*a[1]*a[2]), 1.0/3.0);
  for (i=0; i<=2; i++)
    DEEVE_diam[i] = scale*a[i];
  printf("\nDEEVE has diameters %f %f %f km\n",
         DEEVE_diam[0], DEEVE_diam[1], DEEVE_diam[2]);

  /*  PA rotators:
          Treat the principal moments computed assuming uniform density as the
          true principal moments, and determine which principal axes correspond
          to the largest and smallest principal moments
      NPA rotators:
          Treat the "spin.inertia" principal moments used as parameters in Euler's
          equations as the model's true principal moments of inertia, and determine
          which principal axes correspond to the largest and smallest principal
          moments.  For comparison with the results already displayed, show the
          ratios of these true principal moments, and then show the DEEVE diameters
          computed using the true principal moments.                                 */

  if ((*mod).spin.pa) {
      max_truemoment = max_moment;
      if (pmoment[0] == max_truemoment)
        shortpa_index = 0;
      else if (pmoment[1] == max_truemoment)
        shortpa_index = 1;
      else
        shortpa_index = 2;
      min_truemoment = pmoment[0];
      longpa_index = 0;
      if (pmoment[1] < min_truemoment) {
        min_truemoment = pmoment[1];
        longpa_index = 1;
      }
      if (pmoment[2] < min_truemoment) {
        min_truemoment = pmoment[2];
        longpa_index = 2;
      }
      for (i=0; i<=2; i++)
        if (i != shortpa_index && i != longpa_index) {
          int_truemoment = pmoment[i];
          intpa_index = i;
        }
  } else {
      max_truemoment = (*mod).spin.inertia[0].val;
      shortpa_index = 0;
      if ((*mod).spin.inertia[1].val > max_truemoment) {
        max_truemoment = (*mod).spin.inertia[1].val;
        shortpa_index = 1;
      }
      if ((*mod).spin.inertia[2].val > max_truemoment) {
        max_truemoment = (*mod).spin.inertia[2].val;
        shortpa_index = 2;
      }
      min_truemoment = (*mod).spin.inertia[0].val;
      longpa_index = 0;
      if ((*mod).spin.inertia[1].val < min_truemoment) {
        min_truemoment = (*mod).spin.inertia[1].val;
        longpa_index = 1;
      }
      if ((*mod).spin.inertia[2].val < min_truemoment) {
        min_truemoment = (*mod).spin.inertia[2].val;
        longpa_index = 2;
      }
      for (i=0; i<=2; i++)
        if (i != shortpa_index && i != longpa_index) {
          int_truemoment = (*mod).spin.inertia[i].val;
          intpa_index = i;
        }
      printf("\nAll of the above assumes uniform (unit) density.  For comparison,\n");
      printf("the ratios of the model's actual principal moments of inertia are\n\n");
      for (i=0; i<=2; i++)
        printf("      I%d/I%d = %f",
               i+1, shortpa_index+1, (*mod).spin.inertia[i].val/max_truemoment);
      printf("\n");
      a[0] = sqrt( (5/vol) * (-(*mod).spin.inertia[0].val + (*mod).spin.inertia[1].val
                                                          + (*mod).spin.inertia[2].val) );
      a[1] = sqrt( (5/vol) * ( (*mod).spin.inertia[0].val - (*mod).spin.inertia[1].val
                                                          + (*mod).spin.inertia[2].val) );
      a[2] = sqrt( (5/vol) * ( (*mod).spin.inertia[0].val + (*mod).spin.inertia[1].val
                                                          - (*mod).spin.inertia[2].val) );
      scale = 2*pow( 3*vol/(4*PIE*a[0]*a[1]*a[2]), 1.0/3.0);
      for (i=0; i<=2; i++)
        DEEVE_diam_actual[i] = scale*a[i];
      printf("\n-- and the corresponding DEEVE diameters are\n");
      printf("\n                    %f %f %f km\n",
             DEEVE_diam_actual[0], DEEVE_diam_actual[1], DEEVE_diam_actual[2]);
  }
  fflush(stdout);

  /*  For NPA rotators, work out the angular momentum and energy
      to see if this is a long-axis mode or a short-axis mode.    */

  if (!(*mod).spin.pa) {

    /*  First, get the spin state at the spin-state reference epoch:
        intspin, the intrinsic (sidereal) spin vector in body-fixed coordinates;
        and ae, the ecliptic-to-body-fixed coordinate transformation matrix.      */

    euler2mat( ae, (*mod).spin.angle[0].val, (*mod).spin.angle[1].val, (*mod).spin.angle[2].val);
    for (i=0; i<=2; i++)
      intspin[i] = (*mod).spin.omega[i].val;

    /*  Add the spin offsets to intspin  */

    for (i=0; i<=2; i++)
      intspin[i] += (*par).pa_omegaoff[i];

    /*  Compute the angular momentum vector in PA coordinates (= body-fixed
        coordinates, see below), then transform it to ecliptic coordinates

        For NPA rotators, the inteuler routine evolves the spin state (via Euler's equations)
        on the assumption that PA coordinates are IDENTICAL to body-fixed coordinates.  Thus
        in the code below we should make the same assumption.  That is, we should *not* use
        the "ap" matrix to transform intspin from body-fixed to PA coordinates, or angmom
        from PA to body-fixed coordinates: if ap is not the identity matrix, this simply
        implies nonuniform density, given that ap was computed assuming uniform density.       */

    for (i=0; i<=2; i++)
      angmom[i] = (*mod).spin.inertia[i].val * intspin[i];
    cotrans( angmom, ae, angmom, -1);

    /*  Compute the magnitude of angular momentum L, a unit vector pointing along the
        angular momentum vector (in ecliptic coordinates), and the ecliptic latitude
        and longitude to which the angular momentum vector points                      */

    angmom_mag = sqrt(angmom[0]*angmom[0] + angmom[1]*angmom[1] + angmom[2]*angmom[2]);
    for (i=0; i<=2; i++)
      angmom_hat[i] = angmom[i]/angmom_mag;
    angmom_lat = asin(angmom_hat[2])*R2D;
    angmom_long = atan2(angmom_hat[1], angmom_hat[0])*R2D;
    if (angmom_long < 0.0)
      angmom_long += 360.0;
    printf("\nAngular momentum L points to (lambda, beta) = (%f, %+f) deg\n\n",
           angmom_long, angmom_lat);

    /*  Compute the energy E and the ratio L^2 / 2E (where L = angular momentum)  */

    energy = 0.5*(   (*mod).spin.inertia[0].val * intspin[0]*intspin[0]
                   + (*mod).spin.inertia[1].val * intspin[1]*intspin[1]
                   + (*mod).spin.inertia[2].val * intspin[2]*intspin[2] );
    L2_over_2E = (angmom_mag*angmom_mag)/(2*energy);

    /*  Compute the three ratios L^2 / (2*E*I_i)  */

    printf("L^2 / (2*energy*actual_inertia) = %f %f %f",
           L2_over_2E/min_truemoment, L2_over_2E/int_truemoment,
           L2_over_2E/max_truemoment);

    /*  Compute the X-hat and Y-hat unit vectors of the inertial XYZ coordinate
        system of Samarasinha & A'Hearn (1991), expressed in ecliptic coordinates.
        By definition, Z-hat points along angular momentum L.  To fully constrain
        the X and Y axes we add the requirement that the ecliptic +x axis -- the
        vernal equinox -- must lie within the +X half of the XZ plane.              */

    X_hat_ecl[0] = 1.0;
    X_hat_ecl[1] = X_hat_ecl[2] = 0.0;
    vecmag = cross( Y_hat, angmom_hat, X_hat_ecl);
    if (vecmag < SMALLVAL) {
        Y_hat[1] = 1.0;
        Y_hat[0] = Y_hat[2] = 0.0;
    } else {
        for (i=0; i<=2; i++)
          Y_hat[i] /= vecmag;
    }
    cross( X_hat, Y_hat, angmom_hat);

    /*  This is a long-axis mode if L^2 < 2*E*I2  */

    LAM = (L2_over_2E < int_truemoment) ? 1 : 0;
    if (LAM)
      printf("  (long-axis mode)\n");
    else
      printf("  (short-axis mode)\n");

    /*  Do the necessary computations of periods and amplitudes as viewed
        in an inertial reference frame, depending on whether this is a
        long-axis or short-axis mode; see Samarasinha & A'Hearn 1991
        (Icarus 93, 194-225) (also pp. 116-120 of vol. 1 of Landau &
        Lifshitz, 3rd edition)                                             */

    printf("\nPeriods, amplitudes, and values vs. time of Euler angles\n"
           "as viewed in the inertial reference frame of Samarasinha & A'Hearn (1991):\n\n");

    if (LAM) {

        /*  long-axis mode (LAM)  */

        /*  Compute constant (unit = days) used to normalize time (eq. A31)  */

        tnorm = sqrt( min_truemoment*int_truemoment*max_truemoment
                           / (2*energy*(int_truemoment - min_truemoment)
                                      *(max_truemoment - L2_over_2E    ) ) );

        /*  Compute parameter k^2 (eq. A32)  */

        k2 = (max_truemoment - int_truemoment)*(L2_over_2E - min_truemoment)
             / ((int_truemoment - min_truemoment)*(max_truemoment - L2_over_2E));
        kc2 = 1 - k2;

        /*  Compute the period of rotational motion about the long axis
            (eq. A45); in body-fixed coordinates, this is the period of
            the spin vector's motion (Landau & Lifshitz, vol. 1, p. 118).
            K_cel is the complete elliptic integral of the first kind.     */

        K_cel = ellf(PIOVER2, sqrt(k2));
        period_psi = 4*tnorm*K_cel;
        psi_dot_min = (angmom_mag/int_truemoment)
                      * sqrt((int_truemoment - min_truemoment)
                             * (int_truemoment - L2_over_2E)
                             / (L2_over_2E*min_truemoment));
        psi_dot_max = (angmom_mag/max_truemoment)
                      * sqrt((max_truemoment - min_truemoment)
                             * (max_truemoment - L2_over_2E)
                             / (L2_over_2E*min_truemoment));

        /*  Working from the instantaneous rate at which the long axis
            precesses about the angular momentum vector (eq. A38), a
            rate which is periodic with period = period_psi/2, integrate
            over one such period to get the average precession rate, and
            from this define an average precession period.

            upper_limit is the upper integration limit
                  = (period_psi/2)
                    * sqrt( 2*energy*(int_truemoment - min_truemoment)
                                    *(max_truemoment - L2_over_2E    )
                             / (min_truemoment*int_truemoment*max_truemoment) )
                  = (period_psi/2) * (1/tnorm)
                  = 2*K_cel
            -- see eq. A31 and the text following eq. A48.                       */

        upper_limit = 2*K_cel;
        phi_dot_avg = (1/upper_limit) * qsimp(phi_dot_LAM, 0.0, upper_limit);
        period_phi_avg = TWOPI/phi_dot_avg;
        phi_dot_min = angmom_mag/max_truemoment;
        phi_dot_max = angmom_mag/int_truemoment;

        /*  For long-axis modes the long axis nutates (nods) with period
            = period_psi/2 (Samarasinha & A'Hearn 1991, p. 199) between
            fixed angular limits (eqs. A39, A40).                         */

        period_theta = period_psi/2;
        theta_min = acos( sqrt( min_truemoment*(max_truemoment - L2_over_2E)
                                / (L2_over_2E*(max_truemoment - min_truemoment)) ) );
        theta_max = acos( sqrt( min_truemoment*(int_truemoment - L2_over_2E)
                                / (L2_over_2E*(int_truemoment - min_truemoment)) ) );

        printf("      Rotation about the long axis:\n\n");
        printf("            P_psi   = %G days\n\n", period_psi);
        printf("            %G deg/day <= d(psi)/dt <= %G deg/day\n\n",
               psi_dot_min*R2D, psi_dot_max*R2D);
        printf("      Precession of the long axis about the angular momentum vector:\n\n");
        printf("            <P_phi> = %G days\n\n", period_phi_avg);
        printf("            %G deg/day <= d(phi)/dt <= %G deg/day\n\n",
               phi_dot_min*R2D, phi_dot_max*R2D);
        printf("      Nutation of the long axis:\n\n");
        printf("            P_theta = %G days\n\n", period_theta);
        printf("            %G deg <= theta <= %G deg\n\n", theta_min*R2D, theta_max*R2D);

        /*  Samarasinha & A'Hearn (1991) define their coordinates such that
            0 <= theta < pi/2 for a long-axis mode (see eqs. A39-A40 and the text that
            follows), where theta is the second Euler angle.  Thus when we specify the
            spin component along the long axis we must choose the "positive" side of
            the long axis to be the side that makes an acute angle with the angular
            momentum vector (the inertial +Z-axis): see Fig. A1.  This may require
            negating the corresponding body-fixed spin component.                       */

        longsign = (intspin[longpa_index] < 0.0) ? -1 : 1;

        /*  Choose the appropriate sides of the other two body-fixed axes so that
            (+long, +intermediate, +short) forms a right-handed triple (see Fig. A1)  */

        if ((longpa_index + 1) % 3 == intpa_index)
          intsign = shortsign = longsign;
        else
          intsign = shortsign = -longsign;

        /*  Compute dimensionless time tau at spin reference epoch t0 by using the
            "get_tau0" routine to search for the value that yields (via eqs. A33-A35)
            three spin components that match the three spin components specified in
            the mod file for epoch t0                                                  */

        dn0 = longsign*intspin[longpa_index]
              / sqrt(2*energy*(max_truemoment - L2_over_2E)
                     / (min_truemoment*(max_truemoment - min_truemoment)));
        sn0 = intsign*intspin[intpa_index]
              / sqrt(2*energy*(L2_over_2E - min_truemoment)
                     / (int_truemoment*(int_truemoment - min_truemoment)));
        cn0 = shortsign*intspin[shortpa_index]
              / sqrt(2*energy*(L2_over_2E - min_truemoment)
                     / (max_truemoment*(max_truemoment - min_truemoment)));
        tau0 = get_tau0( sn0, cn0, dn0, 4*K_cel);

        /*  Compute the unit vector pointing perpendicular to both angular momentum L
            and the long axis, in the direction of positive d(theta)/dt where theta
            is the second Euler angle; see Fig. A1 of Samarasinha & A'Hearn (1991).
            Then use this to get first Euler angle phi at spin reference epoch t0.     */

        for (i=0; i<=2; i++)
          long_hat[i] = 0.0;
        long_hat[longpa_index] = (double) longsign;
        cotrans( long_hat, ae, long_hat, -1);
        cross( theta_dot_vec, angmom_hat, long_hat);
        phi0 = atan2( dot( theta_dot_vec, Y_hat), dot( theta_dot_vec, X_hat));

        /*  Compute Euler angles as a function of time  */

        tau_incr = epoch_incr/tnorm;
        tau_prev = tau0;
        tau = tau0 + (startepoch - (*mod).spin.t0)/tnorm;
        phi = phi0;
        for (i=0; i<=max_iter; i++) {

          /*  Integrate the time derivative of first Euler angle phi to obtain the
              increment in phi during this time interval (eqs. A38 and A21); then
              use this increment to get phi itself, given that we initialized phi
              to have its value (determined earlier) at spin reference epoch t0     */

          phi += tnorm * qsimp(phi_dot_LAM, tau_prev, tau);
          phi -= TWOPI*floor(phi/TWOPI);

          /*  Compute Jacobian elliptic function sn(tau,kc2) (along with cn and dn)  */

          sncndn(tau, kc2, &sn, &cn, &dn);

          /*  Compute second and third Euler angles theta and psi (eqs. A37, A36)  */

          theta = acos( dn * sqrt( min_truemoment*(max_truemoment - L2_over_2E)
                                   / (L2_over_2E*(max_truemoment - min_truemoment)) ) );
          psi = atan2( sqrt(int_truemoment/(int_truemoment - min_truemoment)) * sn,
                       sqrt(max_truemoment/(max_truemoment - min_truemoment)) * cn);
          psi -= TWOPI*floor(psi/TWOPI);

          /*  Display the Euler angles to the screen  */

          epoch = startepoch + i*epoch_incr;
          dist = ephem2mat( ephem, ephem, epoch, oe, se, orbspin,
                            &solar_phase, &solar_azimuth, 0);
          epoch_ltc = epoch - DAYSPERAU*dist;
          jd2cal( &year, &mon, &day, &hour, &min, &sec, epoch);
          jd2cal( &year0, &mon0, &day0, &hour0, &min0, &sec0, epoch_ltc);
          printf("      JD %.5lf  =  UT %4d %s %2.2d %2.2d:%2.2d:%2.2d"
                 "   (viewed on Earth at %2.2d:%2.2d:%2.2d)"
                 ":  phi, theta, psi = %11.6f %11.6f %11.6f deg\n",
                 epoch, year, monthName[mon-1], day, hour, min, sec,
                 hour0, min0, sec0, phi*R2D, theta*R2D, psi*R2D);

          /*  Increment dimensionless time tau for the next time interval  */

          tau_prev = tau;
          tau += tau_incr;
        }

    } else {

        /*  short-axis mode (SAM)  */

        /*  Compute constant (unit = days) used to normalize time (eq. A55)  */

        tnorm = sqrt( min_truemoment*int_truemoment*max_truemoment
                           / (2*energy*(max_truemoment - int_truemoment)
                                      *(    L2_over_2E - min_truemoment) ) );

        /*  Compute parameter k^2 (eq. A56)  */

        k2 = (int_truemoment - min_truemoment)*(max_truemoment - L2_over_2E)
             / ((max_truemoment - int_truemoment)*(L2_over_2E - min_truemoment));
        kc2 = 1 - k2;

        /*  Compute the period of oscillatory rolling about the long axis
            (eq. A71); in body-fixed coordinates, this is the period of
            the spin vector's motion (Landau & Lifshitz, vol. 1, p. 118).
            K_cel is the complete elliptic integral of the first kind.     */

        K_cel = ellf(PIOVER2, sqrt(k2));
        period_psi = 4*tnorm*K_cel;

        /*  Compute the amplitude of oscillatory rolling about the
            long axis (eq. A65)                                     */

        ampl_psi = asin( sqrt( int_truemoment*(max_truemoment - L2_over_2E)
                               / (L2_over_2E*(max_truemoment - int_truemoment)) ) );

        /*  Working from the instantaneous rate at which the long axis
            rotates about the angular momentum vector (eq. A62), a
            rate which is periodic with period = period_psi/2, integrate
            over one such period to get the average rotation rate, and
            from this define an average rotation period.

            upper_limit is the upper integration limit
                  = (period_psi/2)
                    * sqrt( 2*energy*(max_truemoment - int_truemoment)
                                    *(    L2_over_2E - min_truemoment)
                            / (min_truemoment*int_truemoment*max_truemoment) )
                  = (period_psi/2) * (1/tnorm)
                  = 2*K_cel
            -- see eq. A55 and the text following eq. A74.                      */

        upper_limit = 2*K_cel;
        phi_dot_avg = (1/upper_limit) * qsimp(phi_dot_SAM, 0.0, upper_limit);
        period_phi_avg = TWOPI/phi_dot_avg;
        phi_dot_min = angmom_mag/max_truemoment;
        phi_dot_max = angmom_mag/L2_over_2E;

        /*  For short-axis modes the long axis nutates (nods) with period
            = period_psi (Samarasinha & A'Hearn 1991, p. 199); this nutation
            is centered on theta = pi/2 and has a fixed amplitude (eq. A68).  */

        period_theta = period_psi;
        ampl_theta = asin( sqrt( min_truemoment*(max_truemoment - L2_over_2E)
                                 / (L2_over_2E*(max_truemoment - min_truemoment)) ) );

        printf("      Rotation of the long axis about the angular momentum vector:\n\n");
        printf("            <P_phi> = %G days\n\n", period_phi_avg);
        printf("            %G deg/day <= d(phi)/dt <= %G deg/day\n\n",
               phi_dot_min*R2D, phi_dot_max*R2D);
        printf("      Oscillatory roll about the long axis:\n\n");
        printf("            P_psi   = %G days,  A_psi =   %G deg\n\n",
               period_psi, ampl_psi*R2D);
        printf("      Nutation of the long axis, centered on theta = 90 deg:\n\n");
        printf("            P_theta = %G days,  A_theta = %G deg\n\n",
               period_theta, ampl_theta*R2D);

        /*  Samarasinha & A'Hearn (1991) define their coordinates such that
            -pi/2 <= psi <= pi/2 for a short-axis mode (eqs. A60 and A63-A64), where
            psi is the third Euler angle.  Thus when we specify the spin component
            along the short axis we must choose the "positive" side of the short axis
            to be the side that makes an acute angle with the angular momentum vector
            (the inertial +Z-axis): see Fig. A1.  This may require negating the
            corresponding body-fixed spin component.                                   */

        shortsign = (intspin[shortpa_index] < 0.0) ? -1 : 1;

        /*  Choose the appropriate sides of the other two body-fixed axes so that
            (+long, +intermediate, +short) forms a right-handed triple (see Fig. A1)  */

        if ((shortpa_index + 1) % 3 == longpa_index)
          longsign = intsign = shortsign;
        else
          longsign = intsign = -shortsign;

        /*  Compute dimensionless time tau at spin reference epoch t0 by using the
            "get_tau0" routine to search for the value that yields (via eqs. A57-A59)
            three spin components that match the three spin components specified in
            the mod file for epoch t0                                                  */

        cn0 = longsign*intspin[longpa_index]
              / sqrt(2*energy*(max_truemoment - L2_over_2E)
                     / (min_truemoment*(max_truemoment - min_truemoment)));
        sn0 = intsign*intspin[intpa_index]
              / sqrt(2*energy*(max_truemoment - L2_over_2E)
                     / (int_truemoment*(max_truemoment - int_truemoment)));
        dn0 = shortsign*intspin[shortpa_index]
              / sqrt(2*energy*(L2_over_2E - min_truemoment)
                     / (max_truemoment*(max_truemoment - min_truemoment)));
        tau0 = get_tau0( sn0, cn0, dn0, 4*K_cel);

        /*  Compute the unit vector pointing perpendicular to both angular momentum L
            and the long axis, in the direction of positive d(theta)/dt where theta
            is the second Euler angle; see Fig. A1 of Samarasinha & A'Hearn (1991).
            Then use this to get first Euler angle phi at spin reference epoch t0.     */

        for (i=0; i<=2; i++)
          long_hat[i] = 0.0;
        long_hat[longpa_index] = (double) longsign;
        cotrans( long_hat, ae, long_hat, -1);
        cross( theta_dot_vec, angmom_hat, long_hat);
        phi0 = atan2( dot( theta_dot_vec, Y_hat), dot( theta_dot_vec, X_hat));

        /*  Compute Euler angles as a function of time  */

        tau_incr = epoch_incr/tnorm;
        tau_prev = tau0;
        tau = tau0 + (startepoch - (*mod).spin.t0)/tnorm;
        phi = phi0;
        for (i=0; i<=max_iter; i++) {

          /*  Integrate the time derivative of first Euler angle phi to obtain the
              increment in phi during this time interval (eqs. A62 and A21); then
              use this increment to get phi itself, given that we initialized phi
              to have its value (determined earlier) at spin reference epoch t0     */

          phi += tnorm * qsimp(phi_dot_SAM, tau_prev, tau);
          phi -= TWOPI*floor(phi/TWOPI);

          /*  Compute Jacobian elliptic function sn(tau,kc2) (along with cn and dn)  */

          sncndn(tau, kc2, &sn, &cn, &dn);

          /*  Compute second and third Euler angles theta and psi (eqs. A61, A60)  */

          theta = acos( cn * sqrt( min_truemoment*(max_truemoment - L2_over_2E)
                                   / (L2_over_2E*(max_truemoment - min_truemoment)) ) );
          psi = atan2( sqrt(int_truemoment*(max_truemoment - L2_over_2E)
                            / (max_truemoment - int_truemoment)         ) * sn,
                       sqrt(max_truemoment*(L2_over_2E - min_truemoment)
                            / (max_truemoment - min_truemoment)         ) * dn);

          /*  Display the Euler angles to the screen  */

          epoch = startepoch + i*epoch_incr;
          dist = ephem2mat( ephem, ephem, epoch, oe, se, orbspin,
                            &solar_phase, &solar_azimuth, 0);
          epoch_ltc = epoch - DAYSPERAU*dist;
          jd2cal( &year, &mon, &day, &hour, &min, &sec, epoch);
          jd2cal( &year0, &mon0, &day0, &hour0, &min0, &sec0, epoch_ltc);
          printf("      JD %.5lf  =  UT %4d %s %2.2d %2.2d:%2.2d:%2.2d"
                 "   (viewed on Earth at %2.2d:%2.2d:%2.2d)"
                 ":  phi, theta, psi = %11.6f %11.6f %11.6f deg\n",
                 epoch, year, monthName[mon-1], day, hour, min, sec,
                 hour0, min0, sec0, phi*R2D, theta*R2D, psi*R2D);

          /*  Increment dimensionless time tau for the next time interval  */

          tau_prev = tau;
          tau += tau_incr;
        }

    }

    fflush(stdout);

  }


  /*  - - - - - - - - - start of "epoch search" section  - - - - - - - - - - -  */

  /*  Now get the epoch at which the positive side of the long principal
      axis is in the plane of the sky and approaching the observer.

      For PA rotators, use the principal moments computed assuming uniform
      density to determine which principal axis is the "long" one; for NPA
      rotators, use the "spin.inertia" principal moments instead.           */

  printf("\nFind epochs when the positive side of the\n");
  printf("long principal axis (PA%d) is in the plane of the sky"
            " while approaching us:\n", longpa_index+1);

  /*  Start by setting the unit vector (in body coordinates) pointing
      along the long principal axis in the positive direction.         */

  for (i=0; i<=2; i++)
    n_longpa[i] = ap[i][longpa_index];

  /*  Add the angle offsets to the model Euler angles.
      (Function longpa_losproj later will add the spin offsets separately for
      each trial epoch.)  Save the original angles to be restored later.       */

  for (i=0;i<=2;i++) {
    anglesave[i] = (*mod).spin.angle[i].val;
    (*mod).spin.angle[i].val += (*par).pa_angleoff[i];
  }

  /*  Display the epoch range for the search  */

  printf("\n      Searching over JD range %.5lf to %.5lf\n", startepoch, stopepoch);
  fflush(stdout);

  /*
      Compute the line-of-sight component (towards Earth) of the unit vector
      pointing along the long principal axis in the positive direction; the
      value returned by function longpa_losproj is in the range (-2, 2),
      where values > 1 and < -1 indicate that this half of this axis is
      receding rather than approaching.

      Do this for successive trial epochs until we find two values which
      straddle zero: This tells us that the correct epoch is somewhere in
      between.  Then use bisection (Numerical Recipes function rtbis) to find
      the correct epoch.  Then repeat this entire process to locate successive
      epochs until we've covered the entire epoch range specified.

      NOTE: This method could miss an epoch if ANGLEINCR_DEG (and hence angleincr)
            is too large, that is, if it corresponds to more than a quarter-rotation.
            It can also fail if the long axis passes through the plane two or more
            times between consecutive trial epochs, because bisection will locate
            only one of these events.  The latter problem is only likely to occur
            for NPA rotators, and can be addressed (at the cost of speed) by
            making ANGLEINCR_DEG very small.
  */

  epoch = startepoch;
  bval = longpa_losproj(epoch);
  n_zeros = 0;
  n_iter = 0;
  
  do {

    /*  Bracket the desired epoch: longpa_losproj will return successive
        values which have opposite sign and whose absolute values are < 1  */

    do {
        aval = bval;
        epoch += epoch_incr;
        n_iter++;
        bval = longpa_losproj(epoch);
    } while (n_iter < max_iter &&
             (fabs(aval) >= 1.0 || fabs(bval) >= 1.0 || aval*bval > 0.0));

    /*  Now we can use bisection to home in on the correct epoch
        (unless by chance we're already exactly there).

            bestepoch  = light-time-corrected epoch (from longpa_losproj)
            bestepoch0 = epoch at which event is observed on Earth         */

    if (bval == 0.0) {
        n_zeros++;
        bestepoch0 = epoch;
        bestepoch = t;
        jd2cal( &year, &mon, &day, &hour, &min, &sec, bestepoch);
        jd2cal( &year0, &mon0, &day0, &hour0, &min0, &sec0, bestepoch0);
        printf("\n      JD %.5lf  =  UT %4d %s %2.2d %2.2d:%2.2d:%2.2d"
                   "   (viewed on Earth at %2.2d:%2.2d:%2.2d)",
               bestepoch, year, monthName[mon-1], day, hour, min, sec,
               hour0, min0, sec0);
        fflush(stdout);
    } else if (fabs(aval) < 1.0 && fabs(bval) < 1.0 && aval*bval < 0.0) {
        n_zeros++;
        bestepoch0 = rtbis(longpa_losproj, epoch-epoch_incr, epoch, EPOCHTOL);
        bestroot = longpa_losproj(bestepoch0);
        bestepoch = t;

        /*  bestroot is the sine of the angle which the positive end of
            the long principal axis makes with the plane of the sky at
            the epoch returned by NR bisection routine rtbis; this angle
            should be very small if this epoch (bestepoch) is valid
            (i.e., if the odd motion of an NPA rotator hasn't tricked us).

            We have to test the angle explicitly because rtbis doesn't
            check that the function value is close to zero at the epoch
            it returns: It simply ASSUMES that there's a valid root
            within the starting epoch interval, and returns a value when
            bisection has reduced the interval to be smaller than EPOCHTOL.  */

        if (fabs(asin(bestroot)) < angletol) {
          jd2cal( &year, &mon, &day, &hour, &min, &sec, bestepoch);
          jd2cal( &year0, &mon0, &day0, &hour0, &min0, &sec0, bestepoch0);
          printf("\n      JD %.5lf  =  UT %4d %s %2.2d %2.2d:%2.2d:%2.2d"
                     "   (viewed on Earth at %2.2d:%2.2d:%2.2d)",
                 bestepoch, year, monthName[mon-1], day, hour, min, sec,
                 hour0, min0, sec0);
        }
        fflush(stdout);
    }

    /*  Look for the next epoch if we're not at the end of the search range  */

  } while (n_iter < max_iter);

  /*  Notify user if no epochs were found; then restore the original angles  */

  if (n_zeros == 0)
    printf("\n      No such epochs exist within that search range");
  for (i=0; i<=2; i++)
    (*mod).spin.angle[i].val = anglesave[i];

  /*  - - - - - - - - - end of "epoch search" section  - - - - - - - - - - - -  */

  /*  Write six pgm or ppm files for model views along the principal axes  */

  printf("\n\n*********************************************************\n\n");

  write_pa_views( par, mod, ap, longpa_index, shortpa_index);

  printf("\n");
  fflush(stdout);

}


/*  phi_dot_LAM is the instantaneous rate (rad/day) at which the
    long axis precesses about the angular momentum vector
    for a long-axis mode (Samarasinha & A'Hearn 1991, eq. A38)    */

double phi_dot_LAM(double tau)
{
  double sn, cn, dn, numer, denom;

  /*  Compute Jacobian elliptic function sn(tau,kc2) (along with cn and dn)  */

  sncndn(tau, kc2, &sn, &cn, &dn);

  /*  Compute instantaneous precession rate  */

  numer = (int_truemoment - min_truemoment)
          + (max_truemoment - int_truemoment)*sn*sn;
  denom = max_truemoment*(int_truemoment - min_truemoment)
          + min_truemoment*(max_truemoment - int_truemoment)*sn*sn;
  return angmom_mag*(numer/denom);
}


/*  phi_dot_SAM is the instantaneous rate (rad/day) at which the
    long axis rotates about the angular momentum vector
    for a short-axis mode (Samarasinha & A'Hearn 1991, eq. A62)   */

double phi_dot_SAM(double tau)
{
  double sn, cn, dn, numer, denom;

  /*  Compute Jacobian elliptic function sn(tau,kc2) (along with cn and dn)  */

  sncndn(tau, kc2, &sn, &cn, &dn);

  /*  Compute instantaneous rotation rate  */

  numer = (L2_over_2E - min_truemoment) + (max_truemoment - L2_over_2E)*sn*sn;
  denom = max_truemoment*(L2_over_2E - min_truemoment)
          + min_truemoment*(max_truemoment - L2_over_2E)*sn*sn;
  return angmom_mag*(numer/denom);
}


/*  Compute the line-of-sight component (towards Earth) of the unit vector
    pointing along the long principal axis in the positive direction; the
    value returned by function longpa_losproj is in the range (-2, 2),
    where values > 1 and < -1 indicate that this half of this axis is
    receding rather than approaching.                                       */

double longpa_losproj(double t0)
{
  int i, los_sign, k, n, n_integrate;
  double dist, solar_phase, solar_azimuth, orbspin[3], intspin[3], spin[3],
         oe[3][3], se[3][3], ae[3][3], n_plusDop[3], n_longpa_obs[3], n_dot_n,
         t_integrate[MAXIMP+2], impulse[MAXIMP+2][3];

  /*  Compute oe, the ecliptic-to-observer coordinate transformation matrix,
      and orbspin, the plane-of-sky motion's contribution to the apparent
      spin vector.  Correct for one-way light-travel time if desired.         */

  dist = ephem2mat( *sephem, *sephem, t0, oe, se, orbspin,
                    &solar_phase, &solar_azimuth, 0);
  t = t0 - DAYSPERAU*dist;
  ephem2mat( *sephem, *sephem, t, oe, se, orbspin,
             &solar_phase, &solar_azimuth, 0);

  /*  Figure out which spin impulses will be "encountered" in evolving the spin state
      from the initial spin epoch (spin.t0) to t; then create lists of epochs and
      impulses that will be used by the inteuler routine to break up the integration
      (i.e., the evolution of the spin state) into several smaller time intervals,
      punctuated by spin impulses.  (Code adapted from the realize_spin routine)       */

  k = 0;
  t_integrate[k] = (*smod).spin.t0;
  for (i=0; i<=2; i++)
    impulse[k][i] = 0.0;
  if (t >= (*smod).spin.t0) {

      /*  We'll be integrating forward in time, so add the spin impulses  */

      for (n=0; n<(*smod).spin.n_impulse; n++) {
        if ((*smod).spin.t_impulse[n] > (*smod).spin.t0
                         && (*smod).spin.t_impulse[n] <= t) {
          k++;
          t_integrate[k] = (*smod).spin.t_impulse[n];
          for (i=0; i<=2; i++)
            impulse[k][i] = (*smod).spin.impulse[n][i].val;
        }
      }
      if (t_integrate[k] < t) {
        k++;
        t_integrate[k] = t;
        for (i=0; i<=2; i++)
          impulse[k][i] = 0.0;
      }

  } else {

      /*  We'll be integrating backwards in time, so subtract the spin impulses  */

      for (n=(*smod).spin.n_impulse-1; n>=0; n--) {
        if ((*smod).spin.t_impulse[n] < (*smod).spin.t0
                         && (*smod).spin.t_impulse[n] >= t) {
          k++;
          t_integrate[k] = (*smod).spin.t_impulse[n];
          for (i=0; i<=2; i++)
            impulse[k][i] = -(*smod).spin.impulse[n][i].val;
        }
      }
      if (t_integrate[k] > t) {
        k++;
        t_integrate[k] = t;
        for (i=0; i<=2; i++)
          impulse[k][i] = 0.0;
      }
  }
  n_integrate = k + 1;
  for (k=n_integrate; k<MAXIMP+2; k++) {
    t_integrate[k] = -HUGENUMBER;
    for (i=0; i<=2; i++)
      impulse[k][i] = 0.0;
  }

  /*  Integrate Euler's equations to get intspin, the target's
      intrinsic spin vector in body-fixed coordinates, and ae,
      the ecliptic-to-body-fixed coordinate transformation matrix.  */

  inteuler( (*smod).spin, t_integrate, impulse, n_integrate, intspin, ae,
            (*smod).spin.pa, (*spar).int_method, (*spar).int_abstol);

  /*  Add the spin offsets to intspin, then transform the
      complete intrinsic spin vector to ecliptic coordinates.   */

  for (i=0;i<=2;i++)
    intspin[i] += (*spar).pa_omegaoff[i];
  cotrans( intspin, ae, intspin, -1);

  /*  Get the total apparent spin vector in ecliptic coordinates.  */

  for (i=0; i<=2; i++)
    spin[i] = orbspin[i] + intspin[i];

  /*  Now transform it to observer coordinates, and compute the
      unit vector which lies in the plane of the sky and points
      in the direction of increasing Doppler.                    */

  cotrans( spin, oe, spin, 1);
  n_plusDop[0] = -spin[1];
  n_plusDop[1] = spin[0];
  n_plusDop[2] = 0.0;
  normalize(n_plusDop);

  /*  Take the unit vector pointing along the long principal axis in
      the positive direction and transform it from body coordinates
      to ecliptic coordinates and then to observer coordinates.       */

  for (i=0;i<=2;i++)
    n_longpa_obs[i] = n_longpa[i];
  cotrans( n_longpa_obs, ae, n_longpa_obs, -1);
  cotrans( n_longpa_obs, oe, n_longpa_obs, 1);

  /*
      Compute the dot product of our two unit vectors to see if the
      positive end of the long principal axis is approaching us
      (positive dot product) or receding (negative dot product).

      If approaching, return the z-component of n_longpa, that is,
      the component pointing towards the observer.

      If receding, return a value > 1 or < -1, to indicate
      that we're in the "wrong half" of the rotation and hence
      very far from the desired epoch.
  */

  n_dot_n = n_longpa_obs[0]*n_plusDop[0] + n_longpa_obs[1]*n_plusDop[1];
  if (n_dot_n > 0) {
      return n_longpa_obs[2];
  } else {
      los_sign = (n_longpa_obs[2] >= 0) ? 1 : -1;
      return (2*los_sign - n_longpa_obs[2]);
  }
}


/*  For a non-principal-axis model, get dimensionless time tau at spin reference
    epoch t0 by matching the three spin vector components given in the mod file
    for that epoch -- or rather, by matching three ratios (sn0, cn0, and dn0)
    that are computed from those components.  The search is performed over the
    range 0 <= tau <= P_tau where P_tau is the period of the sn and cn Jacobian
    elliptic functions.  See Samarasinha & A'Hearn (1991) eqs. A33-A35 for
    long-axis modes and A57-A59 for short-axis modes.

    Samarasinha & A'Hearn (1991) define tau (eqs. A31 and A55) such that tau = 0
    implicitly corresponds to psi = 0, where psi is the third of the three Euler
    angles orienting the principal axes relative to the inertial XYZ coordinate
    system of Fig. A1.  For our application we instead specify particular values
    of the spin components (and thus of the three Euler angles) at a particular
    epoch t0, so we must search for the value of tau at this epoch if we wish to
    make use of the various formulas that involve tau.                            */

double get_tau0( double sn0, double cn0, double dn0, double P_tau)
{
  int n_tau, i;
  double delta_tau, sn, cn, dn, minabsdiff, tau_min, tau, absdiff, tau0;

  /*  Divide the search range in tau into small increments  */

  n_tau = 10000;
  delta_tau = P_tau/n_tau;

  /*  For each trial value of tau, compute Jacobian elliptic functions sn, cn,
      and dn (via the Numerical Recipes "sncndn" routine) and check how close
      these values are to target values sn0, cn0, and dn0; save the value that
      yields the smallest sum of the absolute values of the three differences   */

  sn = 0.0;
  cn = dn = 1.0;
  minabsdiff = fabs(sn - sn0) + fabs(cn - cn0) + fabs(dn - dn0);
  tau_min = 0.0;
  for (i=1; i<=n_tau; i++) {
    tau = i*delta_tau;
    sncndn(tau, kc2, &sn, &cn, &dn);
    absdiff = fabs(sn - sn0) + fabs(cn - cn0) + fabs(dn - dn0);
    if (absdiff < minabsdiff) {
      minabsdiff = absdiff;
      tau_min = tau;
    }
  }

  /*  Choose a small interval around the best (so far) trial value of tau
      and use bisection to zero in on the value that yields sn = sn0, then
      check to make sure that this value also gives cn = cn0 and dn = dn0   */

  sn0_static = sn0;
  tau0 = rtbis(delta_sn, tau_min-delta_tau, tau_min+delta_tau, EPOCHTOL);
  sncndn(tau0, kc2, &sn, &cn, &dn);
  absdiff = fabs(sn - sn0) + fabs(cn - cn0) + fabs(dn - dn0);
  if (absdiff > 1e-6) {
    fprintf( stderr, "tau0 %f sn cn dn %f %f %f sn0 cn0 dn0 %f %f %f\n",
             tau0, sn, cn, dn, sn0, cn0, dn0);
    bailout("routine get_tau0 in show_moments.c could not solve for tau0\n");
  }

  return tau0;
}


/*  The following function is repeatedly called by the "get_tau0" function while
    it is using bisection to home in on dimensionless time tau0; given a possible
    value of tau0 as argument, it compute Jacobian ellipsoid function sn and
    returns the difference between sn and the target value sn0, made accessible
    to the routine in the form of global variable sn0_static.                      */

double delta_sn( double tau0)
{
  double sn, cn, dn;

  /*  Compute Jacobian elliptic function sn(tau,kc2) (along with cn and dn)  */

  sncndn(tau0, kc2, &sn, &cn, &dn);

  /*  Return the difference between sn and the target value of sn  */

  return (sn - sn0_static);
}


/*  Write six pgm or ppm files for model views along the principal axes  */

void write_pa_views( struct par_t *par, struct mod_t *mod, double ap[3][3],
                     int longpa_index, int shortpa_index)
{
  const char *axistype[3]   = {"long", "int", "short"};
  const int clevels[3] = {255, 255, 0};     /* max RGB levels for unseen surface     */
  const int hlevels[3] = {  0,   0, 255};   /* max RGB levels for highlighted facets */
  double orbit_offset[3] = {0.0, 0.0, 0.0};
 
  char name[MAXLEN];
  int i, j, k, l, c, f, icolor, axis, pa_indices[3], flip_pa2, m, matched_facet;
  double posmax, intensityfactor, maxbrightness;
  double **brightness, **image_pgm=NULL, ***image_ppm=NULL;
  struct pos_t pos;

  /*  Initialize variable to avoid compilation warning  */

  matched_facet = 0;

  /*
      We need to fool the posvis routine in order to produce views from the
      six desired perspectives.  Normally posvis produces the body-to-observer
      coordinate transformation matrix oa by using the oe (ecliptic-to-observer)
      and ae (ecliptic-to-body) matrices associated with the data for that
      observing epoch: oa = oe*transpose(ae).  Here we have no data, so we must
      create oe and ae from scratch.

      We've already computed matrix ap (principal-axis-to-body).  If we set
      oe = identity matrix and ae = ap, posvis will set oa = transpose(ap):
      This body-to-principal-axis transformation will give us the view seen by
      an "observer" stationed somewhere on the principal-component +z axis.
      But that leaves five other desired views.

      So we will indeed set ae = ap, but will set oe to the matrix which then
      rotates the principal axes such that the one we want to view from
      becomes the +z axis.  For example, to view from along the +y principal
      axis with the +z axis pointing upward, we want to transform such that
      y --> z, z --> y, and x --> -x.

      The six perspectives:

          pa_long_pos.pgm  (view from +x): +y rightward, +z upward
          pa_int_pos.pgm   (view from +y): +x leftward,  +z upward
          pa_short_pos.pgm (view from +z): +x rightward, +y upward
          pa_long_neg.pgm  (view from -x): +y leftward,  +z upward
          pa_int_neg.pgm   (view from -y): +x rightward, +z upward
          pa_short_neg.pgm (view from -z): +x leftward,  +y upward
  */

  /*  Set up the POS structure  */

  pos.n = ((*par).pos_pixels-1)/2;
  pos.b = matrix( -pos.n, pos.n, -pos.n, pos.n);
  pos.cosi = matrix( -pos.n, pos.n, -pos.n, pos.n);
  pos.cose = matrix( -pos.n, pos.n, -pos.n, pos.n);
  pos.z = matrix( -pos.n, pos.n, -pos.n, pos.n);
  pos.body = imatrix( -pos.n, pos.n, -pos.n, pos.n);
  pos.comp = imatrix( -pos.n, pos.n, -pos.n, pos.n);
  pos.f = imatrix( -pos.n, pos.n, -pos.n, pos.n);
  pos.km_per_pixel = (*par).pos_width/(2.0*pos.n);
  pos.bistatic = 0;

  /*  Set up the "brightness" matrix which will store the grayscale
      pixel values for each view.  Also set up the pgm or ppm image  */

  brightness = matrix( -pos.n, pos.n, -pos.n, pos.n);
  if ((*par).mark_unseen || (*par).pa_highlight)
    image_ppm = d3tensor( -pos.n, pos.n, -pos.n, pos.n, 0, 2);
  else
    image_pgm = matrix( -pos.n, pos.n, -pos.n, pos.n);

  /*  Assign the ae transformation matrix so that posvis
      takes us from body-fixed to principal-axis coordinates  */

  for (i=0; i<=2; i++)
    for (j=0; j<=2; j++)
      if ((*par).pa_bodycoords)
        pos.ae[i][j] = (j == i) ? 1.0 : 0.0;
      else
        pos.ae[i][j] = ap[i][j];

  /*  Get the pixel value that will be set to bright white
      (if zero, the brightest pixel will map to bright white)  */

  posmax = ((*par).pa_scatlaw == RADARVIEW) ? (*par).radposmax : (*par).optposmax;

  /*  Compute the optical intensity factor for apply_photo  */

  intensityfactor = pow( pos.km_per_pixel/AU, 2.0);

  /*  Figure out which principal axis is long vs. intermediate vs. short  */

  pa_indices[0] = longpa_index;
  pa_indices[2] = shortpa_index;
  for (i=0; i<=2; i++)
    if (i != shortpa_index && i != longpa_index)
      pa_indices[1] = i;

  /*  Display how the (long, intermediate, short) principal axes in the images that
      we're about to produce are related to the (PA1, PA2, PA3) principal axes
      obtained by diagonalizing the inertia tensor; we may need to use -PA2 rather
      than +PA2 in order to get a right-handed triple for (long, intermediate, short)  */

  if ((pa_indices[0] == 0 && pa_indices[2] == 1) || (pa_indices[0] == 1 && pa_indices[2] == 2)
                                                 || (pa_indices[0] == 2 && pa_indices[2] == 0))
    flip_pa2 = 1;
  else
    flip_pa2 = 0;
  if ((*par).pa_bodycoords) {
      printf("Images have axes oriented along body-fixed coordinate axes\n\n");
  } else {
      printf("Images have axes oriented with (+long, +intermediate, +short) = (");
      if (pa_indices[0] == 1 && flip_pa2)
        printf("-PA%d, ", pa_indices[0]+1);
      else
        printf("+PA%d, ", pa_indices[0]+1);
      if (pa_indices[1] == 1 && flip_pa2)
        printf("-PA%d, ", pa_indices[1]+1);
      else
        printf("+PA%d, ", pa_indices[1]+1);
      if (pa_indices[2] == 1 && flip_pa2)
        printf("-PA%d)\n\n", pa_indices[2]+1);
      else
        printf("+PA%d)\n\n", pa_indices[2]+1);
  }

  /*  Get the POS pixel values for all six images, along
      with the pixel value which will map to bright white.

      Loop so that we first do the two views from along the
      long principal axis, then intermediate, then short     */

  for (axis=0; axis<=2; axis++) {

    /*  View from along the positive end of this principal axis  */

    /*  To start, assign transformation matrix oe such that posvis will
        rotate the principal axes to transform the one we want to view
        from into the +z axis (while also getting x and y correct)       */

    if ((*par).pa_bodycoords)
      pa_to_obs( pos.oe, axis, 0, 0, 2);
    else
      pa_to_obs( pos.oe, axis, 0, longpa_index, shortpa_index);

    /*  Clear the POS view and then fill it in from the desired perspective  */

    posclr( &pos);
    (*par).posbnd = 0;
    for (c=0; c<(*mod).shape.ncomp; c++)
      if (posvis( &(*mod).shape.comp[c].real, orbit_offset, &pos,
                  (int) (*par).pos_smooth, 0, 0, c))
        (*par).posbnd = 1;
    if ((*par).posbnd)
      printf("WARNING: +PA%d view extends beyond POS frame\n", pa_indices[axis]);

    /*  Compute the POS pixel values  */
int s = 0;
    if ((*par).pa_scatlaw == OPTICALVIEW) {
        for (k=(-pos.n); k<=pos.n; k++)
          for (l=(-pos.n); l<=pos.n; l++)
            pos.cosi[k][l] = pos.cose[k][l];
        apply_photo( mod, 0, 0.0, intensityfactor, &pos, 0, s, i);
        for (k=(-pos.n); k<=pos.n; k++)
          for (l=(-pos.n); l<=pos.n; l++)
            brightness[k][l] = pos.b[k][l];
    } else if ((*par).pa_scatlaw == RADARVIEW) {
        for (k=(-pos.n); k<=pos.n; k++)
          for (l=(-pos.n); l<=pos.n; l++)
            if (pos.cose[k][l] > 0.0)
              brightness[k][l] =
                     radlaw( &(*mod).photo, 0, pos.cose[k][l], pos.comp[k][l],
                             pos.f[k][l]);
            else
              brightness[k][l] = 0.0;
    } else {
        for (k=(-pos.n); k<=pos.n; k++)
          for (l=(-pos.n); l<=pos.n; l++)
            brightness[k][l] = pos.cose[k][l];
    }

    /*  Get the maximum pixel value  */

    maxbrightness = -HUGENUMBER;
    for (k=(-pos.n); k<=pos.n; k++)
      for (l=(-pos.n); l<=pos.n; l++)
        maxbrightness = MAX( maxbrightness, brightness[k][l]);

    /*  Now write the POS view to disk as a pgm image, unless the
        "mark_unseen" parameter is turned on, in which case write it
        as a ppm image with areas that are always hidden/shadowed
        from view marked in color                                     */

    if ((*par).mark_unseen || (*par).pa_highlight) {
        sprintf( name, "pa_%s_pos.ppm", axistype[axis]);
        for (k=(-pos.n); k<=pos.n; k++)
          for (l=(-pos.n); l<=pos.n; l++) {
            if (pos.f[k][l] < 0) {
                for (icolor=0; icolor<=2; icolor++)
                  image_ppm[k][l][icolor] = 0.0;      /* blank sky */
            } else {
                c = pos.comp[k][l];
                f = pos.f[k][l];
                if ((*par).pa_highlight) {
                  matched_facet = 0;
                  m = 0;
                  while (!matched_facet && m < MAXCHOSENFACETS && (*par).pa_facet[m] >= 0)
                    if ((*par).pa_comp[m] == c && (*par).pa_facet[m] == f)
                      matched_facet = 1;
                    else
                      m++;
                }
                for (icolor=0; icolor<=2; icolor++)
                  if ((*par).pa_highlight && matched_facet)
                    image_ppm[k][l][icolor] = brightness[k][l]
                                              * hlevels[icolor] / 255.0;
                  else if ((*par).mark_unseen && !(*mod).shape.comp[c].real.f[f].seen)
                    image_ppm[k][l][icolor] = brightness[k][l]
                                              * clevels[icolor] / 255.0;
                  else
                    image_ppm[k][l][icolor] = brightness[k][l];
            }
          }
        if (posmax == 0.0)
          wimasppm0( image_ppm, -pos.n, pos.n, -pos.n, pos.n, 0, 0, 0, name);
        else
          wimasppmsc( image_ppm, -pos.n, pos.n, -pos.n, pos.n, 0.0, posmax,
                      0, 0, 0, name);
    } else {
        sprintf( name, "pa_%s_pos.pgm", axistype[axis]);
        for (k=(-pos.n); k<=pos.n; k++)
          for (l=(-pos.n); l<=pos.n; l++)
            image_pgm[k][l] = brightness[k][l];
        if (posmax == 0.0)
          wimaspgm0( image_pgm, -pos.n, pos.n, -pos.n, pos.n, 0, 0, 0, name);
        else
          wimaspgmsc( image_pgm, -pos.n, pos.n, -pos.n, pos.n, 0.0, posmax,
                      0, 0, 0, name);
    }

    /*  Print the name of the image file to the screen, and if the "optposmax" or
        "radposmax" parameter is nonzero (for an optical or radar scattering law,
        respectively), also print the value of that parameter vs. the image's
        maximum pixel value                                                        */

    if (posmax != 0.0)
      printf("# %s  (max pixel value = %10.4e, bright white >= %10.4e)\n",
             name, maxbrightness, posmax);
    else
      printf("# %s\n", name);
    fflush(stdout);

    /*  Do it all again, this time viewing from along
        the negative end of the same principal axis    */

    if ((*par).pa_bodycoords)
      pa_to_obs( pos.oe, axis, 1, 0, 2);
    else
      pa_to_obs( pos.oe, axis, 1, longpa_index, shortpa_index);
    posclr( &pos);
    (*par).posbnd = 0;
    for (c=0; c<(*mod).shape.ncomp; c++)
      if (posvis( &(*mod).shape.comp[c].real, orbit_offset, &pos,
                  (int) (*par).pos_smooth, 0, 0, c))
        (*par).posbnd = 1;
    if ((*par).posbnd)
      printf("WARNING: -PA%d view extends beyond POS frame\n", pa_indices[axis]);
    if ((*par).pa_scatlaw == OPTICALVIEW) {
        for (k=(-pos.n); k<=pos.n; k++)
          for (l=(-pos.n); l<=pos.n; l++)
            pos.cosi[k][l] = pos.cose[k][l];
        apply_photo( mod, 0, 0.0, intensityfactor, &pos, 0, s, i);
        for (k=(-pos.n); k<=pos.n; k++)
          for (l=(-pos.n); l<=pos.n; l++)
            brightness[k][l] = pos.b[k][l];
    } else if ((*par).pa_scatlaw == RADARVIEW) {
        for (k=(-pos.n); k<=pos.n; k++)
          for (l=(-pos.n); l<=pos.n; l++)
            if (pos.cose[k][l] > 0.0)
              brightness[k][l] =
                     radlaw( &(*mod).photo, 0, pos.cose[k][l], pos.comp[k][l],
                             pos.f[k][l]);
            else
              brightness[k][l] = 0.0;
    } else {
        for (k=(-pos.n); k<=pos.n; k++)
          for (l=(-pos.n); l<=pos.n; l++)
            brightness[k][l] = pos.cose[k][l];
    }
    maxbrightness = -HUGENUMBER;
    for (k=(-pos.n); k<=pos.n; k++)
      for (l=(-pos.n); l<=pos.n; l++)
        maxbrightness = MAX( maxbrightness, brightness[k][l]);
    if ((*par).mark_unseen || (*par).pa_highlight) {
        sprintf( name, "pa_%s_neg.ppm", axistype[axis]);
        for (k=(-pos.n); k<=pos.n; k++)
          for (l=(-pos.n); l<=pos.n; l++) {
            if (pos.f[k][l] < 0) {
                for (icolor=0; icolor<=2; icolor++)
                  image_ppm[k][l][icolor] = 0.0;      /* blank sky */
            } else {
                c = pos.comp[k][l];
                f = pos.f[k][l];
                if ((*par).pa_highlight) {
                  matched_facet = 0;
                  m = 0;
                  while (!matched_facet && m < MAXCHOSENFACETS && (*par).pa_facet[m] >= 0)
                    if ((*par).pa_comp[m] == c && (*par).pa_facet[m] == f)
                      matched_facet = 1;
                    else
                      m++;
                }
                for (icolor=0; icolor<=2; icolor++)
                  if ((*par).pa_highlight && matched_facet)
                    image_ppm[k][l][icolor] = brightness[k][l]
                                              * hlevels[icolor] / 255.0;
                  else if ((*par).mark_unseen && !(*mod).shape.comp[c].real.f[f].seen)
                    image_ppm[k][l][icolor] = brightness[k][l]
                                              * clevels[icolor] / 255.0;
                  else
                    image_ppm[k][l][icolor] = brightness[k][l];
            }
          }
        if (posmax == 0.0)
          wimasppm0( image_ppm, -pos.n, pos.n, -pos.n, pos.n, 0, 0, 0, name);
        else
          wimasppmsc( image_ppm, -pos.n, pos.n, -pos.n, pos.n, 0.0, posmax,
                      0, 0, 0, name);
    } else {
        sprintf( name, "pa_%s_neg.pgm", axistype[axis]);
        for (k=(-pos.n); k<=pos.n; k++)
          for (l=(-pos.n); l<=pos.n; l++)
            image_pgm[k][l] = brightness[k][l];
        if (posmax == 0.0)
          wimaspgm0( image_pgm, -pos.n, pos.n, -pos.n, pos.n, 0, 0, 0, name);
        else
          wimaspgmsc( image_pgm, -pos.n, pos.n, -pos.n, pos.n, 0.0, posmax,
                      0, 0, 0, name);
    }
    if (posmax != 0.0)
      printf("# %s  (max pixel value = %10.4e, bright white >= %10.4e)\n",
             name, maxbrightness, posmax);
    else
      printf("# %s\n", name);
    fflush(stdout);
  }

  /*  Clean up storage space for the POS views and ppm images  */

  free_matrix( pos.b, -pos.n, pos.n, -pos.n, pos.n);
  free_matrix( pos.cosi, -pos.n, pos.n, -pos.n, pos.n);
  free_matrix( pos.cose, -pos.n, pos.n, -pos.n, pos.n);
  free_matrix( pos.z, -pos.n, pos.n, -pos.n, pos.n);
  free_imatrix( pos.body, -pos.n, pos.n, -pos.n, pos.n);
  free_imatrix( pos.comp, -pos.n, pos.n, -pos.n, pos.n);
  free_imatrix( pos.f, -pos.n, pos.n, -pos.n, pos.n);
  free_matrix( brightness, -pos.n, pos.n, -pos.n, pos.n);
  if ((*par).mark_unseen || (*par).pa_highlight)
    free_d3tensor( image_ppm, -pos.n, pos.n, -pos.n, pos.n, 0, 2);
  else
    free_matrix( image_pgm, -pos.n, pos.n, -pos.n, pos.n);
}


/*  Assign the transformation matrix that will take us from principal axis
    to observer coordinates, as appropriate for the desired model view
    (see comments at the top of the "write_pa_views" routine above)         */

void pa_to_obs( double op[3][3], int axis, int negflag, int longpa_index, int shortpa_index)
{
  double long_to_int_view[3][3] = {{ 0.0,  0.0, -1.0},
                                   { 0.0,  1.0,  0.0},
                                   { 1.0,  0.0,  0.0}};
  double long_to_short_view[3][3] = {{ 0.0,  0.0,  1.0},
                                     { 1.0,  0.0,  0.0},
                                     { 0.0,  1.0,  0.0}};
  int i, j;

  /*  First assign values for the "long_pos" view (from the positive end of the
      long principal axis).  This view has the positive end of the intermediate axis
      pointing rightward and the positive end of the short axis pointing upward.
      The assignments below give the second principal axis (PA2) a "backwards"
      orientation (relative to the plane-of-sky orientations listed in the comments
      to the "write_pa_views" routine) if needed to maintain a right-handed triple,
      with the result that these three matrices contain some negative matrix elements.  */

  for (i=0; i<=2; i++)
    for (j=0; j<=2; j++)
      op[i][j] = 0.0;

  if (longpa_index == 0) {
      if (shortpa_index == 1) {
          op[1][1] = -1.0;
          op[0][2] = op[2][0] = 1.0;
      } else {
          op[0][1] = op[1][2] = op[2][0] = 1.0;
      }
  } else if (longpa_index == 1) {
      if (shortpa_index == 0) {
          op[0][2] = op[1][0] = op[2][1] = 1.0;
      } else {
          op[2][1] = -1.0;
          op[0][0] = op[1][2] = 1.0;
      }
  } else {
      if (shortpa_index == 0) {
          op[0][1] = -1.0;
          op[1][0] = op[2][2] = 1.0;
      } else {
          op[0][0] = op[1][1] = op[2][2] = 1.0;
      }
  }

  /*  Now transform to the "int_pos" or "short_pos" view if desired, viewing
      the model from the positive end of the intermediate or short principal axis  */

  if (axis == 1)
    mmmul( op, long_to_int_view, op);
  else if (axis == 2)
    mmmul( op, long_to_short_view, op);

  /*  Transform to the view from the negative end of this principal axis if desired;
      the +y axis (upward in the plane of the plane-of-sky image) is the same as for
      the corresponding positive view, so just negate the first and last rows of op   */

  if (negflag)
    for (i=0; i<=2; i+=2)  /* skip i=1 */
      for (j=0; j<=2; j++)
        op[i][j] = -op[i][j];
}

#undef ANGLEINCR_DEG
#undef ANGLETOL_DEG
#undef EPOCHTOL
