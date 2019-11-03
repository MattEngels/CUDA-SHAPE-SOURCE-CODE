/*****************************************************************************************
                                                                                  covar.c

Calculates and writes to disk (as "covar.dat") the covariance matrix of all the free
parameters.

Modified 2013 July 18 by CM:
    Delete the "check_par" routine: use the par->fpntr and par->fparstep vectors
        created by the "mkparlist" routine rather than creating the pntr and step
        vectors here.  (mkparlist is now called before the covar routine is called.)
    Guard against taking square roots of negative numbers when calculating standard errors

Modified 2013 May 20 by CM:
    Implement ovoid shape components

Modified 2012 March 24 by CM:
    Add the "dopscale" parameter for delay-Doppler and Doppler datasets
    Add "realize_dopscale" call to func, in order to implement '=' state for Doppler
        scaling factors

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
    Add parameters for the "tabular" radar scattering law to check_par

Modified 2008 August 10 by CM:
    Change "delcor_step" parameter to be a vector rather than a scalar

Modified 2006 October 1 by CM:
    Replace ellipsoid diameters D with two_a, a_over_b, b_over_c
    Add "scalefactor" to harmonic and vertex shape structures
    Add two new arguments to realize_delcor
    Add three new arguments to realize_photo

Modified 2006 March 6 by PT:
    Add "spin.omegadot" parameter for changing spin rate

Modified 2005 September 7 by CM:
    Add parameters for the "harmlommel" "harmhapke" and "harmkaas" optical
        scattering laws to check_par
    Add parameters for the "harmcosine" radar scattering law to check_par

Modified 2005 August 1 by CM:
    Add parameters for the "inhokaas" optical scattering law to check_par

Modified 2005 July 20 by CM:
    Add parameters for the "gaussian" and "hagfors" and "cosine_qs" and 
        "gauss+cosine" and "hagfors+cosine" and "cosine+cosine" and
        inhomogeneous "inhocosine" radar scattering laws to check_par
    Eliminate parameters for the "flat" radar scattering law from check_par

Modified 2005 July 4 by CM:
    Add parameters for the inhomogeneous "inholommel" and "inhohapke"
        optical scattering laws to check_par

Modified 2005 March 15 by CM:
    Replace routines check_photo and check_spin with check_par, a modified
        version of mkparlist; this means that the covar action can now
        handle ANY kind of free parameter
    Take advantage of the Hessian matrix's symmetry to halve the number of
        off-diagonal elements which must be explicitly computed

Modified 2005 February 24 by CM:
    Add "realize_xyoff" call to func, in order to implement '=' state
        for horizontal and vertical offsets in plane-of-sky datasets

Modified 2005 January 25 by CM:
    Eliminated an unused variable

Modified 2004 July 3 by CM:
    Write output to "covar.dat" (as advertised in the header comment
         above) rather than to the screen
    Give some screen updates on the progress of the calculation
    Change "%e" write format to "%13.6e" to align columns
    Write the correlation matrix in addition to the covariance matrix
    Add code to handle the 'lommel', 'lambertian', and 'geometrical'
        optical scattering laws

Modified 2004 April 29 by CM:
    For Kaasalainen scattering law, switch from weighting factor "c"
        (ranging from 0 to infinity) to "wt" (ranging from 0 to 1)

Modified 2004 April 3 by CM:
    Add "list_breakdown" argument to chi2 routine

Modified 2004 February 25 by CM:
    Added Kaasalainen "Lambert + Lommel-Seeliger" scattering law
    Use inertia_step for moments of inertia rather than spin_step
    Use photo_step for photometric parameters rather than length_step
    Recognize that angle_step and spin_step have already been converted
        from deg and deg/day to rad and rad/day by routine read_par
    realize_photo now takes two arguments instead of one

Modified 2003 April 24 by CM:
    Added "realize_delcor" call to func, in order to implement '=' state
        for delay correction polynomial coefficients
*****************************************************************************************/

#include "head.h"

double func( struct par_t *par, struct mod_t *mod, struct dat_t *dat,
             double *x, double sx, double *y, double sy);


void covar( struct par_t *par, struct mod_t *mod, struct dat_t *dat)
{
  FILE *fp;
  double **m, f[4], dummy=0.0, fmin;
  int n, i, j;

  /* Allocate memory for m, the covariance matrix */
  n = par->nfpar;
  m = matrix( 1, n, 1, n);

  /* Announce that the computations are beginning  */
  printf("#\n");
  printf("# starting calculation for %d free parameters\n", n);
  printf("#\n");
  printf("#      -- calculating chi squared for unmodified parameters\n");
  fflush(stdout);

  /* Compute the objective function for the unmodified model  */
  fmin = func( par, mod, dat, &dummy, 0.0, &dummy, 0.0);

  /* Compute the Hessian matrix:
   *
   *    Hessian[i][j] = second partial derivative of chi squared
   *                    with respect to (parameter i)(parameter j)
   *
   * Note that this is a symmetric matrix.                           */

  for (i=0; i<n; i++) {
    for (j=i; j<n; j++) {
      printf("#      -- calculating Hessian element [%2d][%2d]\n", i+1, j+1);
      fflush(stdout);
      if (i == j) {

          /* Calculate a diagonal Hessian element  */
          f[0] = func( par, mod, dat, par->fpntr[i],               0.0, &dummy, 0.0) / fmin;
          f[1] = func( par, mod, dat, par->fpntr[i],  par->fparstep[i], &dummy, 0.0) / fmin;
          f[2] = func( par, mod, dat, par->fpntr[i], -par->fparstep[i], &dummy, 0.0) / fmin;
          m[i+1][i+1] = (f[1] + f[2] - 2*f[0])
                        / (par->fparstep[i] * par->fparstep[i]);        /* 2nd deriv. */

      } else {

          /* Calculate an off-diagonal Hessian element  */
          f[0] = func( par, mod, dat, par->fpntr[i],  par->fparstep[i],
                                      par->fpntr[j],  par->fparstep[j] ) / fmin;
          f[1] = func( par, mod, dat, par->fpntr[i], -par->fparstep[i],
                                      par->fpntr[j], -par->fparstep[j] ) / fmin;
          f[2] = func( par, mod, dat, par->fpntr[i],  par->fparstep[i],
                                      par->fpntr[j], -par->fparstep[j] ) / fmin;
          f[3] = func( par, mod, dat, par->fpntr[i], -par->fparstep[i],
                                      par->fpntr[j],  par->fparstep[j] ) / fmin;
          m[i+1][j+1] = m[j+1][i+1] = (f[0] + f[1] - f[2] - f[3])
                                      / (4 * par->fparstep[i] * par->fparstep[j]);
      }
    }
  }
  printf("#\n");
  printf("# finished calculation\n");
  printf("#\n");

  /* Open output file  */
  printf("# writing output to file: covar.dat ...\n");
  fflush(stdout);
  FOPEN( fp, "covar.dat", "w");

  /* Write the Hessian matrix to disk  */
  fprintf( fp, "\n");
  fprintf( fp, "HESSIAN MATRIX:\n");
  for (i=1; i<=n; i++) {
    for (j=1; j<n; j++)
      fprintf( fp, "%13.6e ", m[i][j]);
    fprintf( fp, "%13.6e\n", m[i][n]);
  }
  fprintf( fp, "\n");

  /* Invert the Hessian matrix to obtain the covariance matrix for parameter
   * estimates, then write this to disk       */
  matinv( m, n);

  fprintf( fp, "COVARIANCE MATRIX:\n");
  for (i=1; i<=n; i++) {
    for (j=1; j<n; j++)
      fprintf( fp, "%13.6e ", m[i][j]);
    fprintf( fp, "%13.6e\n", m[i][n]);
  }
  fprintf( fp, "\n");

  /* Write correlation matrix to disk  */
  fprintf( fp, "CORRELATION MATRIX:\n");
  for (i=1; i<=n; i++) {
    for (j=1; j<n; j++) {
      if (m[i][i] > 0.0 && m[j][j] > 0.0)
        fprintf( fp, "%13.6e ", m[i][j]/sqrt(m[i][i]*m[j][j]));
      else
        fprintf( fp, "     ---      ");
    }
    if (m[i][i] > 0.0 && m[n][n] > 0.0)
      fprintf( fp, "%13.6e\n", m[i][n]/sqrt(m[i][i]*m[n][n]));
    else
      fprintf( fp, "     ---     \n");
  }
  fprintf( fp, "\n");

  /* Write the standard errors to disk  */
  fprintf( fp, "STANDARD ERRORS:\n");
  for (i=1; i<=n; i++)
    if (m[i][i] >= 0.0)
      fprintf( fp, "%13.6e\n", sqrt(m[i][i]));
    else
      fprintf( fp, "     ---     \n");
  fprintf( fp, "\n");

  /* Close the output file and clean up storage space  */
  fclose( fp);
  printf("# writing completed\n");
  fflush(stdout);
  free_matrix( m, 1, n, 1, n);
}


/*  Chi squared function  */

double func( struct par_t *par, struct mod_t *mod, struct dat_t *dat,
             double *x, double sx, double *y, double sy)
{
  double savex, savey;

  savex = *x;
  savey = *y;
  *x += sx;
  *y += sy;
  realize_mod( par, mod);
  realize_spin( par, mod, dat); 
  realize_photo( par, mod, 1.0, 1.0, 0);
  realize_delcor( dat, 0.0, 0);
  realize_dopscale( par, dat, 1.0, 0);
  realize_xyoff( dat);
  calc_fits( par, mod, dat);
  *x = savex;
  *y = savey;
  return chi2( par, dat, 0);
}
