/*****************************************************************
                                                     proj_area.c

Compute the projected area of a model, averaged over half a
rotation, for each of a series of subobserver latitudes between
0 and 90 degrees; also compute the projected area averaged over
all possible subobserver positions.

The subobserver latitude increment is set as close as possible
to the "area_latincr" parameter while still dividing evenly into
90 degrees.  For each subobserver latitude, the projected area
is evaluated at a series of subobserver longitudes whose spacing
is set as close as possible to the "area_longincr" parameter
while still dividing evenly into 180 degrees.

Modified 2007 August 7 by CM:
    Fix bug introduced 2007 August 4: forgot to allocate memory
        for pos.body and pos.comp matrices

Modified 2007 August 4 by CM:
    Add orbit_offset and body arguments to posvis routine and remove
        facet argument

Modified 2006 June 21 by CM:
    For POS renderings, changed res to km_per_pixel

Modified 2005 June 27 by CM:
    Renamed "round" function to "iround" to avoid conflicts

Written 2004 December 20 by CM
******************************************************************/

#include "head.h"

void proj_area( struct par_t *par, struct mod_t *mod)
{
  double orbit_offset[3] = {0.0, 0.0, 0.0};
  int i, j, k, l, n_theta, n_phi, n_nonblankpixels;
  double phi, theta, psi, theta_incr, phi_incr, mean_projarea,
         mean_projarea_theta, pi_over_2;
  struct pos_t pos;

  /* We need to fool the posvis routine in order to produce a view from the
   * desired perspective. Normally posvis produces the body-to-observer coordi-
   * nate transformation matrix oa by using oe (ecliptic-to-observer) and ae
   * (ecliptic-to-body) matrices associated with the data for that observing
   * epoch: oa = oe*transpose(ae). Here we have no data, so we must create oe
   * and ae from scratch.
   * If we set ae = identity matrix and oe = the matrix corresponding to our
   * specified direction, we'll get the view seen by an "observer" stationed
   * somewhere along that direction.  */

  /* Set up the POS structure  */
  pos.n = (par->pos_pixels-1)/2;
  pos.b = matrix( -pos.n, pos.n, -pos.n, pos.n);
  pos.cosi = matrix( -pos.n, pos.n, -pos.n, pos.n);
  pos.cose = matrix( -pos.n, pos.n, -pos.n, pos.n);
  pos.z = matrix( -pos.n, pos.n, -pos.n, pos.n);
  pos.body = imatrix( -pos.n, pos.n, -pos.n, pos.n);
  pos.comp = imatrix( -pos.n, pos.n, -pos.n, pos.n);
  pos.f = imatrix( -pos.n, pos.n, -pos.n, pos.n);
  pos.km_per_pixel = par->pos_width/(2.0*pos.n);
  pos.bistatic = 0;

  /* Set the ae transformation matrix to the identity matrix  */
  for (i=0; i<=2; i++)
    for (j=0; j<=2; j++)
      pos.ae[i][j] = (j == i) ? 1.0 : 0.0;

  /* Loop over viewing co-latitudes between 90 and 0 deg, inclusive (in de-
   * creasing order so that latitudes increase)                */
  psi = 0.0;                                /* Euler angle 2 */
  pi_over_2 = PIE/2;
  n_theta = iround(pi_over_2/par->area_latincr) + 1;
  n_theta = MAX( n_theta, 2);
  theta_incr = pi_over_2/(n_theta - 1);
  mean_projarea = 0.0;
  printf("#\n");
  printf("# subobserver    rotation-averaged\n");
  printf("#  latitude       projected area\n");
  printf("#    (deg)            (km^2)\n");
  printf("# -----------    -----------------\n");
  fflush(stdout);
  for (i=n_theta-1; i>=0; i--) {
    theta = i*theta_incr;                   /* Euler angle 1 */

    /* Loop over viewing longitudes for this co-latitude  */
    if (i == 0) {
        n_phi = 1;  /* pole-on view */
    } else {
        n_phi = iround(PIE/(*par).area_longincr);
        n_phi = MAX( n_phi, 1);
    }
    phi_incr = PIE/n_phi;
    mean_projarea_theta = 0.0;
    for (j=0; j<n_phi; j++) {

      /* Assign oe transformation matrix so it corresponds to the specified
       * latitude and longitude  */
      phi = j*phi_incr;                     /* Euler angle 0 */
      euler2mat( pos.oe, phi, theta, psi);

      /* Clear POS view and fill it in from the desired perspective  */
      posclr( &pos);
      if (posvis( &mod->shape.comp[0].real, orbit_offset, &pos,
                  (int) par->pos_smooth, 0, 0, 0))
        printf("WARNING: View extends beyond POS frame for lat=%4.1f long=%6.1f\n",
               R2D*(pi_over_2 - theta), R2D*(phi - pi_over_2));

      /* Compute the projected area for this viewing position  */
      n_nonblankpixels = 0;
      for (k=pos.xlim[0]; k<=pos.xlim[1]; k++)
        for (l=pos.ylim[0]; l<=pos.ylim[1]; l++)
          if (pos.cose[k][l] > 0.0)
            n_nonblankpixels++;
      mean_projarea_theta += n_nonblankpixels * pos.km_per_pixel * pos.km_per_pixel;
    }

    /* Compute/display mean projected area for this subobserver latitude  */
    mean_projarea_theta /= n_phi;
    printf("#    %4.1f          %12.6e\n",
           R2D*(pi_over_2 - theta), mean_projarea_theta);
    fflush(stdout);

    /* Use extended trapezoidal rule (Numerical Recipes for C, Section 4.1) to
     * compute the mean projected area over all viewing positions             */
    if (i == 0 || i == n_theta-1)
      mean_projarea += 0.5*mean_projarea_theta*sin(theta)*theta_incr;
    else
      mean_projarea += mean_projarea_theta*sin(theta)*theta_incr;
  }

  /* Display the mean projected area over all viewing positions  */
  printf("#\n");
  printf("#     ALL          %12.6e\n", mean_projarea);
  printf("#\n");
  fflush(stdout);

  /* Clean up storage space for the POS view  */
  free_matrix( pos.b, -pos.n, pos.n, -pos.n, pos.n);
  free_matrix( pos.cosi, -pos.n, pos.n, -pos.n, pos.n);
  free_matrix( pos.cose, -pos.n, pos.n, -pos.n, pos.n);
  free_matrix( pos.z, -pos.n, pos.n, -pos.n, pos.n);
  free_imatrix( pos.body, -pos.n, pos.n, -pos.n, pos.n);
  free_imatrix( pos.comp, -pos.n, pos.n, -pos.n, pos.n);
  free_imatrix( pos.f, -pos.n, pos.n, -pos.n, pos.n);
}
