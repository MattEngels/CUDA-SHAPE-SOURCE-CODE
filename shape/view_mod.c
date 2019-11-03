/*****************************************************************************************
                                                                               view_mod.c

Write a pgm or ppm image of a model as viewed from a specified body-fixed latitude and
longitude.

The latitude and longitude are given by parameters view_lat and view_long, respectively.
Imagine viewing the model from a position within the body-fixed equatorial plane at
longitude view_long, with the body-fixed +z axis pointing straight upward; then rotate the
body-fixed +z axis directly toward yourself through angle view_lat.  This is the view
depicted in the image.

The output file is view_DDD_sdd.pgm, where DDD is view_long expressed in degrees,
integer-rounded, and shifted to the range [0,360), and sdd is view_lat expressed in
degrees, integer-rounded, and preceded by the plus or minus sign.

Depending on how one chooses to annotate the image (e.g., with shafts along one or more
principal axes) the output file may be color (ppm) rather than grayscale (pgm).


Modified 2014 February 19 by CM:
    Replace the "is_optical" argument to the plot_surface routine with the "iradlaw"
        argument
    Bug fix: When view_scatlaw = optical, need to call the apply_photo routine to compute
        brightness values

Modified 2013 June 25 by CM:
    Let the plot_surface routine print the image filename to the screen rather than doing
        it here
    Add "posmax" argument to the annotate_plot routine and remove "is_optical" argument

Modified 2012 March 5 by CM:
    Implement "view_highlight" parameter to higlight selected facets in the output image

Modified 2009 July 29 by CM:
    Add "color_output" argument to the "write_pnm" routine

Modified 2009 July 29 by CM:
    Add "color_output" argument to the "write_pnm" routine

Modified 2009 April 10 by CM:
    Add the ability to add plot annotations: spin vector, COM cross,
        principal-axis shafts, angular momentum vector.  (As a
        result the output image file may be ppm rather than pgm.)
        Use the "annotate_plot" routine (in write_pos.c) for this so
        as not to duplicate code.  The new "view_posfactor"
        parameter controls the size of the annotations.
    Remove the code for plotting the asteroid's surface model in the
        image and writing the image to disk: rely instead on the
        "plot_surface" and "write_pnm" routines (in write_pos.c).

Modified 2007 August 10 by CM:
    Initialize uninitialized variables

Modified 2007 August 4 by CM:
    Add orbit_offset and body arguments to posvis routine and
        remove facet argument
    Add c (component) argument to radlaw routine
    Add body argument to apply_photo routine
    Add body, bodyill, comp, and compill matrices to POS frames

Modified 2006 October 1 by CM:
    Add "intensityfactor" argument to apply_photo

Modified 2006 September 1 by CM and MCN:
    When "mark_unseen" parameter is turned on, add check that
        facet number pos.f[i][j] is nonnegative

Modified 2006 June 21 by CM:
    For POS renderings, change res to km_per_pixel

Modified 2005 September 18 by CM:
    Implement the "view_shadows" parameter: if this parameter is
        turned on, shape places the Sun at the body-fixed
        coordinates given by the "view_sunlat" and "view_sunlong"
        parameters and computes shadowing accordingly

Modified 2005 August 1 by CM:
    Implement the "view_scatlaw" parameter, which determines
        whether the image will use the optical scattering law,
        the radar scattering law, or (default) the Lambert law

Modified 2005 July 21 by CM:
    Fix routine to work for models with more than one component

Modified 2005 June 23 by CM:
    If the "mark_unseen" parameter is turned on, use color to mark
        regions which are always hidden/shadowed from Earth's
        view.  In this case the output images are in ppm format
        rather than the usual pgm format.

Modified 2004 December 20 by CM:
    Write the projected area to the screen

Written 2004 November 23 by CM
*****************************************************************************************/

#include "head.h"

void view_mod( struct par_t *par, struct mod_t *mod)
{
  double orbit_offset[3] = {0.0, 0.0, 0.0};
                                                                                                
  char name[MAXLEN];
  int i, j, long_deg, lat_deg, k, l, n_nonblankpixels, c,
      sunlong_deg, sunlat_deg, color_output, iradlaw;
  int **color;
  double phi, theta, psi, long_deg_dbl, lat_deg_dbl, projected_area,
         maxbrightness, posmax, sunlong_deg_dbl, sunlat_deg_dbl,
         solar_phase, intensityfactor, spin_body[3], spin_ecl[3];
  double **brightness, **z;
  struct pos_t pos;

  /*  Initialize variables to avoid compilation warnings  */

  sunlong_deg = sunlat_deg = 0;

  /*
      We need to fool the posvis routine in order to produce a view from the
      desired perspective.  Normally posvis produces the body-to-observer
      coordinate transformation matrix oa by using the oe (ecliptic-to-observer)
      and ae (ecliptic-to-body) matrices associated with the data for that
      observing epoch: oa = oe*transpose(ae).  Here we have no data, so we must
      create oe and ae from scratch.

      If we set ae = identity matrix and oe = the matrix corresponding to
      our specified direction, we'll get the view seen by an "observer"
      stationed somewhere along that direction.
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
  pos.cosill = matrix( -pos.n, pos.n, -pos.n, pos.n);
  pos.zill = matrix( -pos.n, pos.n, -pos.n, pos.n);
  pos.bodyill = imatrix( -pos.n, pos.n, -pos.n, pos.n);
  pos.compill = imatrix( -pos.n, pos.n, -pos.n, pos.n);
  pos.fill = imatrix( -pos.n, pos.n, -pos.n, pos.n);
  if ((*par).view_scatlaw == OPTICALVIEW || (*par).view_shadows)
    pos.bistatic = 1;
  else
    pos.bistatic = 0;

  /*
      Prepare storage matrices to list three quantities for each
         line of sight (each POS pixel):

         1) color type (0 = asteroid surface, 1 = unseen ast. surf., 2 = spin vector, etc.)
         2) brightness (grayscale)
         3) z-coordinate (distance toward observer) of the closest
               plotting element (asteroid surface, arrow, etc.)
  */

  color = imatrix( -pos.n, pos.n, -pos.n, pos.n);
  brightness = matrix( -pos.n, pos.n, -pos.n, pos.n);
  z = matrix( -pos.n, pos.n, -pos.n, pos.n);

  /*  Set the ae transformation matrix to the identity matrix  */

  for (i=0; i<=2; i++)
    for (j=0; j<=2; j++)
      pos.ae[i][j] = (j == i) ? 1.0 : 0.0;

  /*  Assign the oe transformation matrix so that it
      corresponds to the specified latitude and longitude  */

  phi = (*par).view_long + PIE/2;       /* Euler angle 0 */
  theta = PIE/2 - (*par).view_lat;      /* Euler angle 1 */
  psi = 0.0;                            /* Euler angle 2 */
  euler2mat( pos.oe, phi, theta, psi);

  /*  Assign the se transformation matrix so that it
      corresponds to the specified solar latitude and longitude  */

  if ((*par).view_shadows) {
      phi = (*par).view_sunlong + PIE/2;    /* Euler angle 0 */
      theta = PIE/2 - (*par).view_sunlat;   /* Euler angle 1 */
      psi = 0.0;                            /* Euler angle 2 */
      euler2mat( pos.se, phi, theta, psi);
  } else if ((*par).view_scatlaw == OPTICALVIEW) {
      for (i=0; i<=2; i++)
        for (j=0; j<=2; j++)
          pos.se[i][j] = pos.oe[i][j];
  }

  /*  Compute the solar phase angle  */

  if ((*par).view_shadows)
    solar_phase = acos( cos((*par).view_lat) * cos((*par).view_sunlat)
                            * cos((*par).view_long - (*par).view_sunlong)
                        + sin((*par).view_lat) * sin((*par).view_sunlat)  );
  else
    solar_phase = 0.0;

  /*  Compute the optical intensity factor for apply_photo  */

  intensityfactor = pow( pos.km_per_pixel/AU, 2.0);

  /*  Get the integer-rounded longitude and latitude for
      use in the output file name; longitude range is
      [0,360) deg and latitude range is [-90,+90] deg.     */

  long_deg_dbl = R2D*(*par).view_long;
  lat_deg_dbl = R2D*(*par).view_lat;
  if (fabs(lat_deg_dbl) > 90.000001) {
    lat_deg_dbl -= 360.0*floor((lat_deg_dbl + 90.0)/360.0);
    if (lat_deg_dbl > 90.000001) {
      lat_deg_dbl = 180.0 - lat_deg_dbl;
      long_deg_dbl += 180.0;
    }
  }
  long_deg_dbl = (long_deg_dbl + 0.5) - 360.0*floor((long_deg_dbl + 0.5)/360.0);
  long_deg = (int) floor(long_deg_dbl);
  lat_deg = (int) floor(lat_deg_dbl + 0.5);

  /*  Now do the same for the Sun's longitude and latitude  */

  if ((*par).view_shadows) {
    sunlong_deg_dbl = R2D*(*par).view_sunlong;
    sunlat_deg_dbl = R2D*(*par).view_sunlat;
    if (fabs(sunlat_deg_dbl) > 90.000001) {
      sunlat_deg_dbl -= 360.0*floor((sunlat_deg_dbl + 90.0)/360.0);
      if (sunlat_deg_dbl > 90.000001) {
        sunlat_deg_dbl = 180.0 - sunlat_deg_dbl;
        sunlong_deg_dbl += 180.0;
      }
    }
    sunlong_deg_dbl = (sunlong_deg_dbl + 0.5)
                      - 360.0*floor((sunlong_deg_dbl + 0.5)/360.0);
    sunlong_deg = (int) floor(sunlong_deg_dbl);
    sunlat_deg = (int) floor(sunlat_deg_dbl + 0.5);
  }

  /*  Clear the POS view and then fill it in from the desired perspective  */

  posclr( &pos);
  (*par).posbnd = 0;
  for (c=0; c<(*mod).shape.ncomp; c++)
    if (posvis( &(*mod).shape.comp[c].real, orbit_offset, &pos,
                (int) (*par).pos_smooth, 0, 0, c))
      (*par).posbnd = 1;
  if ((*par).posbnd)
    printf("WARNING: View extends beyond POS frame\n");

  /*  Display the projected area (prior to applying shadowing)  */

  n_nonblankpixels = 0;
  for (k=pos.xlim[0]; k<=pos.xlim[1]; k++)
    for (l=pos.ylim[0]; l<=pos.ylim[1]; l++)
      if (pos.cose[k][l] > 0.0)
        n_nonblankpixels++;
  projected_area = n_nonblankpixels * pos.km_per_pixel * pos.km_per_pixel;
  printf("#\n");
  printf("# projected area for (long, lat) = (%03d, %+03d) deg is %12.6e km^2\n",
         long_deg, lat_deg, projected_area);
  printf("#\n");
  fflush(stdout);

  /*  Now view the model from the source (sun) and get the facet number
      and distance toward the source of each pixel in this projected view;
      use this information to determine which POS pixels are shadowed       */
 
  (*par).posbnd = 0;
  if ((*par).view_shadows) {
    for (c=0; c<(*mod).shape.ncomp; c++)
      if (posvis( &(*mod).shape.comp[c].real, orbit_offset, &pos, 0, 1, 0, c))
        (*par).posbnd = 1;
    if ((*par).posbnd)
      printf("WARNING: View from Sun extends beyond POS frame\n");

    /*  Identify and mask out shadowed POS pixels  */

    posmask( &pos, (*par).mask_tol);
  }

  /*  If we're using the optical scattering law for the image,
      call apply_photo to compute the POS pixel values          */
int s = 0;
  if ((*par).view_scatlaw == OPTICALVIEW)
    apply_photo( mod, 0, solar_phase, intensityfactor, &pos, 0, s, i);

  /*  Figure out the name of the image file  */

  color_output = ((*par).mark_unseen || (*par).view_highlight || (*par).plot_angmom ||
                  (*par).plot_pa[0]  || (*par).plot_pa[1]  || (*par).plot_pa[2]);
  if (color_output) {
      if ((*par).view_shadows)
        sprintf( name, "view_%03d_%+03d_sun_%03d_%+03d.ppm",
                 long_deg, lat_deg, sunlong_deg, sunlat_deg);
      else
        sprintf( name, "view_%03d_%+03d.ppm", long_deg, lat_deg);
  } else {
      if ((*par).view_shadows)
        sprintf( name, "view_%03d_%+03d_sun_%03d_%+03d.pgm",
                 long_deg, lat_deg, sunlong_deg, sunlat_deg);
      else
        sprintf( name, "view_%03d_%+03d.pgm", long_deg, lat_deg);
  }

  /*  Start the plot by filling in the asteroid surface
      (this will also display the image filename to the screen)  */

  iradlaw = ((*par).view_scatlaw == RADARVIEW) ? 0 : -1;
  plot_surface( par, mod, (*par).view_scatlaw, iradlaw, name,
                &maxbrightness, &posmax, &pos, color, brightness, z);
  printf("#\n");
  fflush(stdout);

  /*  Get the (initial) spin vector in ecliptic coordinates  */

  for (i=0; i<=2; i++)
    spin_body[i] = (*mod).spin.omega[i].val;
  cotrans( spin_ecl, pos.ae, spin_body, -1);

  /*  Add each of the plot annotations that has been specified:
      spin vector, center of mass, angular momentum vector       */

  annotate_plot( par, mod, spin_ecl, maxbrightness, posmax,
                 &pos, color, brightness, z);

  /*  Write the POS view to disk as a pgm or ppm image
      depending on how parameters like "mark_unseen" are set  */

  write_pnm( posmax, pos.n, color, brightness, color_output, name);

  /*  Clean up storage space for the POS view and image(s)  */

  free_matrix( pos.b, -pos.n, pos.n, -pos.n, pos.n);
  free_matrix( pos.cosi, -pos.n, pos.n, -pos.n, pos.n);
  free_matrix( pos.cose, -pos.n, pos.n, -pos.n, pos.n);
  free_matrix( pos.z, -pos.n, pos.n, -pos.n, pos.n);
  free_imatrix( pos.body, -pos.n, pos.n, -pos.n, pos.n);
  free_imatrix( pos.comp, -pos.n, pos.n, -pos.n, pos.n);
  free_imatrix( pos.f, -pos.n, pos.n, -pos.n, pos.n);
  free_matrix( pos.cosill, -pos.n, pos.n, -pos.n, pos.n);
  free_matrix( pos.zill, -pos.n, pos.n, -pos.n, pos.n);
  free_imatrix( pos.bodyill, -pos.n, pos.n, -pos.n, pos.n);
  free_imatrix( pos.compill, -pos.n, pos.n, -pos.n, pos.n);
  free_imatrix( pos.fill, -pos.n, pos.n, -pos.n, pos.n);
  free_imatrix( color, -pos.n, pos.n, -pos.n, pos.n);
  free_matrix( brightness, -pos.n, pos.n, -pos.n, pos.n);
  free_matrix( z, -pos.n, pos.n, -pos.n, pos.n);
}
