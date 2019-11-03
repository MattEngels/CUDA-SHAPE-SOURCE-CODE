/*****************************************************************************************
                                                                                  slice.c

Compute and display the intersection between a model's surface and a user-specified slice
plane.  This plane is specified via three numbers: the "slice_long" and "slice_lat"
parameters, which together determine the outward unit normal to the plane; and the
"slice_offset" parameter, the distance (which can be negative) along this normal from the
origin to the plane.

The coordinates of this 1-D intersection contour -- the xy coordinates of all
intersections between the model's sides (edges) and the slice plane -- are written to a
disk file.  The xy plane is defined as follows: first imagine viewing the model from a
position within the body-fixed equatorial plane at longitude slice_long, with the
body-fixed +z axis pointing straight upward; then rotate the body-fixed +z axis directly
toward yourself through an angle equal to latitude slice_lat.  In other words, x and y are
observer coordinates, where we have treated slice_long and slice_lat just as we would
treat the "view_long" and "view_lat" parameters for the "view" action.

Although the routine tries to write the points for each model component's intersection
contour in the correct order, it may get it wrong if a contour has a complex shape -- and
it can't possibly succeed if a contour is broken into multiple disjoint "islands."  In
this case one can take the output text file, edit it by hand to correct the order (or to
subdivide a contour so that there is a separate group of points for each "island"), and
then turn on the "slice_read" parameter so that the routine will READ the coordinates from
disk rather than computing the coordinates and writing them to disk.

Additionally, an image of the model, viewed from the perspective defined by the
"slice_viewlong" and "slice_viewlat" parameters, is written to disk; this image shows the
slice plane cutting the model, with the 1-D intersection contour (where not hidden by
closer portions of the model's surface) marked in bright white.  If the "mark_unseen"
parameter is turned on, sections of the contour in "unseen" (unobserved or poorly
constrained) parts of the surface will be bright yellow rather than bright white.  The
+x axis is marked on the edge of the slice plane with a small white square, and the
+y axis with a small black square.

The routine normally uses linear interpolation between the points of each model
component's intersection contour in order to ensure a gap-free curve on the image.
However, if the order of points is incorrect or if there are multiple disjoint "islands"
(see above), interpolation will produce a nonsensical curve; in this case one should turn
off interpolation by turning off the "slice_dointerp" parameter.

The size of the slice plane in the image is determined by the "slice_planefrac" parameter;
the brightness of blank sky lying behind the slice plane is determined by the
"slice_skyfactor" parameter; and the brightness of portions of the model's surface that
lie behind the slice plane is determined by the "slice_dimfactor" parameter.

The text file containing the contour's xy coordinates, whether for input or for output
(see above), is slice_DDD_sdd_ooo_km.dat: DDD is slice_long expressed in degrees,
integer-rounded, and shifted to the range [0,360); sdd is slice_lat expressed in degrees,
integer-rounded, and preceded by the plus or minus sign; and ooo is slice_offset expressed
in km, given as an integer if possible or to three decimal places (with a decimal point
included) if not.

The output image file has an even longer name: slice_DDD_sdd_ooo_km_view_DDD_sdd.pgm,
where the second "DDD_sdd" corresponds to the slice_viewlong and slice_viewlat parameters.
The image file suffix is ".ppm" (color) rather than ".pgm" (grayscale) if "mark_unseen" is
turned on.

There are also "slice_sunlong" and "slice_sunlat" and "slice_scatlaw" and "slice_shadows"
parameters that behave exactly as do the corresponding parameters for the "view" action,
and that change the output image filename in exactly the same way.

Modified 2014 February 19 by CM:
    Replace the "is_optical" argument to the plot_surface routine with the "iradlaw"
        argument
    Bug fix: When slice_scatlaw = optical, need to call the apply_photo routine to compute
        brightness values

Modified 2013 June 24 by CM:
    Let the plot_surface routine print the image filename to the screen rather than doing
        it here
    Add "posmax" argument to the annotate_plot routine and remove "is_optical" routine

Modified 2009 November 15 by CM:
    Eliminated unused variables

Modified 2009 July 29 by CM:
    Add "color_output" argument to the "write_pnm" routine

Modified 2009 April 10 by CM:
    Add the ability to add plot annotations: spin vector, COM cross,
        principal-axis shafts, angular momentum vector.  (As a
        result the output image file may be ppm rather than pgm.)
        Use the "annotate_plot" routine (in write_pos.c) for this so
        as not to duplicate code.  The new "slice_posfactor"
        parameter controls the size of the annotations.
    Remove the code for plotting the asteroid's surface model in the
        image and writing the image to disk: rely instead on the
        "plot_surface" and "write_pnm" routines (in write_pos.c).

Modified 2007 August 10 by CM:
    Eliminate unused variable and initialize uninitialized variables

Modified 2007 August 4 by CM:
    Add orbit_offset and body arguments to posvis routine and remove
        facet argument
    Add c (component) argument to radlaw routine
    Add body argument to apply_photo routine
    Add body, bodyill, comp, and compill matrices to POS frames

Written 2007 February 21 by CM
*****************************************************************************************/

#include "head.h"

void get_longlat_string( char *outstring, int maxSize,
                         double longitude, double latitude);
int point_is_visible( double x_obs, double y_obs, double z_obs,
                      struct mod_t *mod, struct pos_t *pos, int **color);


void slice( struct par_t *par, struct mod_t *mod)
{
  double orbit_offset[3] = {0.0, 0.0, 0.0};
                                                                                                
  FILE *fp;
  char slicelonglatstring[8], viewlonglatstring[8], sunlonglatstring[8],
       name[MAXLEN], offsetstring[MAXLEN];
  int i, j, k, l, c, ns, s, v0, v1, n, n2, m, n_interpolate,
      n_intersections_max, nplane, ngroups, g, color_output, iradlaw;
  int **intersection_is_visible, *n_intersections=NULL,
      **plane_is_visible, **color;
  double phi, theta, psi, posmax, solar_phase, intensityfactor,
         plane_unitnorm[3], v0_vec[3], side_vec[3], numer, denom,
         side_param, intersection_vec[3], xmean, ymean, dx, dy, dz,
         x_obs, y_obs, z_obs, maxbrightness, sa[3][3], vs[3][3],
         plane_vec[3], plane_skybrightness, plane_brightfactor,
         spin_body[3], spin_ecl[3];
  double ***intersection_coords, *x_intersection, *y_intersection,
         *z_intersection, *phi_intersection, **brightness, **z;
  struct pos_t pos;

  /*----------------------------------------------------*/
  /*  Generate strings that will be used in file names  */
  /*----------------------------------------------------*/

  /*  Get the integer-rounded longitude and latitude of the unit normal
      to the slice plane for use in the text file name;
      longitude range is [0,360) deg and latitude range is [-90,+90] deg.  */

  get_longlat_string( slicelonglatstring, 8,
                      par->slice_long, par->slice_lat);

  /*  Now do the same for the viewing longitude and latitude and,
      if necessary, for the Sun's longitude and latitude           */

  get_longlat_string( viewlonglatstring, 8,
                      par->slice_viewlong, par->slice_viewlat);
  if (par->slice_shadows)
    get_longlat_string( sunlonglatstring, 8,
                        par->slice_sunlong, par->slice_sunlat);

  /*  Get a string that shows the slice offset (in integer
      form if possible) for use in the text file name       */

  intifpossible( offsetstring, MAXLEN, par->slice_offset, SMALLVAL, "%.3f");

  /*---------------------------------------------------------------------*/
  /*  Generate or read the coordinates of the slice intersection points  */
  /*---------------------------------------------------------------------*/

  /*  Assign the sa transformation matrix that takes us from body-fixed
      ("asteroid") coordinates to slice-plane (projected) coordinates,
      corresponding to the latitude and longitude for the outward unit
      normal to the slice plane                                          */

  phi = par->slice_long + PIE/2;       /* Euler angle 0 */
  theta = PIE/2 - par->slice_lat;      /* Euler angle 1 */
  psi = 0.0;                             /* Euler angle 2 */
  euler2mat( sa, phi, theta, psi);

  /*  Get the unit normal to the slice plane  */

  plane_unitnorm[0] = cos(par->slice_lat)*cos(par->slice_long);
  plane_unitnorm[1] = cos(par->slice_lat)*sin(par->slice_long);
  plane_unitnorm[2] = sin(par->slice_lat);

  /*  Generate the name of the text file  */

  sprintf( name, "slice_%s_%s_km.dat", slicelonglatstring, offsetstring);

  /*  Read the coordinates from this file if requested;
      otherwise compute them and write them to this file  */

  if (par->slice_read) {

      /*  Open the text input file  */

      FOPEN( fp, name, "r");

      /*  Read the number of groups of intersection points and the
          number of intersection points in each group so that we
          can allocate storage for the intersection point coordinates  */

      ngroups = getint( fp);
      n_intersections_max = -999;
      for (g=0; g<ngroups; g++) {
        n_intersections[g] = getint( fp);
        n_intersections_max = MAX( n_intersections_max, n_intersections[g]);
        for (n=1; n<=n_intersections[g]; n++) {
          x_obs = getdouble( fp);
          y_obs = getdouble( fp);
        }
      }
      intersection_coords = d3tensor( 0, ngroups-1, 1, n_intersections_max, 0, 2);

      /*  Rewind the file, then read and store the xy slice-plane
          coordinates of the intersection points in each group     */

      fseek( fp, 0, SEEK_SET);
      ngroups = getint( fp);
      for (g=0; g<ngroups; g++) {
        n_intersections[g] = getint( fp);
        for (n=1; n<=n_intersections[g]; n++) {
          intersection_coords[g][n][0] = getdouble( fp);
          intersection_coords[g][n][1] = getdouble( fp);
          intersection_coords[g][n][2] = par->slice_offset;
        }
      }

  } else {

      /*  Open the text output file  */

      FOPEN( fp, name, "w");
      fprintf( fp, "{slice normal %7.3f deg longitude %+7.3f deg latitude, offset %+.3f km}\n",
               R2D*par->slice_long, R2D*par->slice_lat, par->slice_offset);

      /*  One group of intersection points per model component  */
 
      ngroups = mod->shape.ncomp;
      fprintf( fp, "\n%d {number of groups of intersection points}\n", ngroups);

      /*  Allocate storage for the number of slice/object intersections for each
          group (component) and for the xy coordinates of these intersection points  */

      n_intersections = ivector( 0, ngroups-1);
      n_intersections_max = -999;
      for (g=0; g<ngroups; g++)
        n_intersections_max = MAX( n_intersections_max, mod->shape.comp[g].real.ns);
      intersection_coords = d3tensor( 0, ngroups-1, 1, n_intersections_max, 0, 2);

      /*  Loop through all groups (components)  */

      for (g=0; g<ngroups; g++) {
        n_intersections[g] = 0;

        /*  Loop through all sides (edges) of this component */

        ns = mod->shape.comp[g].real.ns;
        x_intersection = vector( 1, ns);
        y_intersection = vector( 1, ns);
        phi_intersection = vector( 1, ns);
        xmean = 0.0;
        ymean = 0.0; 
        for (s=0; s<ns; s++) {

          /*  Get the displacement vector of one of the two vertices
              that terminate this side, plus the vector that points
              from this vertex to the other terminal vertex           */

          v0 = mod->shape.comp[g].real.s[s].v[0];
          v1 = mod->shape.comp[g].real.s[s].v[1];
          for (j=0; j<=2; j++) {
            v0_vec[j] = mod->shape.comp[g].real.v[v0].x[j];
            side_vec[j] = mod->shape.comp[g].real.v[v1].x[j]
                              - mod->shape.comp[g].real.v[v0].x[j];
          }

          /*  Test whether or not the slice plane intersects this side  */

          denom = dot( plane_unitnorm, side_vec);
          if (denom != 0.0) {
            numer = par->slice_offset - dot( plane_unitnorm, v0_vec);
            side_param = numer/denom;
            if (side_param >= 0.0 && side_param <= 1.0) {
              n_intersections[g]++;
              for (j=0; j<=2; j++)
                intersection_vec[j] = v0_vec[j] + side_param*side_vec[j];
              cotrans( intersection_vec, sa, intersection_vec, 1);
              x_intersection[n_intersections[g]] = intersection_vec[0];
              y_intersection[n_intersections[g]] = intersection_vec[1];
              xmean += x_intersection[n_intersections[g]];
              ymean += y_intersection[n_intersections[g]];
            }
          }
        }

        /*  Sort the intersection points by increasing azimuth angle (with
            respect to their centroid) and then write them to disk.
            Sorting by azimuth is appropriate for typical intersection
            contours, but it could get the points out of order if the
            contour has a complex shape -- and it can't possibly succeed
            if the contour is broken into multiple disjoint "islands."      */

        fprintf( fp, "\n%d {number of intersections for group %d: xy coords follow}\n",
                 n_intersections[g], g);
        if (n_intersections[g] > 0) {
          xmean /= n_intersections[g];
          ymean /= n_intersections[g];
          for (n=1; n<=n_intersections[g]; n++) {
            phi_intersection[n]
               = atan2( y_intersection[n] - ymean, x_intersection[n] - xmean );
            if (phi_intersection[n] < 0.0)
              phi_intersection[n] += 2*PIE;
          }
          sort3( n_intersections[g], phi_intersection, x_intersection, y_intersection);
          for (n=1; n<=n_intersections[g]; n++) {
            intersection_coords[g][n][0] = x_intersection[n];
            intersection_coords[g][n][1] = y_intersection[n];
            intersection_coords[g][n][2] = par->slice_offset;
            fprintf( fp, "    %13.6e  %13.6e\n",
                     x_intersection[n], y_intersection[n]);
          }
        }

        /*  Deallocate storage  */

        free_vector( x_intersection, 1, ns);
        free_vector( y_intersection, 1, ns);
        free_vector( phi_intersection, 1, ns);

        /*  Loop back to do the next group (component)  */
      }
  }

  /*  Close the text file  */

  fclose( fp);

  /*--------------------*/
  /*  Create the image  */
  /*--------------------*/

  /*  Set up the POS structure  */

  pos.n = (par->pos_pixels-1)/2;
  pos.b = matrix( -pos.n, pos.n, -pos.n, pos.n);
  pos.cosi = matrix( -pos.n, pos.n, -pos.n, pos.n);
  pos.cose = matrix( -pos.n, pos.n, -pos.n, pos.n);
  pos.z = matrix( -pos.n, pos.n, -pos.n, pos.n);
  pos.body = imatrix( -pos.n, pos.n, -pos.n, pos.n);
  pos.comp = imatrix( -pos.n, pos.n, -pos.n, pos.n);
  pos.f = imatrix( -pos.n, pos.n, -pos.n, pos.n);
  pos.km_per_pixel = par->pos_width/(2.0*pos.n);
  pos.cosill = matrix( -pos.n, pos.n, -pos.n, pos.n);
  pos.zill = matrix( -pos.n, pos.n, -pos.n, pos.n);
  pos.bodyill = imatrix( -pos.n, pos.n, -pos.n, pos.n);
  pos.compill = imatrix( -pos.n, pos.n, -pos.n, pos.n);
  pos.fill = imatrix( -pos.n, pos.n, -pos.n, pos.n);
  if (par->slice_scatlaw == OPTICALVIEW)
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

  /*
      We need to fool the posvis routine in order to produce a view from the
      desired perspective.  Normally posvis produces the body-to-observer
      coordinate transformation matrix oa by using the oe (ecliptic-to-observer)
      and ae (ecliptic-to-body) matrices associated with the data for that
      observing epoch: oa = oe*transpose(ae).  Here we have no data, so we must
      create oe and ae from scratch.

      If we set ae = identity matrix and oe = the matrix corresponding to the
      direction given by our specified viewing longitude and latitude, we'll get
      the view seen by an "observer" stationed somewhere along that direction.
  */

  for (i=0; i<=2; i++)
    for (j=0; j<=2; j++)
      pos.ae[i][j] = (j == i) ? 1.0 : 0.0;

  phi = par->slice_viewlong + PIE/2;       /* Euler angle 0 */
  theta = PIE/2 - par->slice_viewlat;      /* Euler angle 1 */
  psi = 0.0;                                 /* Euler angle 2 */
  euler2mat( pos.oe, phi, theta, psi);

  /*  Assign the se transformation matrix so that it
      corresponds to the specified solar latitude and longitude  */

  if (par->slice_shadows) {
      phi = par->slice_sunlong + PIE/2;    /* Euler angle 0 */
      theta = PIE/2 - par->slice_sunlat;   /* Euler angle 1 */
      psi = 0.0;                             /* Euler angle 2 */
      euler2mat( pos.se, phi, theta, psi);
  } else if (par->slice_scatlaw == OPTICALVIEW) {
      for (i=0; i<=2; i++)
        for (j=0; j<=2; j++)
          pos.se[i][j] = pos.oe[i][j];
  }

  /*  Compute the solar phase angle  */

  if (par->slice_shadows)
    solar_phase = acos( cos(par->slice_lat) * cos(par->slice_sunlat)
                            * cos(par->slice_long - par->slice_sunlong)
                        + sin(par->slice_lat) * sin(par->slice_sunlat)  );
  else
    solar_phase = 0.0;

  /*  Compute the optical intensity factor for apply_photo  */

  intensityfactor = pow( pos.km_per_pixel/AU, 2.0);

  /*  Clear the POS view and then fill it in from the desired perspective  */

  posclr( &pos);
  par->posbnd = 0;
  for (c=0; c<mod->shape.ncomp; c++)
    if (posvis( &mod->shape.comp[c].real, orbit_offset, &pos,
                (int) par->pos_smooth, 0, 0, c))
      par->posbnd = 1;
  if (par->posbnd)
    printf("WARNING: View extends beyond POS frame\n");

  /*  Now view the model from the source (sun) and get the facet number
      and distance toward the source of each pixel in this projected view;
      use this information to determine which POS pixels are shadowed       */
 
  par->posbnd = 0;
  if (par->slice_shadows) {
    for (c=0; c<mod->shape.ncomp; c++)
      if (posvis( &mod->shape.comp[c].real, orbit_offset, &pos, 0, 1, 0, c))
        par->posbnd = 1;
    if (par->posbnd)
      printf("WARNING: View from Sun extends beyond POS frame\n");

    /*  Identify and mask out shadowed POS pixels  */

    posmask( &pos, par->mask_tol);
  }

  /*  If we're using the optical scattering law for the image,
      call apply_photo to compute the POS pixel values          */

  if (par->slice_scatlaw == OPTICALVIEW)
    apply_photo( mod, 0, solar_phase, intensityfactor, &pos, 0, s, i);


  /*  Figure out the name of the image file  */

  color_output = (par->mark_unseen || par->plot_angmom ||
                  par->plot_pa[0]  || par->plot_pa[1]  || par->plot_pa[2]);
  if (color_output) {
      if (par->slice_shadows)
        sprintf( name, "slice_%s_%s_km_view_%s_sun_%s.ppm",
                 slicelonglatstring, offsetstring,
                 viewlonglatstring, sunlonglatstring);
      else
        sprintf( name, "slice_%s_%s_km_view_%s.ppm",
                 slicelonglatstring, offsetstring, viewlonglatstring);
  } else {
      if (par->slice_shadows)
        sprintf( name, "slice_%s_%s_km_view_%s_sun_%s.pgm",
                 slicelonglatstring, offsetstring,
                 viewlonglatstring, sunlonglatstring);
      else
        sprintf( name, "slice_%s_%s_km_view_%s.pgm",
                 slicelonglatstring, offsetstring, viewlonglatstring);
  }

  /*  Start the plot by filling in the asteroid surface
      (this will also display the image filename to the screen)  */

  iradlaw = (par->slice_scatlaw == RADARVIEW) ? 0 : -1;
  printf("#\n");
  plot_surface( par, mod, par->slice_scatlaw, iradlaw, name,
                &maxbrightness, &posmax, &pos, color, brightness, z);
  printf("#\n");
  fflush(stdout);

  /*  Get the (initial) spin vector in ecliptic coordinates  */

  for (i=0; i<=2; i++)
    spin_body[i] = mod->spin.omega[i].val;
  cotrans( spin_ecl, pos.ae, spin_body, -1);

  /*  Add each of the plot annotations that has been specified:
      spin vector, center of mass, angular momentum vector       */

  annotate_plot( par, mod, spin_ecl, maxbrightness, posmax,
                 &pos, color, brightness, z);

  /*  Get the vs transformation matrix that takes us from
      slice-plane coordinates to viewing (observer) coordinates  */

  mtrnsps( vs, sa);
  mmmul( vs, pos.oe, vs);

  /*  Add the slice plane to the image.  Loop through slice-plane
      pixels and transform each one to viewing coordinates so that
      we can check its status and adjust the corresponding
      view-plane pixel's brightness level accordingly.  Use the
      "plane_is_visible" array to keep track of which view-plane
      pixels have already had their brightness adjusted.            */

  plane_is_visible = imatrix( -pos.n, pos.n, -pos.n, pos.n);
  for (k=(-pos.n); k<=pos.n; k++)
    for (l=(-pos.n); l<=pos.n; l++)
      plane_is_visible[k][l] = 0;

  nplane = (int) (par->slice_planefrac * pos.n);
  plane_skybrightness = par->slice_skyfactor * maxbrightness;
  plane_brightfactor = 1.0 - par->slice_dimfactor;

  for (i=(-nplane); i<=nplane; i++)
    for (j=(-nplane); j<=nplane; j++) {
      plane_vec[0] = i*pos.km_per_pixel;
      plane_vec[1] = j*pos.km_per_pixel;
      plane_vec[2] = par->slice_offset;
      cotrans( plane_vec, vs, plane_vec, 1);
      k = iround( plane_vec[0]/pos.km_per_pixel);
      l = iround( plane_vec[1]/pos.km_per_pixel);
      if (abs(k) <= pos.n && abs(l) <= pos.n && !plane_is_visible[k][l]) {

        /*  Points in the slice plane that are in front of blank sky are
            assigned a constant brightness; points in the slice plane
            that are in front of the model cause the model's surface
            brightness to be decreased by a constant factor               */

        if (z[k][l] < -0.99*HUGENUMBER) {
            brightness[k][l] = plane_skybrightness;
            z[k][l] = plane_vec[2];
            plane_is_visible[k][l] = 1;
        } else if (plane_vec[2] > z[k][l]) {
            brightness[k][l] = plane_brightfactor * brightness[k][l];
            z[k][l] = plane_vec[2];
            plane_is_visible[k][l] = 1;
        }

        /*  Mark the slice-plane +x axis with a white
            square and the +y axis with a black square  */

        if (plane_is_visible[k][l]) {
          if (i >= 0.92*nplane && abs(j) <= 0.04*nplane)
            brightness[k][l] = maxbrightness;
          else if (abs(i) <= 0.04*nplane && j >= 0.92*nplane)
            brightness[k][l] = 0.0;
        }
      }
    }

  free_imatrix( plane_is_visible, -pos.n, pos.n, -pos.n, pos.n);

  /*  Add the boundary between the asteroid and the slice plane,
      making sure to include only points that are not hidden
      behind closer portions of the surface.  Use the
      "intersection_is_visible" matrix to keep track of which
      POS pixels already are known to be part of the visible
      boundary, so that we don't waste further time on them.      */ 

  intersection_is_visible = imatrix( -pos.n, pos.n, -pos.n, pos.n);
  for (k=(-pos.n); k<=pos.n; k++)
    for (l=(-pos.n); l<=pos.n; l++)
      intersection_is_visible[k][l] = 0;

  for (g=0; g<ngroups; g++) {
    if (n_intersections[g] > 0) {

      /*  Allocate storage for this group  */

      x_intersection = vector( 1, n_intersections[g]);
      y_intersection = vector( 1, n_intersections[g]);
      z_intersection = vector( 1, n_intersections[g]);

      /*  Get the observer coordinates of the
          intersection points for this component  */

      for (n=1; n<=n_intersections[g]; n++) {
        for (j=0; j<=2; j++)
          intersection_vec[j] = intersection_coords[g][n][j];
        cotrans( intersection_vec, vs, intersection_vec, 1);
        x_intersection[n] = intersection_vec[0];
        y_intersection[n] = intersection_vec[1];
        z_intersection[n] = intersection_vec[2];
      }

      /*  Determine the number of points to interpolate between each pair of
          intersection points.  Dense interpolation ensures that the visible
          portion of the intersection contour will not have any gaps.
          However, if the points for this group (a) are out of order due to
          a complex contour shape or (b) form multiple disjoint "islands"
          then turning off interpolation (via the "slice_dointerp" parameter)
          will prevent nonsensical connections from being added to the image.  */

      if (par->slice_dointerp)
        n_interpolate = MAX( 100, (100*pos.n)/n_intersections[g]);
      else
        n_interpolate = 1;   /* don't interpolate between points */

      /*  Add to the image any intersection points (and any
          interpolated points between them) that aren't hidden from view  */

      for (n=1; n<=n_intersections[g]; n++) {
        n2 = (n < n_intersections[g]) ? (n + 1) : 1;
        dx = (x_intersection[n2] - x_intersection[n])/n_interpolate;
        dy = (y_intersection[n2] - y_intersection[n])/n_interpolate;
        dz = (z_intersection[n2] - z_intersection[n])/n_interpolate;
        for (m=0; m<n_interpolate; m++) {
          x_obs = x_intersection[n] + m*dx;
          y_obs = y_intersection[n] + m*dy;
          z_obs = z_intersection[n] + m*dz;
          k = iround( x_obs/pos.km_per_pixel);
          l = iround( y_obs/pos.km_per_pixel);
          if (!intersection_is_visible[k][l] &&
                     point_is_visible( x_obs, y_obs, z_obs, mod, &pos, color)) {
            intersection_is_visible[k][l] = 1;
            brightness[k][l] = maxbrightness;
          }
        }
      }

      /*  Deallocate storage for this group  */

      free_vector( x_intersection, 1, n_intersections[g]);
      free_vector( y_intersection, 1, n_intersections[g]);
      free_vector( z_intersection, 1, n_intersections[g]);
    }
  }

  free_imatrix( intersection_is_visible, -pos.n, pos.n, -pos.n, pos.n);

  /*  Write the POS view to disk as a pgm or ppm image
      depending on how parameters like "mark_unseen" are set  */

  write_pnm( posmax, pos.n, color, brightness, color_output, name);

  /*  Clean up storage space  */

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
  free_ivector( n_intersections, 0, ngroups-1);
  free_d3tensor( intersection_coords, 0, ngroups-1, 1, n_intersections_max, 0, 2);
}


/*  Given floating-point longitude and latitude in radians as input, output
    a string that shows integer-rounded longitude and latitude, for use as
    part of a filename.  The output longitude range is [0,360) deg and the
    output latitude range is [-90,+90] deg.  If the output string isn't
    long enough (8 characters including the terminating null) to hold the
    longitude and latitude, it is instead set to the null string.            */

void get_longlat_string( char *outstring, int maxSize,
                         double longitude, double latitude)
{
  int rounded_long_deg, rounded_lat_deg;
  double long_deg_dbl, lat_deg_dbl;

  if (maxSize < 8) {
    strcpy(outstring, "");
    return;
  }

  long_deg_dbl = R2D*longitude;
  lat_deg_dbl = R2D*latitude;
  if (fabs(lat_deg_dbl) > 90.000001) {
    lat_deg_dbl -= 360.0*floor((lat_deg_dbl + 90.0)/360.0);
    if (lat_deg_dbl > 90.000001) {
      lat_deg_dbl = 180.0 - lat_deg_dbl;
      long_deg_dbl += 180.0;
    }
  }
  long_deg_dbl = (long_deg_dbl + 0.5) - 360.0*floor((long_deg_dbl + 0.5)/360.0);
  rounded_long_deg = (int) floor(long_deg_dbl);
  rounded_lat_deg = (int) floor(lat_deg_dbl + 0.5);

  sprintf( outstring, "%03d_%+03d", rounded_long_deg, rounded_lat_deg);
}


/*  Routine point_is_visible uses code adapted from posvis.c to
    determine whether or not the object/slice intersection point
    at observer coordinates (x_obs, y_obs, z_obs) is visible to
    us as we view the object normal to the slice plane -- that is,
    as we view it along the -z axis in observer coordinates.

    The routine is careful in that it considers the z-coordinates
    of the vertices of individual model facets rather than the
    z-coordinates assigned to POS pixels; the latter values are
    crude in that they refer only to the POS pixel center.          */

int point_is_visible( double x_obs, double y_obs, double z_obs,
                      struct mod_t *mod, struct pos_t *pos, int **color)
{
  int c, f, i, i_obs, j_obs, imin, imax, jmin, jmax, is_visible;
  double v0[3], v1[3], v2[3], s, t, den, zmin, n[3], oa[3][3];
  struct vertices_t *verts;

  /*  Get POS pixel numbers for this point and make sure that
      the point lies within the boundaries of the POS frame    */

  i_obs = iround( x_obs/pos->km_per_pixel);
  j_obs = iround( y_obs/pos->km_per_pixel);
  if ( (i_obs < (-pos->n)) || (i_obs > pos->n) ||
       (j_obs < (-pos->n)) || (j_obs > pos->n)    ) {
    is_visible = 0;
    return is_visible;
  }

  /*  If this POS pixel has already been "covered" by a
      plot annotation, the interesection isn't visible here  */

  if (color[i_obs][j_obs] >= 2) {
    is_visible = 0;
    return is_visible;
  }

  /*  Get transformation matrix oa that takes us from
      body-fixed ("asteroid") coordinates to observer coordinates  */

  mtrnsps(oa, pos->ae);
  mmmul( oa, pos->oe, oa);

  /*  Loop through all model components (or until
      it's clear that our point is not visible)    */

  is_visible = 1;
  c = 0;
  do {
      verts = &mod->shape.comp[c].real;

      /*  Loop through all facets of this model component
          (or until it's clear that our point is not visible)  */

      f = 0;
      do {

          /*  Get the normal to this facet in body-fixed (asteroid)
              coordinates and convert it to observer coordinates     */

          for (i=0; i<=2; i++)
            n[i] = verts->f[f].n[i];
          cotrans( n, oa, n, 1);

          /*  Consider this facet further only if its normal points
              somewhat towards the observer rather than away         */

          if (n[2] > 0.0) {

            /*  Consider the vertices of this facet: convert the three sets of vertex
                coordinates from body-fixed to observer coordinates, then find the
                rectangular region (in POS pixels) that contains the projected facet.
                Assume zero center-of-mass offset due to binary orbital motion.        */

            cotrans( v0, oa, verts->v[verts->f[f].v[0]].x, 1);
            cotrans( v1, oa, verts->v[verts->f[f].v[1]].x, 1);
            cotrans( v2, oa, verts->v[verts->f[f].v[2]].x, 1);
            imin = iround(MIN( v0[0], MIN( v1[0], v2[0]))/pos->km_per_pixel - SMALLVAL);
            imax = iround(MAX( v0[0], MAX( v1[0], v2[0]))/pos->km_per_pixel + SMALLVAL);
            jmin = iround(MIN( v0[1], MIN( v1[1], v2[1]))/pos->km_per_pixel - SMALLVAL);
            jmax = iround(MAX( v0[1], MAX( v1[1], v2[1]))/pos->km_per_pixel + SMALLVAL);

            /*  If this rectangular region also contains our point,
                this facet and our point might share a common POS pixel  */

            if ( (i_obs >= imin) && (i_obs <= imax) && 
                 (j_obs >= jmin) && (j_obs <= jmax)    ) {

              /*  Compute parameters s(x,y) and t(x,y) which define a facet's
                  surface as
                                 z = z0 + s*(z1-z0) + t*(z2-z1)

                  where z0, z1, and z2 are the z-coordinates at the vertices.
                  The conditions 0 <= s <= 1 and 0 <= t <= s require our
                  point to be "within" the (projected) perimeter of facet f.   */ 

              den = 1/( (v1[0]-v0[0])*(v2[1]-v1[1]) - (v2[0]-v1[0])*(v1[1]-v0[1]) );
              s = ( (x_obs-v0[0])*(v2[1]-v1[1]) - (v2[0]-v1[0])*(y_obs-v0[1]) ) * den;
              if ( (s >= -SMALLVAL) && (s <= 1.0+SMALLVAL) ) {
                t = ( (v1[0]-v0[0])*(y_obs-v0[1]) - (x_obs-v0[0])*(v1[1]-v0[1]) ) * den;
                if ( (t >= -SMALLVAL) && (t <= s+SMALLVAL) ) {

                  /*  Our point lies "within" this (projected) facet.
                      If our point's z-coordinate is less than this facet's
                      minimum z-coordinate, the facet hides the point from view.  */

                  zmin = MIN( v0[2], MIN( v1[2], v2[2]));
                  if (z_obs < zmin-SMALLVAL)
                    is_visible = 0;

                }  /* end if 0 <= t <= s (our point lies "within" this facet) */
              }  /* end if 0 <= s <= 1 */
            }  /* end if facet and point might share a common POS pixel */
          }  /* end if facet points towards observer */

          /*  If our point still might be visible,
              loop back to the next facet in this component  */

          f++;
      } while (is_visible && f<verts->nf);

      /*  If our point still might be visible, loop back to the next component  */

      c++;
  } while (is_visible && c<mod->shape.ncomp);

  return is_visible;
}
