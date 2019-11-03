/*****************************************************************************************
                                                                              write_pos.c

Write a model plane-of-sky image to disk, first adding various plotting elements if
desired:

    -- arrow along the spin vector
    -- cross marking the projected center of mass
    -- X marking the subradar/subearth point
    -- shafts marking the positive ends of the three principal axes
    -- arrow along the angular momentum vector

If the PA-shafts or angular momentum vector are used, the output is a colored ppm image;
otherwise it's a grayscale pgm image.

Modified 2014 February 12 by CM:
    Add "ilaw" argument to the radlaw routine to implement multiple radar scattering laws.
        To accomplish this, replace the "is_optical" argument to the write_pos and
        plot_surface routines by the "iradlaw" argument.

Modified 2013 July 28 by CM:
    Implement "write_highlight" parameter to highlight selected facets in the output image

Modified 2013 June 25 by CM:
    Remove restriction that images output for optical data can only be annotated with a
        spin-vector arrow; as a result, can remove "is_optical" argument to the
        annotate_plot routine
    Add "is_optical" and "name" arguments to the plot_surface routine, and use that
        routine to display the name of the image file being written and (if relevant) the
        maximum pixel value vs. the value of the "optposmax" or "radposmax" parameter
    Add "posmax" argument to the annotate_plot, plot_arrow, and plot_subradar routines,
        and use this information to scale the brightness of plot annotations so that they
        aren't affected by the "optposmax" and "radposmax" parameters.  (Don't need to do
        this for the plot_com routine since the COM annotation is black.)
    Enable the "optposmax" and "radposmax" parameters to work even when when "sky_optlaw"
        or "sky_radlaw" (respectively) is set to "lambertian"

Modified 2012 July 4 by CM:
    Initialize variable in "plot_surface" routine to avoid compilation warning

Modified 2012 March 9 by CM:
    Fix bug (introduced 2012 March 5) in "write_pnm" routine: grayscale images weren't
        showing the spin vector even when "plot_spinvec" was turned on

Modified 2012 March 5 by CM:
    Implement "view_highlight" parameter to highlight selected facets in the output image

Modified 2011 August 10 by CM:
    Bug fix for assigning the right colors to principal-axis shafts and
        making sure that they define a right-handed triple

Modified 2010 September 1 by CM:
    Initialize variable to avoid compilation warning

Modified 2010 June 15 by CM:
    Adjust "annotate_plot" routine to work with the new "map" action

Modified 2009 July 29 by CM:
    New "color_output" argument to write_pos is passed to the "write_pnm"
        routine, explicitly telling it whether or not to produce a colored
        image (rather than having it base this decision on pixel colors)

Modified 2009 April 10 by CM:
    Move code out of the main "write_pos" routine and into the new
        "plot_surface" "annotate_plot" and "write_pnm" routines,
        whose headers are in shape2.h rather than here so that other
        routines can call them

Modified 2009 April 3 by CM:
    Add ability to plot angular momentum vector via the new
        "plot_angmom" parameter
    Implement the "pierce_spinvec" and "pierce_angmom" parameters to
        determine the appearance of spin vectors and angular momentum
        vectors in POS images

Modified 2007 August 12 by CM:
    Add "body" parameter to plot_com routine so that the "orbit" action
        doesn't draw a center-of-mass cross on an eclipsed body

Modified 2007 August 11 by CM:
    In plot_arrow routine, correct math error in finding the sides of
        the arrowhead
    In plot_arrow routine, prevent arrow length from being increased
        by more than a factor of 2 for nearly pole-on views
    Permit the "sizefactor" parameter to be negative: this indicates
        that the spin vector arrow is NOT to be lengthened to
        compensate for slant angle.  The size of the arrow (and of the
        other plot annotations) is determined by |sizefactor|.

Modified 2007 August 4 by CM:
    Add "sizefactor" parameter to plot_arrow, plot_com, and plot_subradar
        routines; also add "body" parameter to plot_subradar routine
    Add "com_obs" parameter to plot_arrow routine and adjust the routine
        so that the arrow (or shaft) crosses the observer z=0 plane at
        the observer (x,y) coordinates given by com_obs
    Move headers for plot_arrow, plot_com, and plot_subradar routines
        from here to shape2.h so that other routines can call them.
        Add "maxbrightness" parameter to each of these routines so that
        this move can be made.
    Add c (component) argument to radlaw routine
    Add comp matrix to POS frames

Modified 2006 September 1 by CM and MCN:
    When "mark_unseen" parameter is turned on, add check that facet
        number (*pos).f[i][j] is nonnegative

Modified 2006 June 21 by CM:
    For POS renderings, change res to km_per_pixel

Modified 2005 August 18 by CM:
    Correct bug introduced at the preceding change: the spin vector
        wasn't showing up in grayscale images (because those pixels
        were being set to base grayscale level 0 rather than 255)

Modified 2005 August 1 by CM:
    Enable the "sky_radlaw" and "sky_optlaw" parameters for the "write"
        action (scattering laws to be used for radar and optical POS
        images)

Modified 2005 July 22 by CM:
    Enable the "mark_unseen" parameter for the "write" action:
        unseen regions of the model are shaded yellow

Modified 2005 January 25 by CM:
    Take care of uninitialized variables to avoid compilation warnings

Written 2004 March 27 by CM
*****************************************************************************************/

#include "head.h"


void write_pos( struct par_t *par, struct mod_t *mod, struct pos_t *pos,
                double spin_ecl[3], int iradlaw, int color_output, char *name)
{
  unsigned char scatlaw;
  int n, is_optical;
  int **color;
  double maxbrightness, posmax;
  double **brightness, **z;

  /*
      Prepare storage matrices to list three quantities for each
         line of sight (each POS pixel):

         1) color type (0 = asteroid surface, 1 = unseen ast. surf., 2 = spin vector, etc.)
         2) brightness (grayscale)
         3) z-coordinate (distance toward observer) of the closest
               plotting element (asteroid surface, arrow, etc.)
  */

  n = (*pos).n;
  color = imatrix( -n, n, -n, n);
  brightness = matrix( -n, n, -n, n);
  z = matrix( -n, n, -n, n);

  /*  Start the plot by filling in the asteroid surface  */

  is_optical = (iradlaw < 0);
  scatlaw = (is_optical) ? (*par).sky_optlaw : (*par).sky_radlaw;
  plot_surface( par, mod, scatlaw, iradlaw, name, &maxbrightness, &posmax,
                pos, color, brightness, z);

  /*  Add each of the plot annotations that has been specified:
      spin vector, center of mass, subradar point, angular momentum vector  */

  annotate_plot( par, mod, spin_ecl, maxbrightness, posmax,
                 pos, color, brightness, z);

  /*  Output image to disk as a pnm (pgm or ppm) file  */

  write_pnm( posmax, n, color, brightness, color_output, name);

  /*  Clean up storage space  */

  free_imatrix( color, -n, n, -n, n);
  free_matrix( brightness, -n, n, -n, n);
  free_matrix( z, -n, n, -n, n);
}


/*  Use the desired scattering law to fill in the asteroid surface
    (must call this procedure first, BEFORE adding any plot annotations)   */

void plot_surface( struct par_t *par, struct mod_t *mod, unsigned char scatlaw,
                   int iradlaw, char *name, double *maxbrightness, double *posmax,
                   struct pos_t *pos, int **color, double **brightness, double **z)
{
  int do_highlight, highlight_comp[MAXCHOSENFACETS], highlight_facet[MAXCHOSENFACETS],
      i, j, n, c, f, m, matched_facet, is_optical;

  matched_facet = 0;  /* avoid compilation warning */

  if ((*par).action == WRITE && (*par).write_highlight) {
      do_highlight = 1;
      for (i=0; i<MAXCHOSENFACETS; i++) {
        highlight_comp[i] = (*par).write_comp[i];
        highlight_facet[i] = (*par).write_facet[i];
      }
  } else if ((*par).action == VIEW && (*par).view_highlight) {
      do_highlight = 1;
      for (i=0; i<MAXCHOSENFACETS; i++) {
        highlight_comp[i] = (*par).view_comp[i];
        highlight_facet[i] = (*par).view_facet[i];
      }
  } else {
      do_highlight = 0;
      for (i=0; i<MAXCHOSENFACETS; i++)
        highlight_comp[i] = highlight_facet[i] = -1;
  }
  n = (*pos).n;
  *maxbrightness = -HUGENUMBER;
  for (i=(-n); i<=n; i++)
    for (j=(-n); j<=n; j++) {
      if ((*par).mark_unseen || do_highlight) {
          if ((*pos).f[i][j] < 0) {
              color[i][j] = 0;       /* blank sky: just use any value */
          } else {
              c = (*pos).comp[i][j];
              f = (*pos).f[i][j];
              if (do_highlight) {
                matched_facet = 0;
                m = 0;
                while (!matched_facet && m < MAXCHOSENFACETS && highlight_facet[m] >= 0)
                  if (highlight_comp[m] == c && highlight_facet[m] == f)
                    matched_facet = 1;
                  else
                    m++;
              }
              if (do_highlight && matched_facet)
                color[i][j] = 2;
              else if ((*par).mark_unseen && !(*mod).shape.comp[c].real.f[f].seen)
                color[i][j] = 1;
              else
                color[i][j] = 0;
          }
      } else {
          color[i][j] = 0;
      }
      if (scatlaw == OPTICALVIEW) {
          brightness[i][j] = (*pos).b[i][j];
      } else if (scatlaw == RADARVIEW) {
          if ((*pos).cose[i][j] > 0.0)
            brightness[i][j] = radlaw( &(*mod).photo, iradlaw, (*pos).cose[i][j],
                                       (*pos).comp[i][j], (*pos).f[i][j]);
          else
            brightness[i][j] = 0.0;
      } else {
          brightness[i][j] = (*pos).cose[i][j];
      }
      *maxbrightness = MAX( *maxbrightness, brightness[i][j]);
      z[i][j] = (*pos).z[i][j];
    }

  is_optical = (iradlaw < 0);
  *posmax = (is_optical) ? (*par).optposmax: (*par).radposmax;

  /*  Print the name of the image file to the screen, and if the "optposmax" or
      "radposmax" parameter is nonzero (for optical or radar data, respectively),
      also print the value of that parameter vs. the image's maximum pixel value   */

  if (*posmax != 0.0)
    printf("# %s  (max pixel value = %10.4e, bright white >= %10.4e)\n",
           name, *maxbrightness, *posmax);
  else
    printf("# %s\n", name);
  fflush(stdout);
}


/*  Add plot annotations: spin vector, center of mass, subradar point,
                          principal axes, and angular momentum vector   */

void annotate_plot( struct par_t *par, struct mod_t *mod, double spin_ecl[3],
                    double maxbrightness, double posmax, struct pos_t *pos,
                    int **color, double **brightness, double **z)
{
  int i, j, do_com, do_subradar, do_pa, do_angmom, pa_indices[3], flip_pa2;
  double posfactor, oa[3][3], com_obs[3], spin_obs[3], pmoment[3], ap[3][3],
         op[3][3], min_moment, max_moment, pa_obs[3], spin_body[3], angmom[3];

  /*  Initialize variable to avoid compilation warning  */

  posfactor = 0.0;

  /*  Figure out which annotations to add  */

  do_com = (*par).plot_com;
  do_subradar = (*par).plot_subradar && (*par).action == WRITE;
  do_pa =  (*par).plot_pa[0] || (*par).plot_pa[1] || (*par).plot_pa[2];
  do_angmom = (*par).plot_angmom;

  /*  Set the size of the plot annotations relative to the "standard" size  */

  if ((*par).action == WRITE)
    posfactor = (*par).write_posfactor;
  else if ((*par).action == SLICE)
    posfactor = (*par).slice_posfactor;
  else if ((*par).action == VIEW)
    posfactor = (*par).view_posfactor;
  else if ((*par).action == MAP)
    posfactor = (*par).map_posfactor;
  else
    bailout("annotate_plot routine cannot handle this action\n");

  /*  Get oa = matrix to transform body-fixed to observer coordinates  */

  mtrnsps( oa, (*pos).ae);
  mmmul( oa, (*pos).oe, oa);

  /*  Convert the model's center of mass to observer coordinates  */

  cotrans( com_obs, oa, (*mod).shape.com, 1);

  /*  Add each of the plotting elements which has been specified  */

  if ((*par).plot_spinvec) {

    /*  arrow along spin vector  */

    cotrans( spin_obs, (*pos).oe, spin_ecl, 1);
    plot_arrow( 3, com_obs, spin_obs, 1, (*par).pierce_spinvec, posfactor,
                maxbrightness, posmax, pos, color, brightness, z);
  }

  if (do_com) {

    /*  cross at projected center of mass  */

    plot_com( 4, 0, com_obs, posfactor, maxbrightness, pos, color, brightness, z);
  }

  if (do_subradar) {

    /*  X at subradar point  */

    plot_subradar( 5, 0, posfactor, maxbrightness, posmax, pos, color, brightness, z);
  }

  if (do_pa) {

    /*  cylindrical shaft (headless arrow) along positive end
        of one or more principal axes                          */

    /*  First, take the inertia tensor (computed assuming uniform density),
        diagonalize it in order to get the principal moments (pmoment)
        and the principal-axis-to-body-fixed coordinate transformation matrix (ap)  */

    diag_inertia( (*mod).shape.inertia, pmoment, ap);

    /*  Get op = matrix to transform principal-axis to observer coordinates  */

    mmmul( op, oa, ap);

    /*  Determine which principal axis is long vs. intermediate vs. short  */

    min_moment = HUGENUMBER;
    for (i=0; i<=2; i++)
      if (pmoment[i] < min_moment) {
        min_moment = pmoment[i];
        pa_indices[0] = i;
      }
    max_moment = -HUGENUMBER;
    for (i=0; i<=2; i++)
      if (i != pa_indices[0] && pmoment[i] >= max_moment) {
        max_moment = pmoment[i];
        pa_indices[2] = i;
      }
    for (i=0; i<=2; i++)
      if (i != pa_indices[0] && i != pa_indices[2])
        pa_indices[1] = i;

    /*  Check if we must flip the second principal axis PA2 to get a right-handed
        triple when we go from (PA1, PA2, PA3) to (long, intermediate, short)      */

    if ((pa_indices[0] == 0 && pa_indices[2] == 1) || (pa_indices[0] == 1 && pa_indices[2] == 2)
                                                   || (pa_indices[0] == 2 && pa_indices[2] == 0))
      flip_pa2 = 1;
    else
      flip_pa2 = 0;
    
    /*  Create the cylindrical shaft(s)  */

    for (i=0; i<=2; i++) {
      if ((*par).plot_pa[i]) {
        if (flip_pa2 && pa_indices[i] == 1)
          for (j=0; j<=2; j++)
            pa_obs[j] = -op[j][pa_indices[i]];
        else
          for (j=0; j<=2; j++)
            pa_obs[j] = op[j][pa_indices[i]];
        plot_arrow( 6+i, com_obs, pa_obs, 0, 0, posfactor,
                    maxbrightness, posmax, pos, color, brightness, z);
      }
    }
  }

  if (do_angmom) {

    /*  arrow along angular momentum vector  */

    /*  Transform the intrinsic spin vector from ecliptic to body-fixed coordinates  */

    cotrans( spin_body, (*pos).ae, spin_ecl, 1);

    /*  Compute the angular momentum vector in principal-axis coordinates

        For an NPA rotator, the spin state is evolved (via Euler's equations)
        with body-fixed coordinates treated as principal-axis coordinates;
        so to ensure that the angular momentum vector is constant in time,
        we must also equate these two coordinate systems here.  (For a
        PA rotator it doesn't matter: only one spin component is nonzero.)     */

    for (i=0; i<=2; i++)
      angmom[i] = (*mod).spin.inertia[i].val * spin_body[i];

    /*  Transform the angular momentum vector to observer coordinates  */

    cotrans( angmom, oa, angmom, 1);

    /*  Create the arrow  */

    plot_arrow( 9, com_obs, angmom, 1, (*par).pierce_angmom, posfactor,
                maxbrightness, posmax, pos, color, brightness, z);
  }
}


/*  Add an arrow (if "has_head" is true) or else a cylindrical shaft (if it is false);
    this arrow/shaft pierces all the way through the model (if "pierce_model" is true)
    or else starts at the center of mass (if it is false)                               */

void plot_arrow( int colortype, double com_obs[3], double arrowvec[3], int has_head,
                 int pierce_model, double sizefactor, double maxbrightness, double posmax,
                 struct pos_t *pos, int **color, double **brightness, double **zval)
{
  int i, j, n;
  double km_per_pixel, s_min, s_max, R, f, g, s_shaftmax, R_headmax, headslope, theta_a,
         phi_a, costheta_a, sintheta_a, cosphi_a, sinphi_a, x, y, z, factor1, factor2,
         factor3, factor4, sterm1, sterm2, zterm1, zterm2, rho, s, cosemax, scale;
  double **cose;

  /*  Use arrowvec (the arrow direction in observer coordinates) to determine
      theta_a and phi_a, the corresponding spherical coordinates               */

  theta_a = atan2(sqrt(arrowvec[0]*arrowvec[0] + arrowvec[1]*arrowvec[1]), arrowvec[2]);
  costheta_a = cos(theta_a);
  sintheta_a = sin(theta_a);
  phi_a = atan2(arrowvec[1], arrowvec[0]);
  cosphi_a = cos(phi_a);
  sinphi_a = sin(phi_a);

  /*  Initialize variables  */

  n = (*pos).n;
  km_per_pixel = (*pos).km_per_pixel;
  cose = matrix( -n, n, -n, n);
  for (i=(-n); i<=n; i++)
    for (j=(-n); j<=n; j++)
      cose[i][j] = -HUGENUMBER;     /* temporary storage prior to brightness scaling */

  /*  Set the arrow parameters, using cylindrical coordinates
      (rho, phi, s) with the s-axis along the arrow's length   */

  if (has_head) {

      /*  arrow  */

      s_max = fabs(sizefactor)*0.9*n*km_per_pixel;  /* position of tip of arrowhead (km) */
      R = s_max * 0.065;                            /* radius of arrow shaft (km) */
      if (sizefactor > 0.0)
        s_max /= MAX( fabs(sintheta_a), 0.5);       /* lengthen arrow to compensate for slant */
      f = 0.20;                                     /* (head length) / (origin-to-tip length) */
      g = 1.65;                                     /* (head base radius) / (shaft radius) */
      R_headmax = g*R;                              /* arrowhead's base radius (km) */
      headslope = R_headmax/(f*s_max);              /* arrowhead slope */
  } else {

      /*  cylindrical shaft (headless arrow)  */

      s_max = fabs(sizefactor)*0.9*n*km_per_pixel;
      R = s_max * 0.05;
      f = 0.0;
      g = 0.0;
      R_headmax = g*R;
      headslope = HUGENUMBER;
  }
  s_shaftmax = (1 - f)*s_max;                  /* position of top  of shaft (km) */
  s_min = (pierce_model) ? -s_max : 0.0;       /* position of base of shaft (km) */

  /*  Loop through all POS pixels, checking whether each line of sight
      intersects any of the various arrow components (base, shaft, etc.).

      A given line of sight may intersect more than one component, or may
      intersect the same component twice; hence we keep track of distance
      towards the observer (observer z coordinate) in the zval matrix and
      always keep the CLOSEST intersection found so far.                   */

  /*  For mathematical simplicity, x and y in the code below are similar to
      the x and y observer coordinates but are measured from the COM
      xy position.  z, on the other hand, is exactly the z observer
      coordinate and is NOT measured from the COM z position, since we must
      check it against the observer z coordinates (zval) of other plot
      elements to see which element is closest to us and hence will appear
      in the final plot.                                                     */

  for (i=(-n); i<=n; i++) {
    x = i*km_per_pixel - com_obs[0];
    for (j=(-n); j<=n; j++) {
      y = j*km_per_pixel - com_obs[1];
      factor1 = -x*sinphi_a + y*cosphi_a;

      /*  base of arrow shaft  */

      if (costheta_a < 0.0) {
        z = (s_min - x*sintheta_a*cosphi_a - y*sintheta_a*sinphi_a) / costheta_a
            + com_obs[2];
        factor2 = (x*cosphi_a + y*sinphi_a - s_min*sintheta_a) / costheta_a;
        rho = sqrt(factor1*factor1 + factor2*factor2);
        if (rho <= R && z > zval[i][j]) {
          color[i][j] = colortype;
          zval[i][j] = z;
          cose[i][j] = -costheta_a;
        }
      }

      /*  sides of arrow shaft  */

      if (fabs(factor1) <= R && sintheta_a != 0.0) {
        factor2 = (x*cosphi_a + y*sinphi_a)/sintheta_a;
        factor3 = sqrt(R*R - factor1*factor1);
        z = factor2*costheta_a + factor3/sintheta_a + com_obs[2];
        s = factor2 + factor3*costheta_a/sintheta_a;
        if (s >= s_min && s <= s_shaftmax && z > zval[i][j]) {
          color[i][j] = colortype;
          zval[i][j] = z;
          cose[i][j] = factor3*sintheta_a / R;
        }
      }

      /*  base of arrowhead (if present), else top of shaft  */

      if (costheta_a != 0.0) {
        z = (s_shaftmax - x*sintheta_a*cosphi_a - y*sintheta_a*sinphi_a) / costheta_a
            + com_obs[2];
        factor2 = (x*cosphi_a + y*sinphi_a - s_shaftmax*sintheta_a) / costheta_a;
        rho = sqrt(factor1*factor1 + factor2*factor2);
        if (has_head && costheta_a < 0.0 && rho <= R_headmax && z > zval[i][j]) {
            color[i][j] = colortype;
            zval[i][j] = z;
            cose[i][j] = -costheta_a;
        } else if (!has_head && costheta_a > 0.0 && rho <= R && z > zval[i][j]) {
            color[i][j] = colortype;
            zval[i][j] = z;
            cose[i][j] = costheta_a;
        }
      }

      /*  sides of arrowhead (if present)  */

      if (has_head) {
        factor2 = (headslope*headslope + 1)*costheta_a*costheta_a - 1;
        if (factor2 != 0.0) {

            /*  (arrowhead slope) != (arrow slant w.r.t. line of sight)
                ==> The arrowhead sides are never viewed edge-on from base
                    to tip, so there could be up to two intersection points.  */

            factor3 = headslope*(x*cosphi_a + y*sinphi_a - s_max*sintheta_a);
            factor4 = factor3*factor3 + factor2*factor1*factor1;
            if (factor4 >= 0.0) {
              factor4 = sqrt(factor4);
              zterm1 = ( -(headslope*headslope + 1)*sintheta_a*costheta_a*(x*cosphi_a + y*sinphi_a)
                         + headslope*headslope*s_max*costheta_a )
                       / factor2;
              zterm2 = factor4 / factor2;
              sterm1 = ( -x*sintheta_a*cosphi_a - y*sintheta_a*sinphi_a
                         + headslope*headslope*s_max*costheta_a*costheta_a)
                       / factor2;
              sterm2 = zterm2*costheta_a;
              z = zterm1 + zterm2 + com_obs[2];
              s = sterm1 + sterm2;
              if (s >= s_shaftmax && s < s_max && z > zval[i][j]) {
                color[i][j] = colortype;
                zval[i][j] = z;
                cose[i][j] = -factor4
                               / (headslope * sqrt(headslope*headslope + 1) * (s_max - s));
              }
              z = zterm1 - zterm2 + com_obs[2];
              s = sterm1 - sterm2;
              if (s >= s_shaftmax && s < s_max && z > zval[i][j]) {
                color[i][j] = colortype;
                zval[i][j] = z;
                cose[i][j] = factor4
                               / (headslope * sqrt(headslope*headslope + 1) * (s_max - s));
              }
            }
        } else {

            /*  (arrowhead slope) == (arrow slant w.r.t. line of sight)
                ==> There exists a line of sight that runs tangentially along
                    the arrowhead from base to tip, so apart from this line
                    there can be at most one intersection point.                */

            factor3 = x*cosphi_a + y*sinphi_a - s_max*sintheta_a;
            factor4 = (sintheta_a/costheta_a)*factor3;
            if (factor3 != 0.0) {
              z = ((x*x + y*y - s_max*s_max*sintheta_a*sintheta_a)/factor4 - factor4) / 2
                  + com_obs[2];
              s = ((x*x + y*y - s_max*s_max*sintheta_a*sintheta_a)/factor4
                   + (sintheta_a/costheta_a)*(x*cosphi_a + y*sinphi_a + s_max*sintheta_a)) / 2;
              if (s >= s_shaftmax && s < s_max && z > zval[i][j]) {
                color[i][j] = colortype;
                zval[i][j] = z;
                cose[i][j] = -costheta_a*factor3/(s_max - s);
              }
            }
        }
      }
    }
  }

  /*  Scale the arrow brightness, including an extra scaling to undo the
      effects of the "optposmax" or "radposmax" parameter on the arrow    */

  cosemax = -HUGENUMBER;
  for (i=(-n); i<=n; i++)
    for (j=(-n); j<=n; j++)
      cosemax = MAX( cosemax, cose[i][j]);

  if (cosemax > 0.0) {
    scale = (posmax != 0.0) ? posmax/cosemax : maxbrightness/cosemax;

    for (i=(-n); i<=n; i++)
      for (j=(-n); j<=n; j++)
        if (cose[i][j] > -HUGENUMBER)
          brightness[i][j] = scale*cose[i][j];
  }

  free_matrix( cose, -n, n, -n, n);
}


/*  Add a cross that marks the projected center of mass  */

void plot_com( int colortype, int body, double com_obs[3], double sizefactor, double maxbrightness,
               struct pos_t *pos, int **color, double **brightness, double **z)
{
  int n, i0, j0, halfwidth, halfstrokewidth, imin, imax, jmin, jmax, i, j;
  double s, f;

  /*  Set the parameters of the cross that will mark the projected COM  */

  n = (*pos).n;
  s = MAX( fabs(sizefactor)*0.05*(2*n + 1), 5.0);   /* width and height of cross */
  f = 0.17;                        /* (width of either stroke) / (full width of cross) */

  /*  Blacken the pixels that define a cross at the projected COM  */

  i0 = (int) floor(com_obs[0]/(*pos).km_per_pixel + 0.5);
  j0 = (int) floor(com_obs[1]/(*pos).km_per_pixel + 0.5);

  if (abs(i0) <= n && abs(j0) <= n) {

    halfwidth = (int) (s/2);
    halfstrokewidth = (int) (f*s/2);
    imin = MAX( i0 - halfwidth, -n);
    imax = MIN( i0 + halfwidth, n);
    jmin = MAX( j0 - halfwidth, -n);
    jmax = MIN( j0 + halfwidth, n);

    for (i=imin; i<=imax; i++)
      for (j=jmin; j<=jmax; j++)
        if ((abs(i-i0) <= halfstrokewidth || abs(j-j0) <= halfstrokewidth)
            && (*pos).body[i][j] == body && (*pos).z[i][j] == z[i][j]) {
          color[i][j] = colortype;
          brightness[i][j] = maxbrightness;
        }
  }
}


/*  Add an X that marks the subradar/subearth point  */

void plot_subradar( int colortype, int body, double sizefactor, double maxbrightness,
                    double posmax, struct pos_t *pos, int **color, double **brightness,
                    double **z)
{
  int n, i, j, i0, j0, halfwidth, imin, imax, jmin, jmax;
  double zmax, s, f, abs_slope, max_abs_intercept, x_brightness;

  /*  Initialize variables to avoid compilation warnings  */

  i0 = j0 = 0;

  /*  Find the subradar point  */

  zmax = -HUGENUMBER;
  n = (*pos).n;
  for (i=(-n); i<=n; i++)
    for (j=(-n); j<=n; j++)
      if ((*pos).z[i][j] > zmax && (*pos).body[i][j] == body) {
        zmax = (*pos).z[i][j];
        i0 = i;
        j0 = j;
      }

  /*  Set the parameters of the X that will mark the subradar point

      The four slanting lines which form the sides of the two strokes have
      slope = +/- 1/(1-f) and pass f*s/(2*(1-f)) above/below the X's center  */

  s = MAX( fabs(sizefactor)*0.05*(2*n + 1), 5.0);   /* width and height of X */
  f = 0.33;                        /* (width at top of either stroke) / (full width of X) */
  abs_slope = 1/(1 - f);
  max_abs_intercept = (f*s/2) / (1 - f);   /* relative to center of X */

  /*  Set the brightness of the pixels within the X, including a scaling factor
      that will undo the effects of the "optposmax" or "radposmax" parameter on the X  */

  x_brightness = (posmax != 0.0) ? posmax : maxbrightness;

  /*  Loop through all POS pixels within the square that bounds the X,
      and blacken all pixels that fall within the two strokes of the X  */

  halfwidth = (int) (s/2);
  imin = MAX( i0 - halfwidth, -n);
  imax = MIN( i0 + halfwidth, n);
  jmin = MAX( j0 - halfwidth, -n);
  jmax = MIN( j0 + halfwidth, n);

  for (i=imin; i<=imax; i++)
    for (j=jmin; j<=jmax; j++)
      if ( ( fabs((j-j0) - abs_slope*(i-i0)) <= max_abs_intercept ||
             fabs((j-j0) + abs_slope*(i-i0)) <= max_abs_intercept    )
           && z[i][j] == (*pos).z[i][j]) {
        color[i][j] = colortype;
        brightness[i][j] = x_brightness;
      }

}


/*  Write completed POS image to disk as a pnm (pgm or ppm) image  */

void write_pnm( double posmax, int n,
                int **color, double **brightness, int color_output, char *name)
{
  /* max grayscale levels for asteroid surface, unseen asteroid surface,
     highlighted facet, spin vector, COM, subradar point */
  const int levels[6] = {255, 255, 0, 255, 0, 0};

  /* max RGB levels for various plotting elements */
  const int clevels[10][3] = {{255, 255, 255},     /* asteroid surface  */
                              {255, 255,   0},     /* unseen ast. surf. */
                              {  0,   0, 255},     /* highlighted facet */
                              {255,   0, 255},     /* spin vector       */
                              {  0,   0,   0},     /* COM               */
                              {127,  63,   0},     /* subradar point    */
                              {255,   0,   0},     /* long PA           */
                              {  0, 255,   0},     /* intermediate PA   */
                              {  0, 127, 255},     /* short PA          */
                              {127, 255, 255}};    /* angular momentum  */

  int i, j, icolor;
  double ***cbrightness;

  /*  Write the image to disk  */

  if (color_output) {

      /*  color (ppm) output  */

      /*  Set RGB levels according to the maximum RGB levels and
          the relative brightness values determined above         */

      cbrightness = d3tensor( -n, n, -n, n, 0, 2);

      for (i=(-n); i<=n; i++)
        for (j=(-n); j<=n; j++)
          for (icolor=0; icolor<=2; icolor++)
            cbrightness[i][j][icolor] = brightness[i][j]*clevels[color[i][j]][icolor]/255.0;

      if (posmax == 0.0)
        wimasppm0( cbrightness, -n, n, -n, n, 0, 0, 0, name);
      else
        wimasppmsc( cbrightness, -n, n, -n, n, 0.0, posmax, 0, 0, 0, name);

      free_d3tensor( cbrightness, -n, n, -n, n, 0, 2);

  } else {

      /*  grayscale (pgm) output  */

      /*  Set grayscale levels according to the maximum levels and
          the relative brightness values determined above           */

      for (i=(-n); i<=n; i++)
        for (j=(-n); j<=n; j++)
          brightness[i][j] = brightness[i][j]*levels[color[i][j]]/255.0;

      if (posmax == 0.0)
        wimaspgm0( brightness, -n, n, -n, n, 0, 0, 0, name);
      else
        wimaspgmsc( brightness, -n, n, -n, n, 0.0, posmax, 0, 0, 0, name);
  }
}
