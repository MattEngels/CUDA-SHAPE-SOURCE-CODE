/*****************************************************************************************
                                                                                posmask.c

For bistatic data (lightcurves), identify shadowed plane-of-sky pixels and then mask them
out by setting cos(scattering angle) = 0.0.

The mask is a "plane-of-sky" view from the perspective of the light source (sun).  For
each POS pixel (as viewed from Earth) we find the corresponding mask pixel and check
whether the mask pixel is closer to the source than is the POS pixel.  If it is, it
shadows the POS pixel.

If the "tol" argument is nonzero, a mask pixel must be at least tol km closer to the
source than is the POS pixel which it shadows.  This allows for the imprecision that
results from our comparing pixels rather than model facets.

Modified 2014 August 14 by CM:
    Major bug fix: In test that a mask pixel has the potential to shadow the
        corresponding POS pixel, was referencing the POS image with the mask image's
        coordinates (im, jm) and the mask image with the POS image's coordinates (i, j)
        As a result, some pixels that should have been shadowed were left unshadowed.
    Bug fix: In test that a mask pixel has the potential to shadow the
        corresponding POS pixel, was using "abs" rather than "fabs" for double-precision
        variables i0_dbl and j0_dbl

Modified 2007 August 4 by CM:
    Add body, bodyill, comp, and compill matrices to POS frames

Modified 2006 June 21 by CM:
    Change res to km_per_pixel and resfact to pixels_per_km

Modified 2005 June 27 by CM:
    Renamed INFINITY constant to HUGENUMBER to avoid conflicts
    Renamed "round" function to "iround" to avoid conflicts

Modified 2005 January 25 by CM:
    Changed routine from type "int" to "void"

Modified 2005 January 20 by CM:
    Adjust/correct the bilinear interpolation scheme: now use standard
        bilinear interpolation (e.g., Numerical Recipes, section 3.6)
        whenever possible

Modified 2004 April 7 by CM:
    Improve the shadowing algorithm: Rather than simply using the
        distance towards the source of the *center* of the nearest
        mask pixel, use bilinear interpolation to refine this distance.
        Prior to this change, we would always conclude that POS pixel P
        is shadowed by mask pixel M when M's center is closer to the
        source than P's center, even if the precise (unrounded)
        projection of P's center onto M is NOT closer.

Modified 2004 February 11 by CM:
    Added comments
*****************************************************************************************/

#include "head.h"

void posmask( struct pos_t *pos, double tol)
{
  int i, j, im, jm, i1, j1, i2, j2, i_sign, j_sign, n;
  double x[3], so[3][3], pixels_per_km, i0_dbl, j0_dbl, zill, t, u, bignum;

  bignum = 0.99*HUGENUMBER;  /* z = -HUGENUMBER for blank-sky pixels */

  mtrnsps( so, (*pos).oe);
  mmmul( so, (*pos).se, so);    /* so takes obs into src coords */
  pixels_per_km = 1/(*pos).km_per_pixel;

  /*  Loop through all POS pixels  */

  n = (*pos).n;
  for (i=(-n); i<=n; i++) {               /* for each pixel in the */
    for (j=(-n); j<=n; j++) {             /* observer's view */
      if ((*pos).cose[i][j] != 0.0) {     /* if there's something there */
        x[0] = i*(*pos).km_per_pixel;     /* calculate 3D position */
        x[1] = j*(*pos).km_per_pixel;
        x[2] = (*pos).z[i][j];

        /*  Given the observer coordinates x of of POS pixel (i,j),
            find which pixel (im,jm) this corresponds to in the
            projected view as seen from the source (sun)             */

        cotrans( x, so, x, 1);           /* go into source coordinates */
        i0_dbl = x[0]*pixels_per_km;     /* unrounded (double precision) */
        j0_dbl = x[1]*pixels_per_km;
        im = iround( i0_dbl);            /* center of nearest pixel in mask */
        jm = iround( j0_dbl);

        /*  If the center of the projected pixel "seen" from the source
            (as determined by routine posvis) lies within the boundaries
            of the mask, projects onto the model rather than onto blank
            space, and represents a body, component, and facet different
            from those seen in the POS, calculate the distance from the
            mask pixel to the source and compare it to the distance from
            the POS pixel to the source.                                  */
        if (fabs(i0_dbl) < n && fabs(j0_dbl) < n
                  && (*pos).zill[im][jm] > -bignum
                  && ((*pos).f[i][j]    != (*pos).fill[im][jm]    ||
                      (*pos).comp[i][j] != (*pos).compill[im][jm] ||
                      (*pos).body[i][j] != (*pos).bodyill[im][jm]    )) {

          /*  Rather than using the distance towards the source of the
              CENTER of the mask pixel, use bilinear interpolation to
              get the distance towards the source at the point where
              the line between the source and the POS pixel's center
              intersects the mask pixel.                                */

          i1 = (int) floor( i0_dbl);
          j1 = (int) floor( j0_dbl);

          if ((*pos).zill[i1][j1]     > -bignum &&
              (*pos).zill[i1+1][j1]   > -bignum &&
              (*pos).zill[i1][j1+1]   > -bignum &&
              (*pos).zill[i1+1][j1+1] > -bignum    ) {

              /*  Do standard bilinear interpolation:
                      None of the four surrounding "grid square" pixels
                      in the mask is blank sky                           */

              t = i0_dbl - i1;
              u = j0_dbl - j1;
              zill = (1 - t)*(1 - u)*(*pos).zill[i1][j1]
                     + t*(1 - u)*(*pos).zill[i1+1][j1]
                     + t*u*(*pos).zill[i1+1][j1+1]
                     + (1 - t)*u*(*pos).zill[i1][j1+1];

          } else {

              /*  The following code block is a kludge:
                      One or more of the four surrounding "grid square"
                      pixels in the mask is blank sky, so standard
                      bilinear interpolation won't work                  */

              zill = (*pos).zill[im][jm];

              i_sign = (i0_dbl >= im) ? 1 : -1;
              i2 = im + i_sign;
              if (abs(i2) <= n && (*pos).zill[i2][jm] > -bignum) {
                  zill += fabs(i0_dbl - im)
                          * ((*pos).zill[i2][jm] - (*pos).zill[im][jm]);
              } else {
                  i2 = im - i_sign;
                  if (abs(i2) <= n && (*pos).zill[i2][jm] > -bignum)
                    zill -= fabs(i0_dbl - im)
                            * ((*pos).zill[i2][jm] - (*pos).zill[im][jm]);
              }

              j_sign = (j0_dbl >= jm) ? 1 : -1;
              j2 = jm + j_sign;
              if (abs(j2) <= n && (*pos).zill[im][j2] > -bignum) {
                  zill += fabs(j0_dbl - jm)
                          * ((*pos).zill[im][j2] - (*pos).zill[im][jm]);
              } else {
                  j2 = jm - j_sign;
                  if (abs(j2) <= n && (*pos).zill[im][j2] > -bignum)
                    zill -= fabs(j0_dbl - jm)
                            * ((*pos).zill[im][j2] - (*pos).zill[im][jm]);
              }

          }

          /*  If the interpolated point within the mask pixel is at
              least tol km closer to the source than is the center of
              the POS pixel, the facet represented by the mask pixel
              is shadowing the POS pixel: represent this by setting
              cos(scattering angle) = 0.0 for the POS pixel.           */

          if (zill - x[2] > tol)
            (*pos).cose[i][j] = 0.0;
        }
      }
    }
  }
}
