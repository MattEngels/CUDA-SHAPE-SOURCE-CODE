/***************************************************************************
                                                               checkposet.c

If a plane-of-sky fit image is to be produced by resampling a POS frame
(sky rendering), check whether or not the fit image is wide enough to
"contain" all of the nonzero pixels in the POS frame.

input POS frame (sky rendering) im has dimensions [x1..x2][y1..y2]

The resampling will be done (via the "resampim" routine) over a rectangular
region centered at pixel (x0, y0) in the POS frame, initially extending
xwidth pixels horizontally and ywidth pixels vertically from pixel center
to pixel center (not edge to edge), and then rotated counterclockwise by
rotangle radians prior to resampling.  All five of these parameters are
floating-point.

checkposet returns 0 if all nonzero pixels in the POS frame lie entirely
within the plane-of-sky fit image; otherwise it returns 1.

If the return value is 1, badposet_logfactor is set equal to the logarithm
of a ratio of areas: the area that the plane-of-sky fit image would have
had to have in order to contain all nonzero pixels in the POS frame,
divided by the actual area of the plane-of-sky fit image.  Thus this
logarithm is greater than zero.  If the return value is 0,
badposet_logfactor is set to 0.0.

Written 2009 April 3 by CM
***************************************************************************/

#include "basic.h"
#include "../nr/nr.h"
#include "../macros/func.h"


int checkposet( double **im, int x1, int x2, int y1, int y2,
                double x0, double xwidth, double y0, double ywidth,
                double rotangle, double *badposet_logfactor)
{
  int i, j, badposet;
  int **nonzero;
  double x_fit_lim1, x_fit_lim2, y_fit_lim1, y_fit_lim2, cosangle, sinangle,
         x_sky, y_sky, x_fit, y_fit, x_fit_min, x_fit_max, y_fit_min, y_fit_max;

  /*  Initialize variables; the "0.5" is needed because xwidth and ywidth
      are widths from pixel center to pixel center, not from edge to edge  */

  x_fit_lim1 = x_fit_min = -xwidth/2 - 0.5;
  x_fit_lim2 = x_fit_max =  xwidth/2 + 0.5;
  y_fit_lim1 = y_fit_min = -ywidth/2 - 0.5;
  y_fit_lim2 = y_fit_max =  ywidth/2 + 0.5;
  cosangle = cos(rotangle);
  sinangle = sin(rotangle);

  nonzero = imatrix( x1, x2+1, y1, y2+1);
  for (i=x1; i<=x2+1; i++)
    for (j=y1; j<=y2+1; j++)
      nonzero[i][j] = 0;

  /*  Flag the four corners of every nonzero POS pixel  */

  for (i=x1; i<=x2; i++)
    for (j=y1; j<=y2; j++)
      if (im[i][j] > 0.0)
        nonzero[i][j] = nonzero[i+1][j] = nonzero[i][j+1] = nonzero[i+1][j+1] = 1;

  /*  Check every corner of every nonzero POS pixel to see whether it is bad
      -- that is, whether it lies outside the rotated plane-of-sky fit image.
      Do this by computing each corner's displacement from the fit image's
      center along each of the fit image's two (rotated) symmetry dimensions.
      If a corner is bad, flag the image as bad via the "badposet" flag, and
      update the minimum and maximum displacements along each dimension.       */

  badposet = 0;

  for (i=x1; i<=x2+1; i++) {
    x_sky = (i - 0.5) - x0;
    for (j=y1; j<=y2+1; j++) {
      y_sky = (j - 0.5) - y0;
      if (nonzero[i][j]) {
        x_fit =  x_sky*cosangle + y_sky*sinangle;
        y_fit = -x_sky*sinangle + y_sky*cosangle;
        if (x_fit < x_fit_lim1) {
            badposet = 1;
            x_fit_min = MIN( x_fit_min, x_fit);
        } else if (x_fit > x_fit_lim2) {
            badposet = 1;
            x_fit_max = MAX( x_fit_max, x_fit);
        }
        if (y_fit < y_fit_lim1) {
            badposet = 1;
            y_fit_min = MIN( y_fit_min, y_fit);
        } else if (y_fit > y_fit_lim2) {
            badposet = 1;
            y_fit_max = MAX( y_fit_max, y_fit);
        }
      }
    }
  }

  /*  If the image is bad, compute a factor (greater than 1.0) equal to the
      area of the fit image as extended to include all nonzero POS pixels,
      divided by the area of the actual fit image, and then take the
      logarithm of that factor.                                              */

  if (badposet)
    *badposet_logfactor = log( (x_fit_max - x_fit_min)*(y_fit_max - y_fit_min)
                               / ((xwidth + 1)*(ywidth + 1)));
  else
    *badposet_logfactor = 0.0;

  /*  Clean up storage and return the "badposet" flag  */

  free_imatrix( nonzero, x1, x2+1, y1, y2+1);

  return badposet;
}
