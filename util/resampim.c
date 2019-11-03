/***************************************************************************
                                                                 resampim.c

Recenters, rotates, and resizes an image

input  image im1 has dimensions [x11..x12][y11..y12]
output image im2 has dimensions [x21..x22][y21..y22]

Imagine a rectangular sampling grid with the same number of rows and columns
as im2 has.  Overlay this grid on im1 with its rows parallel to im1's rows;
center it at floating-point im1 coordinates (x0, y0), with the CENTER of the
bottom-left pixel at coordinates (x0 - xwidth/2, y0 - ywidth/2) and the
CENTER of the top-right pixel at (x0 + xwidth/2, y0 + ywidth/2).  Now rotate
this resampling grid counterclockwise through angle rotangle about (x0, y0),
and then resample as per the specified interpolation and rebinning modes
(see below).  The resulting resampled image is im2.

Any portion of this region which lies outside the input image, or for which
some input pixels needed for interpolation lie outside the input image,
will be represented by zeroes in the output image.

interpolation_mode:
    0: no interpolation (just copy an image subset from input to output)
    1: nearest neighbor or block interpolation
    2: bilinear interpolation
    3: bicubic interpolation
    4: cubic convolution

rebin:
    Setting the "rebin" parameter to 1 permits a two-step process when
    "minifying" (undersampling) the input image: first recenter, rotate, and
    interpolate as described above to get a temporary image whose dimensions
    are similar to im1's dimensions while also being integer multiples of
    the desired output dimensions; then rebin this temporary image -- that
    is, average pixel values for blocks of adjacent pixels -- to get the
    significantly smaller output image.

resampim returns 0 if any output pixels, or any input pixels required for
interpolation, lie outside the boundaries of the input image; otherwise it
returns 1.

Modified 2005 July 14 by CM:
    Eliminated calls to bailout routine

Modified 2005 June 27 by CM:
    Renamed "round" function to "iround" to avoid conflicts

Modified 2005 March 1 by CM:
    Added "rotangle" parameter

Modified 2005 February 21 by CM:
    Added cubic convolution option
    Added "rebin" parameter

Modified 2005 January 18 by CM:
    Added bilinear and bicubic option, following section 3.6 of
        Numerical Recipes
    Added option for no interpolation (recentering/cropping only)
***************************************************************************/

#include "basic.h"
#include "../nr/nr.h"
#include "../util/util.h"
#include "../macros/func.h"

#define NONE 0
#define NEAREST 1
#define BILINEAR 2
#define BICUBIC 3
#define CUBICCONV 4

/*  CUBICPAR is a parameter used in the interpolation polynomial for
    cubic convolution.  A value of -1.0 yields the original TRW cubic
    convolution algorithm, whereas Park & Schowengerdt 1983 claim that
    -0.5 produces better results.  Note that this is the same as the
    "cubic" keyword to the IDL "interpolate" and "congrid" procedures.  */

#define CUBICPAR -1.0


double cubicconvpoly(double x)
{
  double absx, interp_poly;

  absx = fabs(x);
  if (absx < 1)
    interp_poly = (CUBICPAR + 2)*absx*absx*absx - (CUBICPAR + 3)*absx*absx + 1;
  else if (absx < 2)
    interp_poly = CUBICPAR*(absx*absx*absx - 5*absx*absx + 8*absx - 4);
  else
    interp_poly = 0.0;

  return interp_poly;
}


int resampim( double **im1, int x11, int x12, int y11, int y12,
              double **im2, int x21, int x22, int y21, int y22,
              double x0, double xwidth, double y0, double ywidth,
              double rotangle, int interpolation_mode, int rebin)
{
  int i, j, k, l, ix1, iy1, ret=1, xbinfactor, ybinfactor, xtemp1, xtemp2,
      ytemp1, ytemp2, do_rebinning, npixels_to_bin;
  double x1, y1, x1start, y1start, cosangle, sinangle, xstep, ystep,
         dx1_over_dxtemp, dx1_over_dytemp, dy1_over_dxtemp, dy1_over_dytemp,
         t, u, interpolated_func, interpolated_xgradient,
         interpolated_ygradient, xinterp;
  double **im1_xderiv, **im1_yderiv, **im1_xyderiv, *func_grid, *xderiv_grid,
         *yderiv_grid, *xyderiv_grid, *xpoly, **imtemp;

  /*  If we're not using interpolation, we can only rotate by an
      integer multiple of 90 degrees                              */

  cosangle = cos(rotangle);
  sinangle = sin(rotangle);
  if (interpolation_mode == NONE && fabs(cosangle) > 1.0e-10
                                 && fabs(sinangle) > 1.0e-10) {
    fprintf( stderr,
             "ERROR in resampim.c: can't use interpolation_mode = NONE with oblique rotangle\n");
    exit(2);
  }

  /*  Get the integer rebinning factors, relevant when
      the "rebin" flag is turned on and we are undersampling
      the input image by a large factor in either dimension   */

  if (rebin) {
      xbinfactor = iround( xwidth/(x22 - x21));
      xbinfactor = MAX( xbinfactor, 1);
      ybinfactor = iround( ywidth/(y22 - y21));
      ybinfactor = MAX( ybinfactor, 1);
      npixels_to_bin = xbinfactor*ybinfactor;
  } else {
      xbinfactor = ybinfactor = npixels_to_bin = 1;
  }
  do_rebinning = (rebin && npixels_to_bin > 1);
  if (do_rebinning) {

      /*  Rebinning: allocate memory for the temporary
                     (interpolated) image, imtemp       */

      xtemp1 = 1;
      xtemp2 = xbinfactor*(x22 - x21 + 1);
      ytemp1 = 1;
      ytemp2 = ybinfactor*(y22 - y21 + 1);
      imtemp = matrix( xtemp1, xtemp2, ytemp1, ytemp2);

  } else {

      /*  No rebinning: save time by pointing the "temporary"
                        image to the final output image        */

      xtemp1 = x21;
      xtemp2 = x22;
      ytemp1 = y21;
      ytemp2 = y22;
      imtemp = im2;
  }

  /*  Compute the step size within the input image for each
      column or row increment within the interpolated image  */

  if (interpolation_mode == NONE) {
      xstep = ystep = 1.0;
  } else {
      xstep = xwidth/(xtemp2 - xtemp1);
      ystep = ywidth/(ytemp2 - ytemp1);
  }
  dx1_over_dxtemp = xstep*cosangle;
  dx1_over_dytemp = ystep*sinangle;
  dy1_over_dxtemp = -xstep*sinangle;
  dy1_over_dytemp = ystep*cosangle;
  x1start = x0 - ((xtemp2 - xtemp1)/2.0)*dx1_over_dxtemp
               - ((ytemp2 - ytemp1)/2.0)*dx1_over_dytemp;
  y1start = y0 - ((xtemp2 - xtemp1)/2.0)*dy1_over_dxtemp
               - ((ytemp2 - ytemp1)/2.0)*dy1_over_dytemp;

  /*  Do the interpolation  */

  switch (interpolation_mode) {
  case NONE:
  case NEAREST:

      /*  The same code handles both cases:
              -- If both xstep and ystep are 1.0, we just copy a subset of the
                     input image to the output image
              -- Otherwise we do nearest-neighbor resampling                    */

      for (i=xtemp1; i<=xtemp2; i++) {
        x1 = x1start;
        y1 = y1start;
        for (j=ytemp1; j<=ytemp2; j++) {
          ix1 = iround( x1);
          iy1 = iround( y1);
          if (ix1 >= x11 && ix1 <= x12 && iy1 >= y11 && iy1 <= y12) {
              imtemp[i][j] = im1[ix1][iy1];
          } else {
              imtemp[i][j] = 0.0;
              ret = 0;
          }
          x1 += dx1_over_dytemp;
          y1 += dy1_over_dytemp;
        }
        x1start += dx1_over_dxtemp;
        y1start += dy1_over_dxtemp;
      }
      break;

  case BILINEAR:

      /*  Do bilinear interpolation:
              x1 and y1 are the (floating-point) coordinates, within
              input image im1, of output image imtemp's columns and rows  */

      for (i=xtemp1; i<=xtemp2; i++) {
        x1 = x1start;
        y1 = y1start;
        for (j=ytemp1; j<=ytemp2; j++) {
          ix1 = (int) floor(x1);
          iy1 = (int) floor(y1);
          if (ix1 >= x11 && ix1 < x12 && iy1 >= y11 && iy1 < y12) {
              t = x1 - ix1;
              u = y1 - iy1;
              imtemp[i][j] = (1 - t)*(1 - u)*im1[ix1][iy1]
                             + t*(1 - u)*im1[ix1+1][iy1]
                             + t*u*im1[ix1+1][iy1+1]
                             + (1 - t)*u*im1[ix1][iy1+1];
          } else {
              imtemp[i][j] = 0.0;
              ret = 0;
          }
          x1 += dx1_over_dytemp;
          y1 += dy1_over_dytemp;
        }
        x1start += dx1_over_dxtemp;
        y1start += dy1_over_dxtemp;
      }

      break;

  case BICUBIC:

      /*  For each pixel in the input image, use centered differencing to get
          get the first partial derivatives with respect to x and to y and
          the second partial derivative with respect to x and y                */

      im1_xderiv = matrix( x11, x12, y11, y12);
      im1_yderiv = matrix( x11, x12, y11, y12);
      im1_xyderiv = matrix( x11, x12, y11, y12);

      for (i=x11+1; i<x12; i++) {
        for (j=y11+1; j<y12; j++) {
          im1_xderiv[i][j] = (im1[i+1][j] - im1[i-1][j])/2;
          im1_yderiv[i][j] = (im1[i][j+1] - im1[i][j-1])/2;
          im1_xyderiv[i][j] = (im1[i+1][j+1] - im1[i+1][j-1] -
                               im1[i-1][j+1] + im1[i-1][j-1]   )/4;
        }
      }

      /*  Create four vectors which will hold the "function" value (pixel value)
          and the partial derivatives at the four corners of the "grid square"
          immediately surrounding a point in the input image at which we want the
          interpolated function value:
                corner 4 = top left,     corner 3 = top right,
                corner 1 = bottom left,  corner 2 = bottom right                     */

      func_grid = vector( 1, 4);
      xderiv_grid = vector( 1, 4);
      yderiv_grid = vector( 1, 4);
      xyderiv_grid = vector( 1, 4);

      /*  Do bicubic interpolation:
              x1 and y1 are the (floating-point) coordinates, within
              input image im1, of output image imtemp's columns and rows,
              and ix1 and ix2 are their next-lowest integer equivalents    */

      for (i=xtemp1; i<=xtemp2; i++) {
        x1 = x1start;
        y1 = y1start;
        for (j=ytemp1; j<=ytemp2; j++) {
          ix1 = (int) floor( x1);
          iy1 = (int) floor( y1);
          if (ix1 > x11 && ix1 < x12-1 && iy1 > y11 && iy1 < y12-1) {
              func_grid[1]    = im1[ix1][iy1];
              func_grid[2]    = im1[ix1+1][iy1];
              func_grid[3]    = im1[ix1+1][iy1+1];
              func_grid[4]    = im1[ix1][iy1+1];
              xderiv_grid[1]  = im1_xderiv[ix1][iy1];
              xderiv_grid[2]  = im1_xderiv[ix1+1][iy1];
              xderiv_grid[3]  = im1_xderiv[ix1+1][iy1+1];
              xderiv_grid[4]  = im1_xderiv[ix1][iy1+1];
              yderiv_grid[1]  = im1_yderiv[ix1][iy1];
              yderiv_grid[2]  = im1_yderiv[ix1+1][iy1];
              yderiv_grid[3]  = im1_yderiv[ix1+1][iy1+1];
              yderiv_grid[4]  = im1_yderiv[ix1][iy1+1];
              xyderiv_grid[1] = im1_xyderiv[ix1][iy1];
              xyderiv_grid[2] = im1_xyderiv[ix1+1][iy1];
              xyderiv_grid[3] = im1_xyderiv[ix1+1][iy1+1];
              xyderiv_grid[4] = im1_xyderiv[ix1][iy1+1];
              bcuint( func_grid, xderiv_grid, yderiv_grid, xyderiv_grid,
                      ix1, ix1+1, iy1, iy1+1, x1, y1,
                      &interpolated_func,
                      &interpolated_xgradient, &interpolated_ygradient);
              imtemp[i][j] = interpolated_func;
          } else {
              imtemp[i][j] = 0.0;
              ret = 0;
          }
          x1 += dx1_over_dytemp;
          y1 += dy1_over_dytemp;
        }
        x1start += dx1_over_dxtemp;
        y1start += dy1_over_dxtemp;
      }

      /*  Clean up storage space  */

      free_matrix( im1_xderiv, x11, x12, y11, y12);
      free_matrix( im1_yderiv, x11, x12, y11, y12);
      free_matrix( im1_xyderiv, x11, x12, y11, y12);
      free_vector( func_grid, 1, 4);
      free_vector( xderiv_grid, 1, 4);
      free_vector( yderiv_grid, 1, 4);
      free_vector( xyderiv_grid, 1, 4);

      break;

  case CUBICCONV:

      /*  For each column and each row of the output image, get the
          corresponding floating-point column and row number (x1 and y1)
          within the input image, the next-lowest integer equivalents
          (ix1 and iy1) of these column and row numbers, and the interpolation
          polynomial values (xpoly) used for cubic convolution along rows.
          Finally, do the cubic convolution: Interpolate along four adjacent
          rows (with each of these interpolations using pixel values from four
          adjacent columns), and then interpolate these four interpolated
          values along a column.                                                */

      xpoly = vector( -1, 2);

      for (i=xtemp1; i<=xtemp2; i++) {
        x1 = x1start;
        y1 = y1start;
        for (j=ytemp1; j<=ytemp2; j++) {
          ix1 = (int) floor( x1);
          iy1 = (int) floor( y1);
          imtemp[i][j] = 0.0;
          if (ix1 > x11 && ix1 < x12-1 && iy1 > y11 && iy1 < y12-1) {
              t = x1 - ix1;
              u = y1 - iy1;
              for (k=-1; k<=2; k++)
                xpoly[k] = cubicconvpoly(k-t);
              for (l=-1; l<=2; l++) {
                xinterp = 0.0;
                for (k=-1; k<=2; k++)
                  xinterp += xpoly[k]*im1[ix1+k][iy1+l];
                imtemp[i][j] += cubicconvpoly(l-u)*xinterp;
              }
          } else {
              ret = 0;
          }
          x1 += dx1_over_dytemp;
          y1 += dy1_over_dytemp;
        }
        x1start += dx1_over_dxtemp;
        y1start += dy1_over_dxtemp;
      }

      /*  Clean up storage space  */

      free_vector( xpoly, -1, 2);

      break;

  default:
      fprintf( stderr,
               "ERROR in resampim.c: don't recognize that interpolation mode\n");
      exit(2);
  }

  /*  Interpolation finished: Rebin if undersampling  */

  if (do_rebinning) {
    for (i=x21; i<=x22; i++)
      for (j=y21; j<=y22; j++) {
        im2[i][j] = 0.0;
        for (k=xtemp1+(i-x21)*xbinfactor; k<xtemp1+(i+1-x21)*xbinfactor; k++)
          for (l=ytemp1+(j-y21)*ybinfactor; l<ytemp1+(j+1-y21)*ybinfactor; l++)
            im2[i][j] += imtemp[k][l];
        im2[i][j] /= npixels_to_bin;
      }
    free_matrix( imtemp, xtemp1, xtemp2, ytemp1, ytemp2);
  }

  return ret;
}

#undef NONE
#undef NEAREST
#undef BILINEAR
#undef BICUBIC
#undef CUBICCONV
#undef CUBICPAR
