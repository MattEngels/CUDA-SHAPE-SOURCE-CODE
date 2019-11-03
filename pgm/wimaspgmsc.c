/***************************************************************************
                                                               wimaspgmsc.c

Dumps a double precision image im[][] to disk as a normalized pgm image.
User provides scaling limits.

Modified 2003 April 29 by CM:
    Truncate scaled pixel values to the range [0, 255] BEFORE
    writing to the (unsigned char) output, so that large values
    don't "wrap around" to small ones

Modified 2003 July 30 by CM:
    Add three parameters (values 0 or 1) to enable rotation/flipping
    of output image.  If both rotation and flipping are desired,
    rotation is performed first, so that xflip and yflip refer to
    the x and y dimensions of the *rotated* image.
***************************************************************************/

#include "basic.h"
#include "../macros/func.h"
#include "../pgm/pgm.h"

void wimaspgmsc( double **im, int xmin, int xmax, int ymin, int ymax, 
                 double min, double max,
                 int clockwiserot, int xflip, int yflip,
                 char *name)
{
  int i, j, k, npix, nc, nr, scaledpix, i1, i_incr, j1, j_incr, m, n;
  unsigned char *pgm;
  double scale;

  scale = 255.99/(max - min);
  npix = (xmax - xmin + 1)*(ymax - ymin + 1);
  pgm = (unsigned char *) calloc( npix, sizeof( unsigned char));
  k = 0;
  if (clockwiserot) {
      nc = (ymax - ymin + 1);
      nr = (xmax - xmin + 1);
      if (yflip) {
          j1 = xmax;
          j_incr = -1;
      } else {
          j1 = xmin;
          j_incr = 1;
      }
      if (xflip) {
          i1 = ymax;
          i_incr = -1;
      } else {
          i1 = ymin;
          i_incr = 1;
      }
      for (j=j1, n=0; n<nr; j+=j_incr, n++)
        for (i=i1, m=0; m<nc; i+=i_incr, m++) {
          scaledpix = MAX( 0, (int)( scale*(im[j][i] - min)));
          scaledpix = MIN( 255, scaledpix);
          pgm[k++] = (unsigned char) scaledpix;
        }
  } else {
      nc = (xmax - xmin + 1);
      nr = (ymax - ymin + 1);
      if (yflip) {
          j1 = ymin;
          j_incr = 1;
      } else {
          j1 = ymax;
          j_incr = -1;
      }
      if (xflip) {
          i1 = xmax;
          i_incr = -1;
      } else {
          i1 = xmin;
          i_incr = 1;
      }
      for (j=j1, n=0; n<nr; j+=j_incr, n++)
        for (i=i1, m=0; m<nc; i+=i_incr, m++) {
          scaledpix = MAX( 0, (int)( scale*(im[i][j] - min)));
          scaledpix = MIN( 255, scaledpix);
          pgm[k++] = (unsigned char) scaledpix;
        }
  }
  pgmwrite( name, pgm, nc, nr, 255);
  free((unsigned char *) pgm);
}
