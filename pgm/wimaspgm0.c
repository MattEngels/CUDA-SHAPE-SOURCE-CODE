/***************************************************************************
                                                                wimaspgm0.c

Dumps a double precision image im[][] to disk as a normalized pgm image.
Truncates everything below 0.

Modified 2003 July 30 by CM:
    Add three parameters (values 0 or 1) to enable rotation/flipping
    of output image.  If both rotation and flipping are desired,
    rotation is performed first, so that xflip and yflip refer to
    the x and y dimensions of the *rotated* image.
***************************************************************************/

#include "basic.h"
#include "../pgm/pgm.h"

void wimaspgm0( double **im, int xmin, int xmax, int ymin, int ymax, 
                int clockwiserot, int xflip, int yflip,
                char *name) 
{
  int i, j, k, npix, nc, nr, i1, i_incr, j1, j_incr, m, n;
  unsigned char *pgm;
  double max, min, scale;

  max = -1.0e20;
  min = 1.0e20;
  for (i=xmin; i<=xmax; i++)
    for (j=ymin; j<=ymax; j++) {
      if (im[i][j] < min)
        min = im[i][j];
      if (im[i][j] > max)
        max = im[i][j];
    }
  if (min < 0.0)
    min = 0.0;
  if (max < 0.0)
    max = 0.0;
  if (max == min)
    scale = 0.0;
  else
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
          if (im[j][i] > 0.0)
            pgm[k++] = (int)( scale*(im[j][i] - min));
          else
            pgm[k++] = 0;
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
          if (im[i][j] > 0.0)
            pgm[k++] = (int)( scale*(im[i][j] - min));
          else
            pgm[k++] = 0;
        }
  }
  pgmwrite( name, pgm, nc, nr, 255);
  free((unsigned char *) pgm);
}
