/***************************************************************************
                                                                 wimasppm.c

Dumps a double precision image im[][][3] to disk as a normalized ppm image.

Arguments clockwiserot, xflip, and yflip (values 0 or 1) enable
rotation/flipping of output image.  If both rotation and flipping are
desired, rotation is performed first, so that xflip and yflip refer to
the x and y dimensions of the *rotated* image.

Written 2005 April 8 by CM
***************************************************************************/

#include "basic.h"
#include "../pgm/pgm.h"

void wimasppm( double ***im, int xmin, int xmax, int ymin, int ymax, 
               int clockwiserot, int xflip, int yflip,
               char *name) 
{
  int i, j, k, nbytes, nc, nr, i1, i_incr, j1, j_incr, m, n, color;
  unsigned char *ppm;
  double max, min, scale;

  max = -1.0e20;
  min = 1.0e20;
  for (i=xmin; i<=xmax; i++)
    for (j=ymin; j<=ymax; j++)
      for (color=0; color<=2; color++) {
        if (im[i][j][color] < min)
          min = im[i][j][color];
        if (im[i][j][color] > max)
          max = im[i][j][color];
      }
  if (max == min)
    scale = 0.0;
  else
    scale = 255.99 / (max - min);

  nbytes = 3*(xmax - xmin + 1)*(ymax - ymin + 1);
  ppm = (unsigned char *) calloc( nbytes, sizeof( unsigned char));
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
        for (i=i1, m=0; m<nc; i+=i_incr, m++)
          for (color=0; color<=2; color++)
            ppm[k++] = (int)( scale*(im[j][i][color] - min));
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
        for (i=i1, m=0; m<nc; i+=i_incr, m++)
          for (color=0; color<=2; color++)
            ppm[k++] = (int)( scale*(im[i][j][color] - min));
  }
  ppmwrite( name, ppm, nc, nr, 255);
  free((unsigned char *) ppm);
}
