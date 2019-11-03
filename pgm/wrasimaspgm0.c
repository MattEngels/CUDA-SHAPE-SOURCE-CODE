/***************************************************************************

								wimaspgm0.c

Dumps a double precision image im[][] to disk as a normalized pgm image.
Truncates everything below 0.  Assumes that im[row][col] is a raster.

***************************************************************************/

#include "basic.h"
#include "pgm.h"

void wrasimaspgm0( double **im, int cmin, int cmax, int rmin, int rmax, 
			  char *name) 
{
  int k, npix, nc, nr, col, row;
  unsigned char *pgm;
  double max, min, scale;

  max = -1.0e20;
  min = 1.0e20;
  for (row=rmin;row<=rmax;row++)
	for (col=cmin;col<=cmax;col++) {
	  if (im[row][col]<min)
		min = im[row][col];
	  if (im[row][col]>max)
		max = im[row][col];
	}
  if (min<0.0)
	min = 0.0;
  if (max<0.0)
	max = 0.0;
  if (max==min)
	scale = 0.0;
  else
	scale = 255.99/(max-min);
  nc = (cmax-cmin+1);
  nr = (rmax-rmin+1);
  npix = nr*nc;
  pgm = (unsigned char *) calloc( npix, sizeof( unsigned char));
  k = 0;
  for (row=rmin;row<=rmax;row++)
	for (col=cmin;col<=cmax;col++) {
	  if (im[row][col]>0.0)
		pgm[k++] = (unsigned char)( scale*(im[row][col]-min));
	  else
		pgm[k++] = 0;
	}
  pgmwrite( name, pgm, nc, nr, 255);
  free((char *) pgm);
}
