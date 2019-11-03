/*  Modified 2004 March 27 by CM:
        now handles pgm input: assign grayscale level to R, G, and B levels  */

#include "basic.h"
#include <string.h>
#include "../macros/files.h"

void ppmread( char *name, unsigned char **ppmbuf, int *nc, int *nr, int *nlev)
{
  unsigned char *pgmbuf;
  char tmp[80];
  int is_pgm, npix, i, j;
  FILE *fp;

  /*  Open the input file and figure out whether it's pgm or ppm format:
      The first header line should read "P5" (pgm with binary data) or
      "P6" (ppm with binary data).

      Note that we don't allow "P2" or "P3" (pgm or ppm with ASCII data),
      nor do we allow pbm format ("P1" or "P4" for ASCII or binary data).  */

  FOPEN( fp, name, "r");
  fscanf( fp, "%s", tmp);
  if (!strcmp( tmp, "P5")) {
      is_pgm = 1;
  } else if (!strcmp( tmp, "P6")) {
      is_pgm = 0;
  } else {
      printf("expected to read P5 or P6, got %s\n", tmp);
      return;
  }
  NEXTLINE(fp); 

  /*  Skip comments  */

  fscanf( fp, "%s", tmp);
  while (!strcmp( tmp, "#")) {
    NEXTLINE(fp);
    fscanf( fp, "%s", tmp);
  }

  /*  Get the number of columns, rows, and brightness levels  */

  *nc = atoi( tmp);
  fscanf( fp, " %d %d", nr, nlev);
  NEXTLINE(fp);

  /*  Allocate memory for the output ppm buffer  */

  npix = (*nr)*(*nc);
  *ppmbuf = (unsigned char *) calloc( 3*npix, sizeof(unsigned char));

  /*  Read the binary data and get the RGB levels for each output pixel  */

  if (is_pgm) {

      /*  pgm input: read the data into a temporary buffer, then
          assign each pixel's grayscale level to the R, G, and B
          levels of the corresponding pixel in the output ppm buffer  */

      pgmbuf = (unsigned char *) calloc( npix, sizeof(unsigned char));
      fread( &pgmbuf[0], sizeof(unsigned char), npix, fp);
      for (i=0, j=0; i<npix; i++, j+=3)
        (*ppmbuf)[j] = (*ppmbuf)[j+1] = (*ppmbuf)[j+2] = pgmbuf[i];
      free((unsigned char *) pgmbuf);

  } else {

      /*  ppm input: just read the data into the output ppm buffer  */

      fread( &(*ppmbuf)[0], sizeof(unsigned char), 3*npix, fp);
  }
  fclose(fp);
}
