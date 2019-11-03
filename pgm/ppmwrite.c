#include "basic.h"
#include "../macros/files.h"

void ppmwrite( char *name, unsigned char *buf, int nc, int nr, int nlev)
{
  FILE *fp;

  FOPEN( fp, name, "w");
  fprintf( fp, "P6\n");
  fprintf( fp, "# created by ppmwrite()\n");
  fprintf( fp, "%d %d\n", nc, nr);
  fprintf( fp, "%d\n", nlev);
  fwrite( &buf[0], sizeof(unsigned char), 3*nr*nc, fp);
  fclose(fp);
}
