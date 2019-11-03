#include "basic.h"
#include "../macros/files.h"


void pgmwrite( char *name, unsigned char *buf, int nc, int nr, int nlev)
{
  FILE *fp;

  FOPEN( fp, name, "w");
  fprintf( fp, "P5\n");
  fprintf( fp, "# created by pgmwrite()\n");
  fprintf( fp, "%d %d\n", nc, nr);
  fprintf( fp, "%d\n", nlev);
  fwrite( &buf[0], sizeof(unsigned char), nr*nc, fp);
  fclose(fp);
}
