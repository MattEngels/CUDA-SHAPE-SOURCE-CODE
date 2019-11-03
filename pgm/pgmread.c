#include "basic.h"
#include <string.h>
#include "../macros/files.h"


void pgmread( char *name, unsigned char **buf, int *nc, int *nr, int *nlev)
{
  char tmp[80];
  FILE *fp;

  FOPEN( fp, name, "r");
  /* expect to see P5 at beginning */
  fscanf( fp, "%s", tmp);
  if (strcmp( tmp, "P5")) {
	printf("expected to read P5, got %s\n", tmp);
	return;
  }
  NEXTLINE(fp); 

  /* skip comments */
  fscanf( fp, "%s", tmp);
  while (!strcmp( tmp, "#")) {
	NEXTLINE(fp);
	fscanf( fp, "%s", tmp);
  }
  *nc = atoi( tmp);
  fscanf( fp, " %d %d", nr, nlev);
  NEXTLINE(fp);

  *buf = (unsigned char *) calloc( (*nr)*(*nc), sizeof(unsigned char));
  fread( &(*buf)[0], sizeof(unsigned char), (*nr)*(*nc), fp);
  fclose(fp);
}
