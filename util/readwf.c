#include "basic.h"
#include "../nr/nr.h"
#include "../macros/files.h"

double readwf( char *name, int *nv, double ***v, int *nf, int ***f)
{
  char tmp[80];
  double x, y, z, rmax=0.0, r;
  int v0, v1, v2, i;
  FILE *fp;

  (*nv) = (*nf) = 0;
  FOPEN( fp, name, "r");
  fscanf( fp, " %s", tmp);
  while (!strcmp( tmp, "v")) {
	++(*nv);
	fscanf( fp, " %lf %lf %lf", &x, &y, &z);
	if ((r = x*x+y*y+z*z)>rmax)
	  rmax = r;
	fscanf( fp, " %s", tmp);
  }
  while ((!strcmp( tmp, "f"))&&(!feof(fp))) {
	++(*nf);
	fscanf( fp, " %d %d %d", &v0, &v1, &v2);
	fscanf( fp, " %s", tmp);
  }
  fclose( fp);

  (*v) = matrix( 1, (*nv), 1, 3);
  (*f) = imatrix( 1, (*nf), 1, 3);

  FOPEN( fp, name, "r");
  for (i=1;i<=(*nv);i++)
	fscanf( fp, " %s %lf %lf %lf", tmp, &(*v)[i][1], &(*v)[i][2], 
		   &(*v)[i][3]);
  for (i=1;i<=(*nf);i++)
	fscanf( fp, " %s %d %d %d", tmp, &(*f)[i][1], &(*f)[i][2], 
		   &(*f)[i][3]);
  fclose( fp);
  return sqrt(rmax);
}
