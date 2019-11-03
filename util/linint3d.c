/***************************************************************************

								 linint3d.c

Uses linear interpolation to calculate the value of a function of x,y,z
with coordinates in [0,1] given the values at the corners of the unit cube.

****************************************************************************/

#include "basic.h"

double linint3d( double x[3], double f[2][2][2])
{
  double fz[2][2], fzy[2], fzyx;
  int i, j;

  for (i=0;i<=1;i++)
	for (j=0;j<=1;j++) {
	  fz[i][j] = f[i][j][0]*(1-x[2])+f[i][j][1]*x[2];
	}
  for (i=0;i<=1;i++)
	fzy[i] = fz[i][0]*(1-x[1])+fz[i][1]*x[1];
  fzyx = fzy[0]*(1-x[0])+fzy[1]*x[0];
/*  printf("%f %f %f %f\n", x[0], x[1], x[2], fzyx); */
  return fzyx;
}
