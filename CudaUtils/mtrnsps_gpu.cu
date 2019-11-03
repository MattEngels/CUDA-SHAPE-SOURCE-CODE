extern "C" {
#include "../shape/head.h"
}

__device__ void dev_mtrnsps( double a[3][3], double b[3][3])
{
	double t[3][3];
	int i, j;

  for (i=0;i<=2;i++)
	for (j=0;j<=2;j++)
	  t[i][j] = b[j][i];
  for (i=0;i<=2;i++)
	for (j=0;j<=2;j++)
	  a[i][j] = t[i][j];
}
__device__ void dev_mtrnsps2(double3 *a, double b[3][3], int frm)
{	/* This version splits the double a[3][3] of the original function into
	 * three separate double3 vector variables. b[3][3] remains unchanged. */
  double t[3][3];
  int i, j, f;
  f = frm *3;

  for (i=0;i<=2;i++)
	  for (j=0;j<=2;j++)
		  t[i][j] = b[j][i];

  a[f+0].x = t[0][0];
  a[f+0].y = t[0][1];
  a[f+0].z = t[0][2];
  a[f+1].x = t[1][0];
  a[f+1].y = t[1][1];
  a[f+1].z = t[1][2];
  a[f+2].x = t[2][0];
  a[f+2].y = t[2][1];
  a[f+2].z = t[2][2];
}
