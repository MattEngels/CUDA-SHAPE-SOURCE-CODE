extern "C" {
#include "../shape/head.h"
}

__device__ void dev_mmmul( double x[3][3], double y[3][3], double z[3][3])
{
  double t[3][3];
  int i, j, k;

  for (i=0;i<=2;i++)
	for (j=0;j<=2;j++) {
	  t[i][j] = 0.0;
	  for (k=0;k<=2;k++)
		t[i][j] += y[i][k]*z[k][j];
	}
  for (i=0;i<=2;i++)
	for (j=0;j<=2;j++)
	  x[i][j] = t[i][j];
}
__device__ void dev_mmmul2(double3 *x, double y[3][3], double3 *z, int frm)
{ /* This version turns the original double x[3][3] and double z[3][3] into
   * double3 pointers with nframes entries.  Selection is made via f  */
	double t[3][3];
	int i, f;
	f = 3*frm;

	for (i=0; i<=2; i++) {

		t[i][0] = 0.0;
		t[i][0] += y[i][0] * z[f+0].x;
		t[i][0] += y[i][1] * z[f+1].x;
		t[i][0] += y[i][2] * z[f+2].x;

		t[i][1] = 0.0;
		t[i][1] += y[i][0] * z[f+0].y;
		t[i][1] += y[i][1] * z[f+1].y;
		t[i][1] += y[i][2] * z[f+2].y;

		t[i][2] = 0.0;
		t[i][2] += y[i][0] * z[f+0].z;
		t[i][2] += y[i][1] * z[f+1].z;
		t[i][2] += y[i][2] * z[f+2].z;
	}

	for (i=0; i<=2; i++) {
		x[f+i].x = t[i][0];
		x[f+i].y = t[i][1];
		x[f+i].z = t[i][2];
	}
}
