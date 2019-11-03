extern "C" {
#include "../shape/head.h"
}

__device__ void dev_cotrans1(double3 *y, double a[3][3], double3 x, int dir) {
	/* This version replaces double y[3] and double x[3] with double3 y and double3 x */
	double t[3];
	int i;

	if (dir == 1)
		for (i = 0; i <= 2; i++) {
			t[i] = 0.0;
			t[i] += a[i][0] * x.x;
			t[i] += a[i][1] * x.y;
			t[i] += a[i][2] * x.z;
		}

	if (dir == (-1))
		for (i = 0; i <= 2; i++) {
			t[i] = 0.0;
			t[i] += a[0][i] * x.x;
			t[i] += a[1][i] * x.y;
			t[i] += a[2][i] * x.z;
		}

	y->x = t[0];
	y->y = t[1];
	y->z = t[2];
}
__device__ void dev_cotrans2(double y[3], double a[3][3], double x[3], int dir)
{
	double t[3];
	int i, j;

	if (dir==1)
		for (i=0;i<=2;i++) {
			t[i] = 0.0;
			for (j=0;j<=2;j++)
				t[i] += a[i][j]*x[j];
		}
	if (dir==(-1))
		for (i=0;i<=2;i++) {
			t[i] = 0.0;
			for (j=0;j<=2;j++)
				t[i] += a[j][i]*x[j];
		}
	for (i=0;i<=2;i++)
		y[i] = t[i];
}
__device__ void dev_cotrans3(double3 *y, double3 *a, double3 x, int dir, int frm)
{
	double3 t;
	int f=3*frm;

	if (dir==1) {
		t.x = 0.0;
		t.x += a[f+0].x * x.x;
		t.x += a[f+0].y * x.y;
		t.x += a[f+0].z * x.z;
		t.y = 0.0;
		t.y += a[f+1].x * x.x;
		t.y += a[f+1].y * x.y;
		t.y += a[f+1].z * x.z;
		t.z = 0.0;
		t.z += a[f+2].x * x.x;
		t.z += a[f+2].y * x.y;
		t.z += a[f+2].z * x.z;
	}
	if (dir==(-1)) {
		t.x = 0.0;
		t.x += a[f+0].x * x.x;
		t.x += a[f+1].x * x.y;
		t.x += a[f+2].x * x.z;
		t.y = 0.0;
		t.y += a[f+0].y * x.x;
		t.y += a[f+1].y * x.y;
		t.y += a[f+2].y * x.z;
		t.z = 0.0;
		t.z += a[f+0].z * x.x;
		t.z += a[f+1].z * x.y;
		t.z += a[f+2].z * x.z;
	}

	y->x = t.x;
	y->y = t.y;
	y->z = t.z;
}
__device__ void dev_cotrans5(double3 *y, double a[3][3], double x[3], int dir, int f) {
	/* This version replaces double y[3] and double x[3] with double3 y and double3 x */
	double t[3];
	int i, j;

	if (dir==1)
			for (i=0;i<=2;i++) {
				t[i] = 0.0;
				for (j=0;j<=2;j++)
					t[i] += a[i][j]*x[j];
			}
		if (dir==(-1))
			for (i=0;i<=2;i++) {
				t[i] = 0.0;
				for (j=0;j<=2;j++)
					t[i] += a[j][i]*x[j];
			}
		y[f].x = t[0];
		y[f].y = t[1];
		y[f].z = t[2];
}
