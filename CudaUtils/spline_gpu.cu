extern "C" {
#include "../shape/head.h"
}

__global__ void lghtcrv_spline_krnl(struct dat_t *ddat, int set, double
		yp1, double ypn, double *u, int ncalc) {
	/*(double *x  - lghtcrv->x
	 * double *y  - lghtcrv->y
	 * int n      - calc
	 * double yp1 - 2.0e30
	 * double ypn - 2.0e30
	 * double *y2 - lghtcrv->y2)*/

	int i, k, n=ncalc;
	double p, qn, sig, un;
	double *x = ddat->set[set].desc.lghtcrv.x;
	double *y = ddat->set[set].desc.lghtcrv.y;
	double *y2 = ddat->set[set].desc.lghtcrv.y2;

	/* single-threaded kernel */
	if (threadIdx.x == 0) {
		u[0] = 0.0;
		if (yp1 > 0.99e30)
			y2[1] = u[1]=0.0;
		else {
			y2[1] = -0.5;
			u[1] = (3.0 / (x[2]-x[1])) * ((y[2]-y[1]) / (x[2]-x[1]) - yp1);
		}

		for (i=2; i<=n-1; i++) {
			sig = (x[i]-x[i-1]) / (x[i+1]-x[i-1]);
			p = sig * y2[i-1] + 2.0;
			y2[i] = (sig - 1.0) / p;
			u[i] = (y[i+1] - y[i]) / (x[i+1] - x[i]) - (y[i] - y[i-1]) / (x[i] - x[i-1]);

			u[i] = (6.0 * u[i] / (x[i+1]-x[i-1]) - sig*u[i-1] ) / p;
		}

		if (ypn > 0.99e30)
			qn = un = 0.0;
		else {
			qn = 0.5;
			un = (3.0 / (x[n] - x[n-1])) * (ypn - (y[n] - y[n-1]) / (x[n] - x[n-1]));
		}
		y2[n] = (un - qn * u[n-1]) / (qn * y2[n-1] + 1.0);

		for (k=n-1; k>=1; k--)
			y2[k] = y2[k] * y2[k+1] + u[k];

	}
}
