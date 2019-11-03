extern "C" {
#include "../shape/head.h"
}

__device__ void dev_mmid( double *y, double *dydx, double xs, double htot,
		  int nstep, double *yout, 
		  void (*dev_derivs)( double, double *, double *))
{
	int n,i;
	double x, swap, h2, h, ym[13], yn[13];
	ym[0] = yn[0] = 0.0;
	h= htot / nstep;

	for (i=1; i<=12; i++) {
		ym[i] = y[i];
		yn[i] = y[i] + h * dydx[i];
	}

	x = xs + h;
	(*dev_derivs)(x, yn, yout);
	h2 = 2.0 * h;

	for (n=2; n<=nstep; n++) {
		for (i=1; i<=12; i++) {
			swap = ym[i] + h2 * yout[i];
			ym[i] = yn[i];
			yn[i] = swap;
		}
		x += h;
		(*dev_derivs)(x, yn, yout);
	}
	for (i=1; i<=12; i++)
		yout[i] = 0.5 * (ym[i] + yn[i] + h * yout[i]);
}


#undef nvar
