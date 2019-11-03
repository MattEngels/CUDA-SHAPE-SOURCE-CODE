/* Modified 2011 August 12 by CM: increased MAXSTP to 1000000; set yscal[i] = 1.0 for all i
       -- that is, treat eps as an absolute tolerance rather than fractional (see discussion
       in Numerical Recipes in C) on the assumption that all variables y are of order unity  */
/* Modified 2011 August 10 by CM: increased MAXSTP to 100000  */
extern "C" {

#include "../shape/head.h"
}

#define MAXSTP 1000000
#define TINY 1.0e-30
#define nvar 12 //new

__device__ void dev_odeint( double ystart[13], double x1, double x2, double eps,
			double h1, double hmin, int *nok, int *nbad,
			void (*dev_derivs)(double,double *,double *),
			void (*drkqc)(double *,double *,double *,double,double,double *,double *,
					double *,void (*)(double,double *,double *)))
{ 	
	int nstp,i;
	double x,hnext,hdid,h;
	double yscal[13], y[13], dydx[13];

	x = x1;
	h = (x2 > x1) ? fabs(h1) : -fabs(h1);
	*nok = (*nbad) = 0;
	yscal[0] = y[0] = dydx[0] = 0.0;

	for (i=1; i<=12; i++)
		y[i] = ystart[i];

	for (nstp=1; nstp<=MAXSTP; nstp++) {

		(*dev_derivs)(x,y,dydx);

		for (i=1; i<=12; i++)
			yscal[i] = 1.0;

		if ((x+h-x2)*(x+h-x1) > 0.0)
			h = x2-x;

		(*drkqc)(y, dydx, &x, h, eps, yscal, &hdid, &hnext, dev_derivs);

		if (hdid == h)	++(*nok);
		else			++(*nbad);

		if ((x-x2)*(x2-x1) >= 0.0) {

			for (i=1; i<=12; i++)
				ystart[i] = y[i];

			return;
		}
		if (fabs(hnext) <= hmin) printf("Step size too small in dev_odeint\n");
		h = hnext;
	}
	printf("Too many steps in routine dev_odeint\n");
}

#undef MAXSTP
#undef TINY
#undef nvar
