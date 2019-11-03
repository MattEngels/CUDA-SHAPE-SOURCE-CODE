/* Modified 2011 August 12 by CM: increased MAXSTP to 1000000; set yscal[i] = 1.0 for all i
       -- that is, treat eps as an absolute tolerance rather than fractional (see discussion
       in Numerical Recipes in C) on the assumption that all variables y are of order unity  */
/* Modified 2011 August 10 by CM: increased MAXSTP to 100000  */



#include "basic.h"

#define MAXSTP 1000000
#define TINY 1.0e-30

//int /*kmax=0,*/kount=0;  /* defining declaration */
//double *xp=0,**yp=0,dxsav=0;  /* defining declaration */

void odeint( double *ystart, int nvar, double x1, double x2, double eps,
			double h1, double hmin, int *nok, int *nbad,
			void (*derivs)(double,double *,double *),
			void (*rkqc)(double *,double *,double *,double,
			double,double *,double *,double *,void (*)())
			)
{ 	
  int nstp, i;
  double xsav, x, hnext, hdid, h;
  double *yscal, *y, *dydx;

  yscal = vector(1, nvar);
  y = vector(1, nvar);
  dydx = vector(1, nvar);
  x = x1;
  h = (x2 > x1) ? fabs(h1) : -fabs(h1);
  *nok = (*nbad) = 0;

  for (i=1; i<=nvar; i++)
	  y[i]=ystart[i];		/* Initialize y with ystart */

  for (nstp=1; nstp<=MAXSTP; nstp++) { /* Do MAXSTP loops at most */

	  (*derivs)(x, y, dydx);		/* get derivative dydx */

	  for (i=1; i<=nvar; i++)
		  yscal[i]=1.0;			/* Initialize scaling vector to proportional */

	  if ((x+h-x2)*(x+h-x1) > 0.0)
		  h = x2 - x;

	  /* Call bsstep - or whatever function was submitted as rkqc - to
	   * calculate the linked differential equations	   */
	  (*rkqc)(y,dydx,&x,h,eps,yscal,&hdid,&hnext,derivs);

	  /* Increase counters for ok and bad, as appropriate */
	  if (hdid == h)	++(*nok);
	  else 	  			++(*nbad);

	  if ((x-x2)*(x2-x1) >= 0.0) {

		  for (i=1; i<=nvar; i++)
			  ystart[i]=y[i];

		  /* We're done and hav a solution, free up memory */
		  free_vector(dydx, 1, nvar);
		  free_vector(y, 1, nvar);
		  free_vector(yscal, 1, nvar);
		  return;
	  }
	  if (fabs(hnext) <= hmin)
		  nrerror("Step size too small in ODEINT");
	  h=hnext;
  }
  nrerror("Too many steps in routine ODEINT");
}

/*void odeint( double *ystart, int nvar, double x1, double x2, double eps,
			double h1, double hmin, int *nok, int *nbad,
			void (*derivs)(double,double *,double *),
			void (*rkqc)(double *,double *,int,double *,double,
			double,double *,double *,double *,void (*)())
			)
{
  int nstp, i;
  double xsav, x, hnext, hdid, h;
  double *yscal, *y, *dydx;

  yscal = vector(1, nvar);
  y = vector(1, nvar);
  dydx = vector(1, nvar);
  x = x1;
  h = (x2 > x1) ? fabs(h1) : -fabs(h1);
  *nok = (*nbad) = kount = 0;

  for (i=1; i<=nvar; i++)
	  y[i]=ystart[i];		 Initialize y with ystart

  if (kmax > 0)				 Storage for the first step
	  xsav = x - dxsav * 2.0;

  for (nstp=1; nstp<=MAXSTP; nstp++) {  Do MAXSTP loops at most

	  (*derivs)(x, y, dydx);		 get derivative dydx

	  for (i=1; i<=nvar; i++)
		   yscal[i]=fabs(y[i])+fabs(dydx[i]*h)+TINY;
		  yscal[i]=1.0;			 Initialize scaling vector to proportional

	  if (kmax > 0) {
		  if (fabs(x-xsav) > fabs(dxsav)) {
			  if (kount < kmax-1) {
				  xp[++kount]=x;
				  for (i=1;i<=nvar;i++)
					  yp[i][kount]=y[i];
				  xsav=x;
			  }
		  }
	  }

	  if ((x+h-x2)*(x+h-x1) > 0.0)
		  h = x2 - x;

	   Call bsstep - or whatever function was submitted as rkqc - to
	   * calculate the linked differential equations
	  (*rkqc)(y,dydx,nvar,&x,h,eps,yscal,&hdid,&hnext,derivs);

	   Increase counters for ok and bad, as appropriate
	  if (hdid == h)	++(*nok);
	  else 	  			++(*nbad);

	  if ((x-x2)*(x2-x1) >= 0.0) {

		  for (i=1; i<=nvar; i++)
			  ystart[i]=y[i];
		  if (kmax) {
			  xp[++kount]=x;
			  for (i=1; i<=nvar; i++)
				  yp[i][kount]=y[i];
		  }
		   We're done and hav a solution, free up memory
		  free_vector(dydx, 1, nvar);
		  free_vector(y, 1, nvar);
		  free_vector(yscal, 1, nvar);
		  return;
	  }
	  if (fabs(hnext) <= hmin)
		  nrerror("Step size too small in ODEINT");
	  h=hnext;
  }
  nrerror("Too many steps in routine ODEINT");
}*/

#undef MAXSTP
#undef TINY
