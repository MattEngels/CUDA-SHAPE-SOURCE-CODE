extern "C" {
#include "../shape/head.h"
}
#define SMALLCOEFF3 1.0e-5
/*  Find all real roots of a cubic equation, using methods given in section 5.6 of
    Numerical Recipes in C.  Element 3 of the input coeff vector is the cubic
    coefficient while element 0 is the constant term.  Up to three real roots are
    stored in the output realroot vector, with any unused elements set to a large
    negative dummy value.  The return value is the number of real roots found.
    The routine includes several tests for coefficients that are equal to zero;
    those tests assume that nonzero coefficients are of order unity.                */
__device__ int cubic_realroots_cuda( double *coeff, double *realroot)
{
	int nrealroots, bsign;
	double a, b, c, discriminant, q, qsqrt, r, r2minusq3, rsign, s, t, theta;
	nrealroots = 0;
	realroot[0] = realroot[1] = realroot[2] = -HUGENUMBER;

	if (fabs(coeff[3]) < SMALLCOEFF3) {
		/*  cubic term is zero  */
		a = coeff[2];
		b = coeff[1];
		c = coeff[0];

		if (fabs(a) < SMALLVAL) {

			if (fabs(b) < SMALLVAL) {
				/*  Error: the cubic, quadratic, and linear terms are zero  */
				if (fabs(c) < SMALLVAL)
					printf("cubic_realroots in realize_mod.c: all four coefficients are zero\n");
				else
					printf("cubic_realroots in realize_mod.c: only the constant term is nonzero\n");

			} else {
				/*  linear equation  */
				realroot[0] = -c/b;
				nrealroots = 1;
			}

		} else {
			/*  quadratic equation  */
			discriminant = b*b - 4*a*c;
			if (discriminant < 0.0)
				printf("cubic_realroots in realize_mod.c: quadratic equation has no real roots\n");
			if (fabs(b) < SMALLVAL) {
				realroot[0] = sqrt(discriminant)/(2*a);
				realroot[1] = -realroot[0];
			} else {
				bsign = (b < 0.0) ? -1 : 1;
				q = -0.5*(b + bsign*sqrt(discriminant));
				realroot[0] = q/a;
				realroot[1] = c/q;
			}
			nrealroots = 2;
		}
	} else {
		/*  cubic term is nonzero: scale to standard form x^3 + ax^2 + b^x + c = 0  */
		a = coeff[2]/coeff[3];
		b = coeff[1]/coeff[3];
		c = coeff[0]/coeff[3];

		/* Check if there is one real root or three. Write out test quantity
		 * r^2 - q^3 explicitly in terms of coefficients a, b, and c in order
		 * to cancel high-order terms and thus reduce the likelihood of
		 * roundoff problems           */

		q = (a*a - 3*b)/9;
		r = (2*a*a*a - 9*a*b + 27*c)/54;

		r2minusq3 = (4*a*a*a*c - a*a*b*b - 18*a*b*c + 27*c*c + 4*b*b*b)/108;
		if (r2minusq3 >= 0.0) {
			/*  one real root  */
			rsign = (r < 0.0) ? -1 : 1;
			s = -rsign*pow( fabs(r) + sqrt(r2minusq3), 1.0/3);
			t = (fabs(s) >= SMALLVAL) ? q/s : 0.0;
			realroot[0] = s + t - a/3;
			nrealroots = 1;
		} else {
			/*  three real roots  */
			qsqrt = sqrt(q);
			theta = acos(r/(q*qsqrt));
			realroot[0] = -2*qsqrt*cos(theta/3) - a/3;
			realroot[1] = -2*qsqrt*cos((theta + 2*PIE)/3) - a/3;
			realroot[2] = -2*qsqrt*cos((theta - 2*PIE)/3) - a/3;
			nrealroots = 3;
		}
	}
	return nrealroots;
}

#undef SMALLCOEFF3
