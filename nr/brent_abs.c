/*  Modified from brent.c by CM on 2004 August 13, so as to accept
    an absolute tolerance argument in addition to the fractional
    tolerance argument                                              */

#include "basic.h"

#define ITMAX 100
#define CGOLD 0.3819660
#define ZEPS 1.0e-10
#define SIGN(a,b) ((b) > 0.0 ? fabs(a) : -fabs(a))
#define SHFT(a,b,c,d) (a)=(b);(b)=(c);(c)=(d);

double brent_abs(double ax,double bx,double cx,double (*f)(double),
	         double tol,double abstol,double *xmin)
{
	int iter;
	double a, b, d, etemp, fu, fv, fw, fx, p, q, r, tol1, tol2, u, v, w, x, xm;
	double e = 0.0;
    double abstol_use = ((abstol > ZEPS) ? abstol : ZEPS);

	a = ((ax < cx) ? ax : cx);
	b = ((ax > cx) ? ax : cx);
	x = w = v = bx;
	fw = fv = fx = (*f)(x);

	for (iter=1; iter<=ITMAX; iter++) {

		xm = 0.5 * (a+b);
		tol2 = 2.0 * (tol1 = tol * fabs(x) + abstol_use);

		if (fabs(x-xm) <= (tol2 - 0.5 * (b-a))) {

			*xmin = x;
			return fx;
		}

		if (fabs(e) > tol1) {

			r = (x-w) * (fx-fv);
			q = (x-v) * (fx-fw);
			p = (x-v) * q - (x-w) * r;
			q = 2.0 * (q-r);

			if (q > 0.0)	p = -p;
			q = fabs(q);
			etemp = e;
			e = d;

			if (fabs(p) >= fabs(0.5*q*etemp) || p <= q*(a-x) || p >= q*(b-x))
				d = CGOLD * (e = (x >= xm ? a-x : b-x));

			else {
				d = p/q;
				u = x+d;

				if (u-a < tol2 || b-u < tol2)
					d = SIGN(tol1, xm-x);
			}
		} else
			d = CGOLD * (e = (x >= xm ? a-x : b-x));

		u = (fabs(d) >= tol1 ? x+d : x+SIGN(tol1,d));
		fu = (*f)(u);

		if (fu <= fx) {
			if (u >= x)		a = x;
			else			b = x;

			SHFT(v, w, x, u)
			SHFT(fv, fw, fx, fu)
		} else {
			if (u < x)		a = u;
			else 			b = u;

			if (fu <= fw || w == x) {
				v = w;
				w = u;
				fv = fw;
				fw = fu;
			} else if (fu <= fv || v == x || v == w) {

				v = u;
				fv = fu;
			}
		}
	}
	nrerror("Too many iterations in BRENT");
	*xmin=x;
	return fx;
}

//double brent_abs(double ax,double bx,double cx,double (*f)(double),
//	         double tol,double abstol,double *xmin)
//{
//	int iter;
//	double a, b, d, etemp, fu, fv, fw, fx, p, q, r, tol1, tol2, u, v, w, x, xm;
//	double e = 0.0;
//    double abstol_use = ((abstol > ZEPS) ? abstol : ZEPS);
//
////    int dbg=0;
////    printf("brent_abs start\n");
//
//	a = ((ax < cx) ? ax : cx);
//	b = ((ax > cx) ? ax : cx);
//	x = w = v = bx;
//	fw = fv = fx = (*f)(x);
//
////	printf("a, %3.8g\n", a);
////	printf("b, %3.8g\n", b);
////	printf("x=w=v=bx, %3.8g\n", x);
////	printf("fw=fv=fx=(*f)(x), %3.8g\n", fw);
//
//	for (iter=1; iter<=ITMAX; iter++) {
////		dbg++;
////		printf("start of loop #, %i\n", dbg);
//
//		xm = 0.5 * (a+b);
//		tol2 = 2.0 * (tol1 = tol * fabs(x) + abstol_use);
//
////		printf("xm, %3.8g\n", xm);
////		printf("tol2, %3.8g\n", tol2);
////		printf("fabs(x-xm), %3.8g\n", fabs(x-xm));
////		printf("(tol2-0.5*(b-a), %3.8g\n", (tol2-0.5*(b-a)));
//
//		if (fabs(x-xm) <= (tol2 - 0.5 * (b-a))) {
//
////			printf("if (fabs(x-xm) <= (tol2-0.5*(b-a))), 1\n");
//
//			*xmin = x;
////			printf("loops in brent_abs, %i\n", dbg);
//			return fx;
//		}
//
////		printf("fabs(e), %3.8g\n", fabs(e));
////		printf("tol1, %3.8g\n", tol1);
//
//		if (fabs(e) > tol1) {
//
////			printf("if (fabs(e) > tol1), 1\n");
////			printf("x, %3.8g\n", x);
////			printf("v, %3.8g\n", v);
////			printf("w, %3.8g\n", w);
////			printf("fx, %3.8g\n", fx);
////			printf("fv, %3.8g\n", fv);
////			printf("fw, %3.8g\n", fw);
//
//			r = (x-w) * (fx-fv);
//			q = (x-v) * (fx-fw);
//			p = (x-v) * q - (x-w) * r;
//
////			printf("r, %3.8g\n", r);
////			printf("q, %3.8g\n", q);
////			printf("p, %3.8g\n", p);
//
//			q = 2.0 * (q-r);
//
////			printf("q, %3.8g\n", q);
//
//			if (q > 0.0) {
//				p = -p;
//
////				printf("if (q > 0.0), 1\n");
//			}
//			q = fabs(q);
//			etemp = e;
//			e = d;
//
////			printf("q, %3.8g\n", q);
////			printf("e, %3.8g\n", e);
//
//			if (fabs(p) >= fabs(0.5*q*etemp) || p <= q*(a-x) || p >= q*(b-x)) {
//				d = CGOLD * (e = (x >= xm ? a-x : b-x));
//
////				printf("if (fabs(p) >= fabs(0.5 * q * etemp) || p <= q*(a-x) || p >= q*(b-x)), 1\n");
////				printf("d, %3.8g\n", d);
//			}
//			else {
////				printf("if (fabs(p) >= fabs(0.5 * q * etemp) || p <= q*(a-x) || p >= q*(b-x)), 0\n");
//
//				d = p/q;
//				u = x+d;
//
////				printf("d, %3.8g\n", d);
////				printf("u, %3.8g\n", u);
//
//				if (u-a < tol2 || b-u < tol2) {
//					d = SIGN(tol1, xm-x);
//
////					printf("if (u-a < tol2 || b-u < tol2), 1\n");
//				} //else printf("if (u-a < tol2 || b-u < tol2), 0\n");
//			}
//		} else {
//			d = CGOLD * (e = (x >= xm ? a-x : b-x));
//
////			printf("if (fabs(e) > tol1), 1\n");
////			printf("d, %3.8g\n", d);
//
//		}
//
//		u = (fabs(d) >= tol1 ? x+d : x+SIGN(tol1,d));
//		fu = (*f)(u);
//
////		printf("u, %3.8g\n", u);
////		printf("fu, %3.8g\n", fu);
//
//		if (fu <= fx) {
////			printf("if (fu <= fx), 1\n");
//
//			if (u >= x) {
//				a = x;
//
////				printf("if (u >= x), 1\n");
////				printf("a, %3.8g\n", a);
//			}
//			else {
//				b = x;
//
////				printf("if (u >= x), 0\n");
////				printf("b, %3.8g\n", b);
//			}
//			SHFT(v, w, x, u)
//			SHFT(fv, fw, fx, fu)
//		} else {
////			printf("if (fu <= fx), 0\n");
//
//			if (u < x) {
//				a = u;
//
////				printf("if (u < x), 1\n");
////				printf("a, %3.8g\n", a);
//			}
//			else {
//				b = u;
//
////				printf("if (u < x), 0\n");
////				printf("b, %3.8g\n", b);
//			}
//			if (fu <= fw || w == x) {
////				printf ("if (fu <= fw || w == x), 1\n");
//
//				v = w;
//				w = u;
//				fv = fw;
//				fw = fu;
//
////				printf("v, %3.8g\n", v);
////				printf("w, %3.8g\n", w);
////				printf("fv, %3.8g\n", fv);
////				printf("fw, %3.8g\n", fw);
//
//			} else if (fu <= fv || v == x || v == w) {
////				printf("if (fu <= fw || w == x), 0\n");
////				printf("else if (fu <= fv || v == x || v == w), 1\n");
//
//				v = u;
//				fv = fu;
//
////				printf("v, %3.8g\n", v);
////				printf("fv, %3.8g\n", fv);
//			}
//		}
//	}
//	nrerror("Too many iterations in BRENT");
//	*xmin=x;
////	printf("loops in brent_abs, %i\n", dbg);
//	return fx;
//}

#undef ITMAX
#undef CGOLD
#undef ZEPS
#undef SIGN
