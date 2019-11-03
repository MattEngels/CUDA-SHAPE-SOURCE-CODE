/*  Modified from brent.c by CM on 2004 August 13, so as to accept
    an absolute tolerance argument in addition to the fractional
    tolerance argument                                              */
extern "C" {
#include "../shape/head.h"
}
#define ITMAX 100
#define CGOLD 0.3819660
#define ZEPS 1.0e-10
#define SIGN(a,b) ((b) > 0.0 ? fabs(a) : -fabs(a))
#define SHFT(a,b,c,d) (a)=(b);(b)=(c);(c)=(d);

__host__ double brent_abs_gpu(double ax,double bx,double cx,double (*f)(double,
		struct vertices_t**, unsigned char*, unsigned char*, int*, int*, int*, int,
		int, cudaStream_t*, double**), double tol,double abstol,double *xmin, struct
		vertices_t **verts, unsigned char *htype, unsigned char *dtype,
		int *nframes, int *nviews, int *lc_n, int nsets, int nf, cudaStream_t *bf_stream,
		double **fit_overflow)
{
	int iter;
	double a, b, d, etemp, fu, fv, fw, fx, p, q, r, tol1, tol2, u, v, w, x, xm;
	double e = 0.0;
	double abstol_use = ((abstol > ZEPS) ? abstol : ZEPS);

	a = ((ax < cx) ? ax : cx);
	b = ((ax > cx) ? ax : cx);
	x = w = v = bx;
	fw = fv = fx = (*f)(x, verts, htype, dtype, nframes, nviews, lc_n, nsets, nf, bf_stream, fit_overflow);

	for (iter=1; iter<=ITMAX; iter++) {
		xm = 0.5 * (a + b);
		tol2 = 2.0 * (tol1 = tol * fabs(x) + abstol_use);

	if (fabs(x - xm) <= (tol2 - 0.5 * (b - a))) {
			*xmin = x;
			return fx;
		}

		if (fabs(e) > tol1) {
			r = (x-w) * (fx-fv);
			q = (x-v) * (fx-fw);
			p = (x-v) *q - (x-w) *r;
			q = 2.0 * (q-r);

			if (q > 0.0) 	p = -p;

			q = fabs(q);
			etemp = e;
			e = d;

			if (fabs(p) >= fabs(0.5 * q * etemp) || p <= q*(a-x) || p >= q*(b-x))
				d = CGOLD*(e=(x >= xm ? a-x : b-x));
			else {
				d = p / q;
				u = x + d;
				if (u-a < tol2 || b-u < tol2)
					d = SIGN(tol1, xm-x);
			}
		} else
			d = CGOLD * (e = (x >= xm ? a-x : b-x));

		u = (fabs(d) >= tol1 ? x+d : x+SIGN(tol1,d));
		fu = (*f)(u, verts, htype, dtype, nframes, nviews, lc_n, nsets, nf, bf_stream, fit_overflow);

		if (fu <= fx) {
			if (u >= x) {
				a = x;
			}
			else b = x;

			SHFT(v, w, x, u)
			SHFT(fv, fw, fx, fu)
		} else {

			if (u < x) 	a = u;
			else 		b = u;

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


#undef ITMAX
#undef CGOLD
#undef ZEPS
#undef SIGN
