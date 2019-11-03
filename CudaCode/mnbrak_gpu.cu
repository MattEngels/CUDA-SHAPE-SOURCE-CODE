extern "C" {
#include "../shape/head.h"
}

#define GOLD 1.618034
#define GLIMIT 100.0
#define TINY 1.0e-20
#define SIGN(a,b) ((b) > 0.0 ? fabs(a) : -fabs(a))
#define SHFT(a,b,c,d) (a)=(b);(b)=(c);(c)=(d);

__host__ void mnbrak_gpu(
		double *ax,
		double *bx,
		double *cx,
		double *fa,
		double *fb,
		double *fc,
		double (*func)
				(double,
				struct vertices_t**,
				unsigned char*,
				unsigned char*,
				int*,
				int*,
				int*,
				int,
				int,
				cudaStream_t*,
				double**),
		struct vertices_t **verts,
		unsigned char *htype,
		unsigned char *dtype,
		int *nframes,
		int *nviews,
		int *lc_n,
		int nsets,
		int nf,
		cudaStream_t *bf_stream,
		double **fit_overflow)
{
	double ulim,u,r,q,fu,dum;

	*fa = (*func)(*ax, verts, htype, dtype, nframes, nviews, lc_n, nsets, nf, bf_stream, fit_overflow);
	*fb = (*func)(*bx, verts, htype, dtype, nframes, nviews, lc_n, nsets, nf, bf_stream, fit_overflow);
	if (*fb > *fa) {
		SHFT(dum,*ax,*bx,dum)
		SHFT(dum,*fb,*fa,dum)
	}

	*cx = (*bx)+GOLD*(*bx-*ax);
	*fc = (*func)(*cx, verts, htype, dtype, nframes, nviews, lc_n, nsets, nf, bf_stream, fit_overflow);

	while (*fb > *fc) {

		r = (*bx-*ax) * (*fb-*fc);
		q = (*bx-*cx) * (*fb-*fa);
		u = (*bx) - ((*bx-*cx)*q - (*bx-*ax)*r)/
				(2.0 * SIGN( MAX( fabs(q-r),TINY), q-r));
		ulim = (*bx) + GLIMIT * (*cx-*bx);

		if ((*bx-u)*(u-*cx) > 0.0) {
			fu=(*func)(u, verts, htype, dtype, nframes, nviews, lc_n, nsets, nf, bf_stream, fit_overflow);

			if (fu < *fc) {
				*ax=(*bx);
				*bx=u;
				*fa=(*fb);
				*fb=fu;
				return;
			} else if (fu > *fb) {
				*cx = u;
				*fc = fu;
				return;
			}
			u = (*cx) + GOLD * (*cx-*bx);
			fu = (*func)(u, verts, htype, dtype, nframes, nviews, lc_n, nsets, nf, bf_stream, fit_overflow);

		} else if ((*cx-u) * (u-ulim) > 0.0) {
			fu = (*func)(u, verts, htype, dtype, nframes, nviews, lc_n, nsets, nf, bf_stream, fit_overflow);

			if (fu < *fc) {
				SHFT(*bx,*cx,u,*cx+GOLD*(*cx-*bx))
				SHFT(*fb,*fc,fu,(*func)(u, verts, htype, dtype, nframes, nviews, lc_n, nsets, nf, bf_stream, fit_overflow))
			}
		} else if ((u-ulim)*(ulim-*cx) >= 0.0) {
			u = ulim;
			fu = (*func)(u, verts, htype, dtype, nframes, nviews, lc_n, nsets, nf, bf_stream, fit_overflow);
		} else {
			u = (*cx)+GOLD*(*cx-*bx);
			fu = (*func)(u, verts, htype, dtype, nframes, nviews, lc_n, nsets, nf, bf_stream, fit_overflow);
		}
		SHFT(*ax,*bx,*cx,u)
		SHFT(*fa,*fb,*fc,fu)
	}
}


#undef GOLD
#undef GLIMIT
#undef TINY
#undef MAX
#undef SIGN
#undef SHFT
