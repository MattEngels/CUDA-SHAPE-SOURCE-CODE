extern "C" {
#include "../shape/head.h"
}

__device__ void dev_lghtcrv_splint(double *xa,double *ya,double *y2a,int n,double x,double *y)
{
	int klo,khi,k;
	double h,b,a;

	klo=1;
	khi=n;
	while (khi-klo > 1) {
		k=(khi+klo) >> 1;
		if (xa[k] > x) khi=k;
		else klo=k;
	}
	h=xa[khi]-xa[klo];
	if (h == 0.0) printf("Bad XA input to routine SPLINT (dev_lghtcrv_splint)\n");
	a=(xa[khi]-x)/h;
	b=(x-xa[klo])/h;
	*y=a*ya[klo]+b*ya[khi]+((a*a*a-a)*y2a[klo]+(b*b*b-b)*y2a[khi])*(h*h)/6.0;
}

__global__ void lghtcrv_splint_krnl(struct dat_t *ddat, int set, int n, int ncalc)
{
	/* Single-threaded kernel where n = lghtcrv->n */
	/* Parameters:
	 * double *xa  - lghtcrv->x
	 * double *ya  - lghtcrv->y
	 * double *y2a - lghtcrv->y2
	 * int n       - ncalc
	 * double x    - lghtcrv->t[i][lghtcrv->v0]
	 * double *y   - lghtcrv->fit[i]	 *
	 *
	 * This is a wrapper kernel that launches the device function. This is done
	 * for memory access reasons.	 */

	int i, v;
	double interp;

	if (threadIdx.x == 0) {

		for (i=1; i<=n; i++) {
			ddat->set[set].desc.lghtcrv.fit[i] = 0.0;

			for (v=0; v<ddat->set[set].desc.lghtcrv.nviews; v++) {
				dev_lghtcrv_splint(ddat->set[set].desc.lghtcrv.x,
						ddat->set[set].desc.lghtcrv.y,
						ddat->set[set].desc.lghtcrv.y2,
						ncalc,
						ddat->set[set].desc.lghtcrv.t[i][ddat->set[set].desc.lghtcrv.v0],
						&interp);
				ddat->set[set].desc.lghtcrv.fit[i] += interp;
			}
			ddat->set[set].desc.lghtcrv.fit[i] /= ddat->set[set].desc.lghtcrv.nviews;
		}
	}
}
