/* Cuda utility functions */
__device__ double atomicAdd_dbl(double* address, double val);
__global__ void clrvect_krnl(struct dat_t *ddat, int size, int s, int f);
__device__ int cubic_realroots_cuda(double *coeff, double *realroot);
__device__ void dev_bsstep(double *y, double *dydx, double *xx, double htry,
		double eps, double *yscal, double *hdid, double *hnext,
		void (*derivs)(double, double *, double *));
__device__ double dev_cel(double qqc, double pp, double aa, double bb);
__device__ void dev_cotrans1(double3 *y, double a[3][3], double3 x, int dir);
__device__ void dev_cotrans2(double y[3], double a[3][3], double x[3], int dir);
__device__ void dev_cotrans3(double3 *y, double3 *a, double3 x, int dir,
		int frm);
__device__ void dev_cotrans5(double3 *y, double a[3][3], double x[3], int dir,
		int f);
__device__ double dev_cross(double z[3], double x[3], double y[3]);
__device__ double dev_dot(double x[3], double y[3]);
__device__ float dev_dot_d3(double3 x, double3 y);
__device__ void dev_euler2mat(double m[3][3], double phi, double theta,
		double psi);
__device__ void dev_facmom(double fv0[3], double fv1[3], double fv2[3],
		double fn[3], double *dv, double dvr[3], double dI[3][3]);
__device__ double dev_facnrm(struct vertices_t verts, int fi);
__device__ int dev_gamma_trans(double *datum, double gamma);
__device__ double dev_gammln(double xx);
__device__ double dev_hapke(double cosi, double cose, double phase, double w,
		double h, double B0, double g, double theta);
__device__ void dev_inteuler(struct spin_t spin, double t[],
		double impulse[][3], int n, double w[3], double m[3][3],
		unsigned char pa, unsigned char method, double int_abstol);
__global__ void euler2mat_realize_mod_krnl(struct mod_t *dmod);
__device__ void dev_lghtcrv_splint(double *xa, double *ya, double *y2a, int n,
		double x, double *y);
__global__ void lghtcrv_spline_krnl(struct dat_t *ddat, int set, double yp1,
		double ypn, double *u, int ncalc);
__global__ void lghtcrv_splint_krnl(struct dat_t *ddat, int set, int n,
		int ncalc);
__device__ void dev_mmid(double *y, double *dydx, double xs, double htot,
		int nstep, double *yout,
		void (*dev_derivs)(double, double *, double *));
__device__ void dev_mmmul(double x[3][3], double y[3][3], double z[3][3]);
__device__ void dev_mmmul2(double3 *x, double y[3][3], double3 *z, int f);
__device__ void dev_mtrnsps(double a[3][3], double b[3][3]);
__device__ void dev_mtrnsps2(double3 *a, double b[3][3], int f);
__device__ double dev_normalize(double *u);
__device__ double dev_normalize3(double3 *u);
__device__ void dev_odeint(double ystart[13], double x1, double x2, double eps,
		double h1, double hmin, int *nok, int *nbad,
		void (*derivs)(double, double *, double *),
		void (*drkqc)(double *, double *, double *, double, double, double *,
				double *, double *, void (*)(double, double *, double *)));
__device__ double dev_plgndr(int l, int m, double x);
__device__ double dev_radlaw(struct photo_t *photo, int ilaw, double cosinc,
		int c, int f);
__device__ double dev_radlaw_mod(struct photo_t *photo, int ilaw,
		double cosinc);
__device__ void dev_realize_impulse(struct spin_t spin, double t,
		double t_integrate[], double impulse[][3], int *n_integrate, int s,
		int f, int k);
__device__ void dev_rzextr(int iest, double xest, double *yest, double *yz,
		double *dy);
__device__ int dev_vp_iround(double x);
__device__ void dev_jacobi(double a[3][3], int n, double d[3], double v[3][3],
		int *nrot);
__device__ double dev_distance(double x[3], double y[3]);
__device__ double dev_vecnorm(double x[3]);
