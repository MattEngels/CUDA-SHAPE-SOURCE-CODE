static double dsqrarg;
#define DSQR(a) ((dsqrarg=(a)) == 0.0 ? 0.0 : dsqrarg*dsqrarg)

static double dmaxarg1,dmaxarg2;
#define DMAX(a,b) (dmaxarg1=(a),dmaxarg2=(b),(dmaxarg1) > (dmaxarg2) ?\
        (dmaxarg1) : (dmaxarg2))

static double dminarg1,dminarg2;
#define DMIN(a,b) (dminarg1=(a),dminarg2=(b),(dminarg1) < (dminarg2) ?\
        (dminarg1) : (dminarg2))

double *vector(int, int);
double *vector_zero(int nl,int nh);
double **matrix(int, int, int, int);
double **matrix_zero(int nrl,int nrh,int ncl,int nch);
double **convert_matrix();
double *dvector(int, int);
double **dmatrix(int, int, int, int);
int *ivector(int, int);
unsigned long *lvector(int, int);
int **imatrix(int, int, int, int);
double **submatrix(double **, int, int, int, int, int, int);
void free_vector(double *, int, int);
void free_dvector(double *, int, int);
void free_ivector(int *, int, int);
void free_lvector(unsigned long *, int, int);
void free_matrix(double **, int, int, int, int);
void free_dmatrix(double **, int, int, int, int);
void free_imatrix(int **, int, int, int, int);
void free_submatrix(double **, int, int, int, int);
void free_convert_matrix(double **, int, int, int, int);
void nrerror(const char *);

typedef struct FCOMPLEX {double r,i;} fcomplex;

fcomplex Cadd(fcomplex a, fcomplex b);
fcomplex Csub(fcomplex a, fcomplex b);
fcomplex Cmul(fcomplex a, fcomplex b);
fcomplex Complex(double re, double im);
fcomplex Conjg(fcomplex z);
fcomplex Cdiv(fcomplex a, fcomplex b);
double Cabs(fcomplex z);
fcomplex Csqrt(fcomplex z);
fcomplex RCmul(double x, fcomplex a);

double brent(double ax, double bx, double cx, double (*f)(double), double tol,
	     double *xmin);
void  bsstep(double *y, double *dydx, double *xx, double htry,
	     double eps, double *yscal, double *hdid, double *hnext, 
	     void (*derivs)(double,double *,double *));
void  caldat(long julian, int *mm, int *id, int *iyyy);
double cel(double qqc, double pp, double aa, double bb);
double gammln(double xx);
long  julday(int mm, int id, int iyyy);
void  lubksb(double **a, int n, int *indx, double *b);
void  ludcmp(double **a, int n, int *indx, double *d);
void mmid( double *y, double *dydx, int nvar, double xs, double htot,
		  int nstep, double *yout, 
		  void (*derivs)( double, double *, double *));
void  mnbrak(double *ax, double *bx, double *cx, double *fa, double *fb, 
		  double *fc, double (*func)(double));
void  odeint(double *ystart, int nvar, double x1, double x2, double eps,
		  double h1, double hmin, int *nok, int *nbad,
		  void (*derivs)(double,double *,double *),
		  void  (*rkqc)(double *,double *,double *,double,double,double
				*,double *,double *,void (*)(double,double *,double *)));
void  piksrt(int n, double *arr);
double plgndr(int l, int m, double x);
double ran1(int *idum);
void rzextr( int iest, double xest, double *yest, double *yz, double *dy,
			int nv, int nuse);
void  spline(double *x, double *y, int n, double yp1, double ypn, double *y2);
void  splint(double *xa, double *ya, double *y2a, int n, double x, double *y);
void powell(double *p,double **xi,int n,double ftol,int *iter,double *fret,
			double (*func)(double *p));
void linmin(double *p,double *xi,int n,double *fret,double (*func)(double *p));
double f1dim(double x);
double gasdev(int *idum);
double poidev( double xm, int *idum);
double zbrent( double (*func)(double), double x1, double x2, double tol);
double qsimp(double (*func)(double), double a, double b);
double trapzd(double (*func)(double), double a, double b, int n);
void jacobi( double **a, int n, double *d, double **v, int *nrot);
double bessj(int n, double x);
double bessj0(double x);
double bessj1(double x);
double bessk(int n, double x);
double bessk0(double x);
double bessk1(double x);
double bessy(int n, double x);
double bessy0(double x);
double bessy1(double x);
double factrl(int n);
double rtbis(double (*func)(double), double x1, double x2, double xacc);
void svbksb(double **u, double w[], double **v, int m, int n, double b[],
	double x[]);
void svdcmp(double **a, int m, int n, double w[], double **v);
void hpsort(unsigned long n, double ra[]);
double ***d3tensor(int nrl, int nrh, int ncl, int nch, int ndl, int ndh);
void free_d3tensor(double ***t, int nrl, int nrh, int ncl, int nch,
	int ndl, int ndh);
double brent_abs(double ax, double bx, double cx, double (*f)(double),
	double tol, double abstol, double *xmin);
void bcucof(double y[], double y1[], double y2[], double y12[], double d1,
	double d2, double **c);
void bcuint(double y[], double y1[], double y2[], double y12[],
	double x1l, double x1u, double x2l, double x2u, double x1,
	double x2, double *ansy, double *ansy1, double *ansy2);
void indexx(unsigned long n, double arr[], unsigned long indx[]);
void sort3(unsigned long n, double ra[], double rb[], double rc[]);
void sort3dii(unsigned long n, double ra[], int rb[], int rc[]);
void sncndn(double uu, double emmc, double *sn, double *cn, double *dn);
double rf(double x, double y, double z);
double ellf(double phi, double ak);
void laguer(fcomplex a[], int m, fcomplex *x, int *its);
void zroots(fcomplex a[], int m, fcomplex roots[], int polish);
