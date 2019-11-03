/*
Modified 2016 July 7 by Matt Engels:
	Adapted for use in shape-cuda.
*/

void euler2mat( double m[3][3], double phi, double theta, double psi);
void mat2euler( double m[3][3], double *phi, double *theta, double *psi);
void free_ucvector(unsigned char *v,int nl,int nh);
double sinc2(double);
double sinc(double);
int iround(double);
double ***tensor(int,int,int,int,int,int);
unsigned char *ucvector(int, int);
double shnorm(int,int);
float *fvector(int, int);
void free_fvector(float *, int, int);
unsigned char **ucmatrix(int nrl,int nrh,int ncl,int nch);
int gettstr( FILE *fp, char *str);
int string_to_int( char *str);
int getint( FILE *fp);
double string_to_double( char *str);
double getdouble( FILE *fp);
double cross( double z[3], double x[3], double y[3]);
double normalize( double *u);
void cotrans( double y[3], double a[3][3], double x[3], int dir);
void mmmul( double x[3][3], double y[3][3], double z[3][3]);
double bessjp( int n, double x);
double bessyp( int n, double x);
double besskp( int n, double x);
double fiteuler( double m[3][3], double *phi, double *theta, double *psi);
void mtrnsps( double a[3][3], double b[3][3]);
double chi2_1d( double *o, double *f, int imin, int imax,
			   double *s, int stype, int scale);
long double *ldvector(int nl,int nh);
void clrmat( double **mat, int i1, int i2, int j1, int j2);
double dot( double x[3], double y[3]);
int resampim( double **im1, int x11, int x12, int y11, int y12,
			  double **im2, int x21, int x22, int y21, int y22,
			  double x0, double xwidth, double y0, double ywidth,
			  double rotangle, int interpolation_mode, int rebin);
double distance( double x[3], double y[3]);
double readwf( char *name, int *nv, double ***v, int *nf, int ***f);
double facnorm( double v0[3], double v1[3], double v2[3], double n[3]);
double matinv( double **m, int n);
void clrvect( double *v, int i1, int i2);
double linint3d( double x[3], double f[2][2][2]);
void gray2rgb( double gray, double *r, double *g, double *b);
int inlist( int i, int *list, int n);
double ptoldist( double p[3], double v0[3], double v1[3], double x[3]);
double ptofdist( double p[3], double v0[3], double v1[3], double v2[3],
				double x[3]);
int intfac( double u[3], double r0[3],
	   double v0[3], double v1[3], double v2[3],
	   double *r, double n[3], double x[3]);
void resamp( double *v1, int N1, double *v2, int N2);
void lowcase( char *str);
int imod( int i, int n);
int inwf( double **v, int **f, int nv, int nf, double x[]);
void dms( double deg, int *dd, int *mm, int *ss);
double vmax( double *v, int n);
double vmin( double *v, int n);
double vav( double *v, int n);
int readline(FILE *fp, char line[], int maxSize);
int allwhite(char *string);
double vecnorm( double x[3]);
unsigned char ***uc3tensor(int nrl, int nrh, int ncl, int nch, int ndl, int ndh);
void free_uc3tensor(unsigned char ***t, int nrl, int nrh, int ncl, int nch,
	int ndl, int ndh);
void free_ucmatrix(unsigned char **m, int nrl, int nrh, int ncl, int nch);
int addsuffix(char *outstring, int maxSize,
              char *instring, char *oldsuffix, char *newsuffix);
void changepath(char *outstring, int maxSize, char *instring, char *newpath);
int direxists(char *dirname);
int createdir(char *dirname);
void waitsecs( double seconds);
char *intifpossible(char *valstring, int n, double val, double abstol,
                    const char *floatformat);
int is_little_endian();
short swap_short(short x);
unsigned short swap_ushort(unsigned short x);
int swap_int(int x);
unsigned int swap_uint(unsigned int x);
long swap_long(long x);
unsigned long swap_ulong(unsigned long x);
float swap_float(float x);
double swap_double(double x);
long double swap_ldouble(long double x);
void swapbuf_short(short *buf, int nvals);
void swapbuf_ushort(unsigned short *buf, int nvals);
void swapbuf_int(int *buf, int nvals);
void swapbuf_uint(unsigned int *buf, int nvals);
void swapbuf_long(long *buf, int nvals);
void swapbuf_ulong(unsigned long *buf, int nvals);
void swapbuf_float(float *buf, int nvals);
void swapbuf_double(double *buf, int nvals);
void swapbuf_ldouble(long double *buf, int nvals);
int nomoredata( FILE *fp);
int countdata( FILE *fp);
void tempname(const char *outname, int maxSize, char *prefix, char *suffix);
void timestring(char *outstring, int maxSize, char *formatstring);
int checkposet( double **im, int x1, int x2, int y1, int y2,
                double x0, double xwidth, double y0, double ywidth,
                double rotangle, double *badposet_factor);
