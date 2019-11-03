void ec2eq( double *u);
void eq2ec( double *u);
void euler2mat( double m[3][3], double phi, double theta, double psi);
void mat2euler( double m[3][3], double *phi, double *theta, double *psi);
void cal2jd( int yy, int mo, int dd, int hh, int mm, int ss,
			double *jd);
void jd2cal( int *yy, int *mo, int *dd, int *hh, int *mm, int *ss,
			double jd);
void rdcal2jd( FILE *fp, double *jd);
void wrtjd2cal( FILE *fp, double jd);
double hapke( double cosi, double cose, double phase,
			 double w, double h, double S0, double g, double theta);
int orbel2xyz( double M, double o, double O, double i, double e, double a,
			   double epoch, double t, double x[3]);
int kepler(double mu, double e, double rmin, double dt,
           double long_asc_node, double inclination, double arg_pericenter,
           double *displacement, double *velocity);
