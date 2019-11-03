/* Flags */
extern int HMT;			/* Use Host multi-threading							*/
extern int HMT_threads;	/* Number of host threads 	*/

/* Functions for host multi-threading : */
double bestfit_hmt(struct par_t *par, struct mod_t *mod, struct dat_t *dat);
double brent_abs_hmt(double ax, double bx, double cx, double (*f) (double,
		pthread_t*), double tol, double abstol, double *xmin, pthread_t
		*hmt_thread);
void calc_fits_hmt( struct par_t *par, struct mod_t *mod, struct dat_t *dat,
		pthread_t *hmt_thread);
void mnbrak_hmt(double *ax, double *bx, double *cx, double *fa, double *fb,
	    double *fc, double (*func)(double, pthread_t*), pthread_t *hmt_stream);
int pos2deldop_hmt( struct par_t *par, struct photo_t *photo,
                double orbit_xoff, double orbit_yoff, double orbit_dopoff,
                struct deldop_t *deldop, int body, int set, int frm, int v);
int pos2doppler_hmt( struct par_t *par, struct photo_t *photo, double orbit_xoff,
		double orbit_yoff, double orbit_dopoff, struct doppler_t *doppler,
		int body, int set, int frm, int v);
void vary_params_hmt(struct par_t *par, struct mod_t *mod, struct dat_t *dat,
		double *deldop_zmax, double *rad_xsec, double *opt_brightness, double
		*cos_subradarlat, pthread_t *hmt_thread);
