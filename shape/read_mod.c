/*****************************************************************************************
                                                                               read_mod.c

Reads an asteroid model of the type used by shape.c.  Provide a file name and a mod_t
structure.  Returned integer value is the number of free parameters.  Also sets up the
shape realization(s).

Modified 2015 December 3 by CM:
    Include law number when displaying optical scattering law(s) to screen

Modified 2014 August 22 by SN:
    Added spin.lib_amp, spin.lib_freq, spin.lib_phase to take into account
    librations. 
    Added a check to prevent using librations and NPA rotation at the same 
    time. 

Modified 2014 March 12 by CM:
    Bug fix: was setting number of optical scattering laws to number of radar laws

Modified 2014 February 12 by CM:
    Implement multiple radar and optical scattering laws

Modified 2013 May 19 by CM:
    Implement ovoid shape components

Modified 2012 April 2 by CM:
    Only carry out the preceding checks (that all albedo parameters are allowed to float
        when "vary_radalb" and/or "vary_optalb" are turned on) for the "fit" action,
        since the "vary_radalb" and "vary_optalb" parameters have no effect for any other
        action

Modified 2011 September 10 by CM:
    If the "vary_radalb" parameter is turned on, check that all nonzero radar albedo
        parameters -- R for a homogeneous law, a and b coefficients for a harmonic
        inhomogeneous law, global and local R values for a faceted inhomogeneous law --
        are allowed to float, and quit if any are held constant; then carry out the
        analogous check for optical albedo if the "vary_optalb" parameter is turned on.
    Quit rather than just printing a warning if the mod file is missing the YORP
        components or the number of spin impulses
    Add "harmlambert" and "inholambert" optical scattering laws

Modified 2011 August 12 by CM:
    Read spin impulses

Modified 2010 June 1 by CM:
    Change "scalefactor" parameter from a scalar to a 3-component vector
        for harmonic and vertex shape structures

Modified 2010 April 27 by CM:
    Enabled "tabular" radar scattering law

Modified 2010 March 21 by CM:
    Implement '=' state for vertex deviations

Modified 2009 November 15 by CM:
    Eliminate unused variable

Modified 2009 August 2 by CM:
    Add check to "setupsides" routine that the three vertex numbers for
        each facet are in bounds (0 <= v < nv) and that no two are equal
    Permit the "harmlommel" "harmhapke" "harmkaas" and "harmcosine_diff"
        inhomogeneous scattering laws to be used with multiple-component
        models

Modified 2009 July 24 by CM:
    Fixed bug initializing the numbers of the two facets attached to
        each side
    Improved error message in setupsides routine

Modified 2009 July 5 by CM:
    Eliminate computation of the rotation matrix that corresponds to each
        component's angular offsets; compute it instead in the "realize_mod"
        routine, in case the offsets are allowed to float

Modified 2009 July 1 by CM:
    Eliminate initialization of the "act" flag for facets in vertex
        realizations, as this is now done in the "realize_mod" routine

Modified 2007 August 10 by CM:
    Eliminate unused variables, initialize others

Modified 2007 August 4 by CM:
    Shift spin Euler angles to the range [0,360) deg rather than
        (-180,180] deg

Modified 2007 January 13 by CM:
    Add a missing line for the "harmhapke" law (allocating memory for the
        "b_save" matrix)

Modified 2006 October 1 by CM:
    Add "scalefactor" to harmonic and vertex shape structures
    Replace ellipsoid diameters D with two_a, a_over_b, b_over_c
    Implement "vary_radalb" and "vary_optalb" parameters by
        allocating memory for and initializing various "saved"
        photometric parameters (R_save, w_save, etc.)

Modified 2006 April 10 by PT:
    Implement case statements for reading final lines of mod file
        in case of end of file encounter

Modified 2006 April 8 by PT:
    Fix memory leak in setupvertices subroutine

Modified 2006 April 7 by PT:
    Implement checks for "spin.omegadot" parameters if they are 
        not included in the mod file or incorrectly declared
    Implement more detailed checks for PA/NPA rotation for when
        YORP parameters are included

Modified 2006 March 6 by PT:
    Added "spin.omegadot" parameters for changing spin rates

Modified 2005 September 8 by CM:
    Implement "harmlommel" "harmhapke" and "harmkaas" optical
        scattering laws
    Implement "harmcosine" radar scattering law

Modified 2005 August 17 by CM:
    Moved computation of spherical harmonic functions afactor and bfactor
        from realize_mod.c to here, so that it can be done just once at
        the start of a fit, not every time a harmonic model is realized

Modified 2005 August 8 by CM:
    Enabled "inhokaas" optical scattering law

Modified 2005 July 20 by CM:
    Enabled "hagfors" and "cosine_qs" and "gauss+cosine" and
        "hagfors+cosine" and "cosine+cosine" and inhomogeneous "inhocosine"
        radar scattering laws
    Eliminated "flat" radar scattering law

Modified 2005 July 4 by CM:
    Enabled inhomogeneous "inholommel" and "inhohapke" optical scattering
        laws to be used with ellipsoid and harmonic models, so long as you
        know in advance how many facets the vertex realization of each
        ellipsoid or harmonic model component will have

Modified 2005 June 27 by CM:
    Renamed "round" function to "iround" to avoid conflicts

Modified 2005 April 15 by CM:
    Fixed bug in handling cases with |ecliptic pole latitude| > 90 deg:
        must add 180 deg to "angle 2" and not only to "angle 0"

Modified 2005 February 28 by CM:
    Eliminate "realize_photo" call for INHOHAPKE optical scattering law,
        since this call will be made elsewhere

Modified 2005 January 25 by CM:
    Take care of unused and uninitialized variables

Modified 2005 January 9 by CM:
    For parallel processing, suppress most screen output for nodes
        other than root

Modified 2004 November 1 by CM:
    Deal with spin vectors which are "over the pole":
        |ecliptic pole latitude| > 90 deg

Modified 2004 August 6 by CM:
    Shift all Euler angles and Euler angle offsets to the range
        (-180, +180] degrees

Modified 2004 April 29 by CM:
    For Kaasalainen scattering law, switch from weighting factor "c"
        (ranging from 0 to infinity) to "wt" (ranging from 0 to 1)

Modified 2004 February 25 by CM:
    Add Kaasalainen "Lambert + Lommel-Seeliger" scattering law
    realize_photo now takes the model's parameter structure as an
        argument, so the parameters must now be passed to read_photo

Modified 2003 April 30 by CM:
    Add the parameter structure to the arguments, so that
        "theta_steps" can be used to override "ntheta" for
        ellipsoid and harmonic components

Modified 2003 April 17 by CM:
    In function read_spin, spin state is now set to PA only
        if the first two spin components are *constant* and zero.
 *****************************************************************************************/

#include "head.h"

int read_shape( FILE *fp, struct par_t *par, struct mod_t *mod);
int read_photo( FILE *fp, struct par_t *par, struct mod_t *mod);
int read_spin( FILE *fp, struct mod_t *mod);
void setupreal( struct mod_t *mod);
void setuprealver( struct mod_t *mod, int c, int **iv, int nt, int *np);
void setuprealfac( struct mod_t *mod, int c, int **iv, int nt, int *np);
void setupsides( struct vertices_t *vt);
void setupvertices( struct vertices_t *vt);
void *read_mod_pthread_sub(void *ptr);

typedef struct read_mod_thread_t
{
    int thread_no;
	struct par_t *parameter;
    struct mod_t *model;
    int nfpar;
    int gpuid;
 } read_mod_data;

int read_mod( struct par_t *par, struct mod_t *mod)
{
	FILE *fp;
	int c, nfpar=0;

	printf("\n# reading model from file: %s ...\n", mod->name);
	fflush(stdout);
	//gpuErrchk(cudaSetDevice(GPU0));

	FOPEN( fp, mod->name, "r");
	nfpar += read_shape( fp, par, mod);
	setupreal( mod);
	nfpar += read_photo( fp, par, mod);
	nfpar += read_spin( fp, mod);
	fclose( fp);
	printf("# finished reading model file\n");
	fflush(stdout);

	for (c=0; c<mod->shape.ncomp; c++) {
		setupsides(&mod->shape.comp[c].real);
		setupvertices(&mod->shape.comp[c].real);
	}
	return nfpar;
}

__host__ int read_mod_pthread( struct par_t *par, struct mod_t *mod0, struct mod_t *mod1,
		pthread_t thread1, pthread_t thread2)
{
	/* Note:  The purpose of pthreading read_mod is to create a mod for each
	 * pthread and associated gpu.  This reduces memory access bottlenecks in
	 * later functions, primarily posvis_gpu. */
	read_mod_data data1, data2;

	data1.thread_no = 0;
	data2.thread_no = 1;
	data1.parameter = data2.parameter = par;
	data1.model = mod0;
	data2.model = mod1;
	data1.nfpar = data2.nfpar = 0;
	data1.gpuid = GPU0;
	data2.gpuid = GPU1;

	/* From here, launch the pthreaded subfunction */
	pthread_create(&thread1, NULL, read_mod_pthread_sub,(void*)&data1);
	pthread_create(&thread2, NULL, read_mod_pthread_sub,(void*)&data2);

	pthread_join(thread1, NULL);
	pthread_join(thread2, NULL);

	gpuErrchk(cudaSetDevice(GPU0));

	return data1.nfpar;
}

void *read_mod_pthread_sub(void *ptr) {

	FILE *fp;
	int c, nfpar=0;
	read_mod_data *data;
	data = (read_mod_data *) ptr;  /* type cast to a pointer to thdata */
	gpuErrchk(cudaSetDevice(data->gpuid));

	if (data->gpuid==GPU0) {
		printf("\n# reading model from file: %s ...\n", data->model->name);
		fflush(stdout);
	}

	FOPEN( fp, data->model->name, "r");
	data->nfpar += read_shape(fp, data->parameter, data->model);
	setupreal(data->model);
	data->nfpar += read_photo(fp, data->parameter, data->model);
	data->nfpar += read_spin(fp, data->model);
	fclose(fp);

	if (data->gpuid==GPU0) {
		printf("# finished reading model file\n");
		fflush(stdout);
	}

	for (c=0; c<data->model->shape.ncomp; c++) {
		setupsides(&data->model->shape.comp[c].real);
		setupvertices(&data->model->shape.comp[c].real);
	}
}

int read_shape( FILE *fp, struct par_t *par, struct mod_t *mod)
{
	char type[80];
	int i, j, nfpar=0, k, nv, mirror, v, match, nf;

	mod->shape.ncomp = getint( fp);      /* read # of components in shape */
	printf("# shape has %d components\n", mod->shape.ncomp);
	fflush(stdout);

	/*=======================================================================*/
	/* if gpu processing via CUDA is enabled, allocate via CUDA function.
	 * If it is not enabled, allocate via the standard C call.				 */

	if (CUDA)
		cudaCalloc((void**)&mod->shape.comp, sizeof(struct comp_t), mod->shape.ncomp);
	else
		mod->shape.comp = (struct comp_t*) calloc(mod->shape.ncomp,
				sizeof( struct comp_t));
	/*=======================================================================*/

	for (i=0; i<mod->shape.ncomp; i++) { /* read each component */
		for (j=0; j<=2; j++) {                              /* read linear offsets */
			if (readparam( fp, &mod->shape.comp[i].off[j]))
				++nfpar;
		}
		for (j=0; j<=2; j++) {                              /* read angular offsets */
			if (readparam( fp, &mod->shape.comp[i].rot[j]))
				++nfpar;
			mod->shape.comp[i].rot[j].val +=
					360.0*floor((180.0 - mod->shape.comp[i].rot[j].val)/360.0);
			mod->shape.comp[i].rot[j].val *= D2R; /* convert to radians */
		}
		gettstr( fp, type);                                 /* read component type */
		printf("# component %d is type %s\n", i, type);
		fflush(stdout);

		if (!strcmp( "ellipse", type)) {
			mod->shape.comp[i].type = ELLIPSE;
			if (readparam( fp, &mod->shape.comp[i].desc.ell.two_a))
				++nfpar;
			if (readparam( fp, &mod->shape.comp[i].desc.ell.a_over_b))
				++nfpar;
			if (readparam( fp, &mod->shape.comp[i].desc.ell.b_over_c))
				++nfpar;
			mod->shape.comp[i].desc.ell.ntheta = getint( fp);
			if (par->theta_steps <= 0 ||
					par->theta_steps == mod->shape.comp[i].desc.ell.ntheta) {
				printf("# %d theta steps\n", mod->shape.comp[i].desc.ell.ntheta);
			} else {
				printf("# theta steps changed from %d to %d\n",
						mod->shape.comp[i].desc.ell.ntheta, par->theta_steps);
				mod->shape.comp[i].desc.ell.ntheta = par->theta_steps;
			}
			fflush(stdout);
		}

		else if (!strcmp( "ovoid", type)) {
			mod->shape.comp[i].type = OVOID;
			if (readparam( fp, &mod->shape.comp[i].desc.ovoid.two_a))
				++nfpar;
			if (readparam( fp, &mod->shape.comp[i].desc.ovoid.a_over_b))
				++nfpar;
			if (readparam( fp, &mod->shape.comp[i].desc.ovoid.b_over_c))
				++nfpar;
			if (readparam( fp, &mod->shape.comp[i].desc.ovoid.k))
				++nfpar;
			mod->shape.comp[i].desc.ovoid.ntheta = getint( fp);
			if (par->theta_steps <= 0 ||
					par->theta_steps == mod->shape.comp[i].desc.ovoid.ntheta) {
				printf("# %d theta steps\n", mod->shape.comp[i].desc.ovoid.ntheta);
			} else {
				printf("# theta steps changed from %d to %d\n",
						mod->shape.comp[i].desc.ovoid.ntheta, par->theta_steps);
				mod->shape.comp[i].desc.ovoid.ntheta = par->theta_steps;
			}
			fflush(stdout);
		}

		else if (!strcmp( "harmonic", type)) {
			mod->shape.comp[i].type = HARMONIC;
			mod->shape.comp[i].desc.har.nhar = getint( fp);
			printf("# %d harmonics\n", mod->shape.comp[i].desc.har.nhar);


			/*=======================================================================*/
			/* if gpu processing via CUDA is enabled, allocate via CUDA function.
			* If it is not enabled, allocate via the standard C call.				 */

			if (CUDA) {
				cudaCalloc((void**)&mod->shape.comp[i].desc.har.a, sizeof(struct param_t *), 
					mod->shape.comp[i].desc.har.nhar + 1);
				cudaCalloc((void**)&mod->shape.comp[i].desc.har.b, sizeof(struct param_t *), 
					mod->shape.comp[i].desc.har.nhar + 1);
			}
			else {
				mod->shape.comp[i].desc.har.a = (struct param_t **)
					calloc(mod->shape.comp[i].desc.har.nhar + 1,
					sizeof(struct param_t *));
				mod->shape.comp[i].desc.har.b = (struct param_t **)
					calloc(mod->shape.comp[i].desc.har.nhar + 1,
					sizeof(struct param_t *));
			}
			/*=======================================================================*/
			
			for (j=0; j<=2; j++)
				if (readparam( fp, &mod->shape.comp[i].desc.har.scalefactor[j]))
					++nfpar;
			if (mod->shape.comp[i].desc.har.scalefactor[0].state == '=')
				bailout("can't use '=' state for 'scale factor 0'\n");
			for (j=0; j<=mod->shape.comp[i].desc.har.nhar; j++) {
				
				/*=======================================================================*/
				/* if gpu processing via CUDA is enabled, allocate via CUDA function.
				* If it is not enabled, allocate via the standard C call.				 */

				if (CUDA) {
					cudaCalloc((void**)&mod->shape.comp[i].desc.har.a[j], sizeof(struct param_t), j + 1);
					cudaCalloc((void**)&mod->shape.comp[i].desc.har.b[j], sizeof(struct param_t), j + 1);
				}
				else {				
					mod->shape.comp[i].desc.har.a[j] = (struct param_t *)
							calloc( j+1, sizeof( struct param_t));
					mod->shape.comp[i].desc.har.b[j] = (struct param_t *)
							calloc( j+1, sizeof( struct param_t));
				}
				/*=======================================================================*/

				for (k=0; k<=j; k++) {
					if (readparam( fp, &mod->shape.comp[i].desc.har.a[j][k]))
						++nfpar;
					if (k > 0)
						if (readparam( fp, &mod->shape.comp[i].desc.har.b[j][k]))
							++nfpar;
				}
			}
			mod->shape.comp[i].desc.har.ntheta = getint( fp);
			if (par->theta_steps <= 0 ||
					par->theta_steps == mod->shape.comp[i].desc.har.ntheta) {
				printf("# %d theta steps\n", mod->shape.comp[i].desc.har.ntheta);
			} else {
				printf("# theta steps changed from %d to %d\n",
						mod->shape.comp[i].desc.har.ntheta, par->theta_steps);
				mod->shape.comp[i].desc.har.ntheta = par->theta_steps;
			}
			fflush(stdout);
		}

		else if (!strcmp( "vertex", type)) {
			mod->shape.comp[i].type = VERTEX;
			mod->shape.comp[i].desc.ver.nv = getint( fp);
			nv = mod->shape.comp[i].desc.ver.nv;
			printf("# %d vertices\n", nv);
			
			/*=======================================================================*/
			/* if gpu processing via CUDA is enabled, allocate via CUDA function.
			* If it is not enabled, allocate via the standard C call.				 */

			if (CUDA) 
				cudaCalloc((void**)&mod->shape.comp[i].desc.ver.v,
						sizeof(struct vertex_t), nv);
			else 
				mod->shape.comp[i].desc.ver.v = (struct vertex_t *)
						calloc( nv, sizeof( struct vertex_t));
			/*=======================================================================*/
			
			for (j=0; j<=2; j++)
				if (readparam( fp, &mod->shape.comp[i].desc.ver.scalefactor[j]))
					++nfpar;
			if (mod->shape.comp[i].desc.ver.scalefactor[0].state == '=')
				bailout("can't use '=' state for 'scale factor 0'\n");

			/*  Read in vertices  */

			mirror = 0;
			for (j=0; j<nv; j++) {
				if (readparam( fp, &mod->shape.comp[i].desc.ver.v[j].r))
					++nfpar;
				for (k=0; k<=2; k++)
					mod->shape.comp[i].desc.ver.v[j].u[k] = getdouble( fp);
				for (k=0; k<=2; k++)
					mod->shape.comp[i].desc.ver.v[j].a[k] = getdouble( fp);
				if (mod->shape.comp[i].desc.ver.v[j].r.state == '=')
					mirror = 1;
				else
					mod->shape.comp[i].desc.ver.v[j].v_mirror = -1;
			}

			/*  If any vertices have the '=' state for the vertex deviation -- meaning that
          this is a southern vertex that will mirror a northern vertex -- find the
          corresponding northern mirror vertices                                       */

			if (mirror)
				for (j=0; j<nv; j++) {
					if (mod->shape.comp[i].desc.ver.v[j].r.state == '=') {
						v = 0;
						match = 0;
						do {
							if (v != j
									&& fabs(mod->shape.comp[i].desc.ver.v[v].a[0] -
											mod->shape.comp[i].desc.ver.v[j].a[0]   ) < SMALLVAL
											&& fabs(mod->shape.comp[i].desc.ver.v[v].a[1] -
													mod->shape.comp[i].desc.ver.v[j].a[1]   ) < SMALLVAL
													&& fabs(mod->shape.comp[i].desc.ver.v[v].a[2] +
															mod->shape.comp[i].desc.ver.v[j].a[2]   ) < SMALLVAL
															&& fabs(mod->shape.comp[i].desc.ver.v[v].u[0] -
																	mod->shape.comp[i].desc.ver.v[j].u[0]   ) < SMALLVAL
																	&& fabs(mod->shape.comp[i].desc.ver.v[v].u[1] -
																			mod->shape.comp[i].desc.ver.v[j].u[1]   ) < SMALLVAL
																			&& fabs(mod->shape.comp[i].desc.ver.v[v].u[2] +
																					mod->shape.comp[i].desc.ver.v[j].u[2]   ) < SMALLVAL)
								match = 1;
							else
								v++;
						} while (!match && v < nv);
						if (!match) {
							printf("No mirror vertex found for component %d vertex %d\n", i, j);
							bailout("read_mod.c\n");
						}
						mod->shape.comp[i].desc.ver.v[j].v_mirror = v;
					}
				}

			/*  Read in facets  */

			mod->shape.comp[i].desc.ver.nf = getint( fp);
			nf = mod->shape.comp[i].desc.ver.nf;
			printf("# %d facets\n", nf);
			
			/*=======================================================================*/
			/* if gpu processing via CUDA is enabled, allocate via CUDA function.
			* If it is not enabled, allocate via the standard C call.				 */

			if (CUDA)
				cudaCalloc((void**)&mod->shape.comp[i].desc.ver.f, sizeof(struct facet_t), nf);
			else
				mod->shape.comp[i].desc.ver.f = (struct facet_t *)
				calloc(nf, sizeof(struct facet_t));
			/*=======================================================================*/
						
			for (j=0; j<nf; j++)
				for (k=0; k<=2; k++) {
					if (j==157)
						printf("\n");
					mod->shape.comp[i].desc.ver.f[j].v[k] = getint( fp);
				}
			fflush(stdout);
		}

		else
			bailout("read_mod.c: invalid shape description\n");

	}
	return nfpar;
}

int read_photo( FILE *fp, struct par_t *par, struct mod_t *mod)
{
	char str[80];
	int noncosine_initialized, ilaw, nfpar=0, nf, c, f, nc, L, l, m, i, n,
			globalflag;
	double angres, cutoff_angle;

	/*  Initialize a flag  */

	noncosine_initialized = 0;

	/*  Read # of radar scattering laws and allocate memory  */

	mod->photo.nradlaws = getint( fp);

	/*=======================================================================*/
	/* if gpu processing via CUDA is enabled, allocate via CUDA function.
	* If it is not enabled, allocate via the standard C call.				 */

	if (CUDA) {
		cudaCalloc((void**)&mod->photo.radtype, sizeof(unsigned char), mod->photo.nradlaws);
		cudaCalloc((void**)&mod->photo.radar, sizeof(union radscat_t),
				mod->photo.nradlaws);
	}
	else {
		mod->photo.radtype = ucvector( 0, mod->photo.nradlaws-1);
		mod->photo.radar = (union radscat_t*) calloc(mod->photo.nradlaws,
			sizeof(union radscat_t));
	}
	/*=======================================================================*/
	
	/*  Read each radar scattering law  */

	for (ilaw=0; ilaw<mod->photo.nradlaws; ilaw++) {
		gettstr( fp, str);
		printf("# radar scattering law %d: %s\n", ilaw, str);
		fflush(stdout);

		if (!strcmp( str, "none"))
			mod->photo.radtype[ilaw] = NOLAW;
		else if (!strcmp( str, "cosine")) {
			mod->photo.radtype[ilaw] = COSINELAW_DIFF;
			if (readparam( fp, &mod->photo.radar[ilaw].RC.R))
				++nfpar;
			if (par->action == FIT && par->vary_radalb && mod->photo.radar[ilaw].RC.R.state == 'c')
				bailout("read_mod.c: must let R float if vary_radalb = yes\n");
			mod->photo.radar[ilaw].RC.R_save = mod->photo.radar[ilaw].RC.R.val;
			if (readparam( fp, &mod->photo.radar[ilaw].RC.C))
				++nfpar;
		}
		else if (!strcmp( str, "tabular")) {
			mod->photo.radtype[ilaw] = TABULARLAW;
			mod->photo.radar[ilaw].tabular.n = getint( fp);
			n = mod->photo.radar[ilaw].tabular.n;
			
			/*=======================================================================*/
			/* if gpu processing via CUDA is enabled, allocate via CUDA function.
			* If it is not enabled, allocate via the standard C call.				 */

			if (CUDA) {
				cudaCalloc((void**)&mod->photo.radar[ilaw].tabular.rho, 
					sizeof(struct param_t), n);
				cudaCalloc((void**)&mod->photo.radar[ilaw].tabular.rho_save,
						sizeof(double), n);
			}
			else {
				mod->photo.radar[ilaw].tabular.rho = (struct param_t *)
					calloc(n, sizeof(struct param_t));
				mod->photo.radar[ilaw].tabular.rho_save = vector( 0, n-1);
			}
			/*=======================================================================*/			
			for (i=0; i<n; i++) {
				if (readparam( fp, &mod->photo.radar[ilaw].tabular.rho[i]))
					++nfpar;
				if (par->action == FIT && par->vary_radalb
						&& mod->photo.radar[ilaw].tabular.rho[i].state == 'c'
						&& mod->photo.radar[ilaw].tabular.rho[i].val != 0.0)
					bailout("read_mod.c: must let all nonzero rho values float if vary_radalb = yes\n");
				mod->photo.radar[ilaw].tabular.rho_save[i] = mod->photo.radar[ilaw].tabular.rho[i].val;
			}
			if (mod->photo.radar[ilaw].tabular.rho[0].state == '=')
				bailout("can't use '=' state for coefficient 0 of the 'tabular' scattering law\n");

			/*  If it hasn't already been done, allocate memory for two storage vectors
          used by the 'noncosine' penalty function, initialize the first one, and
          compute its mean value                                                   */

			if (!noncosine_initialized) {
				par->pen.x_noncosine = vector( 0, n-2);
				par->pen.y_noncosine = vector( 0, n-2);
				par->pen.xmean_noncosine = 0.0;
				angres = (PIE/2) / (n - 1);
				for (i=0; i<(n-1); i++) {
					par->pen.x_noncosine[i] = log10(cos(i*angres));
					par->pen.xmean_noncosine += par->pen.x_noncosine[i];
				}
				par->pen.xmean_noncosine /= (n - 1);
				noncosine_initialized = 1;
			}
		}
		else if (!strcmp( str, "gaussian")) {
			mod->photo.radtype[ilaw] = GAUSSIANLAW;
			if (readparam( fp, &mod->photo.radar[ilaw].quasispec.R))
				++nfpar;
			if (par->action == FIT && par->vary_radalb
					&& mod->photo.radar[ilaw].quasispec.R.state == 'c')
				bailout("read_mod.c: must let R float if vary_radalb = yes\n");
			mod->photo.radar[ilaw].quasispec.R_save = mod->photo.radar[ilaw].quasispec.R.val;
			if (readparam( fp, &mod->photo.radar[ilaw].quasispec.C))
				++nfpar;
			cutoff_angle = getdouble( fp);
			if (cutoff_angle <=  0.0 || cutoff_angle > 90.0)
				bailout("read_mod.c: must have 0.0 < cutoff_angle <= 90.0 deg\n");
			mod->photo.radar[ilaw].quasispec.cos_cutoff = cos( cutoff_angle*D2R);
		}
		else if (!strcmp( str, "hagfors")) {
			mod->photo.radtype[ilaw] = HAGFORSLAW;
			if (readparam( fp, &mod->photo.radar[ilaw].quasispec.R))
				++nfpar;
			if (par->action == FIT && par->vary_radalb
					&& mod->photo.radar[ilaw].quasispec.R.state == 'c')
				bailout("read_mod.c: must let R float if vary_radalb = yes\n");
			mod->photo.radar[ilaw].quasispec.R_save = mod->photo.radar[ilaw].quasispec.R.val;
			if (readparam( fp, &mod->photo.radar[ilaw].quasispec.C))
				++nfpar;
			cutoff_angle = getdouble( fp);
			if (cutoff_angle <=  0.0 || cutoff_angle > 90.0)
				bailout("read_mod.c: must have 0.0 < cutoff_angle <= 90.0 deg\n");
			mod->photo.radar[ilaw].quasispec.cos_cutoff = cos( cutoff_angle*D2R);
		}
		else if (!strcmp( str, "cosine_qs")) {
			mod->photo.radtype[ilaw] = COSINELAW_QS;
			if (readparam( fp, &mod->photo.radar[ilaw].quasispec.R))
				++nfpar;
			if (par->action == FIT && par->vary_radalb
					&& mod->photo.radar[ilaw].quasispec.R.state == 'c')
				bailout("read_mod.c: must let R float if vary_radalb = yes\n");
			mod->photo.radar[ilaw].quasispec.R_save = mod->photo.radar[ilaw].quasispec.R.val;
			if (readparam( fp, &mod->photo.radar[ilaw].quasispec.C))
				++nfpar;
			cutoff_angle = getdouble( fp);
			if (cutoff_angle <=  0.0 || cutoff_angle > 90.0)
				bailout("read_mod.c: must have 0.0 < cutoff_angle <= 90.0 deg\n");
			mod->photo.radar[ilaw].quasispec.cos_cutoff = cos( cutoff_angle*D2R);
		}
		else if (!strcmp( str, "gauss+cosine")) {
			mod->photo.radtype[ilaw] = GAUSSIAN_COSINE;
			if (readparam( fp, &mod->photo.radar[ilaw].hybrid.qs.R))
				++nfpar;
			if (par->action == FIT && par->vary_radalb
					&& mod->photo.radar[ilaw].hybrid.qs.R.state == 'c'
					&& mod->photo.radar[ilaw].hybrid.qs.R.val != 0.0)
				bailout("read_mod.c: must let nonzero R_qs float if vary_radalb = yes\n");
			mod->photo.radar[ilaw].hybrid.qs.R_save = mod->photo.radar[ilaw].hybrid.qs.R.val;
			if (readparam( fp, &mod->photo.radar[ilaw].hybrid.qs.C))
				++nfpar;
			cutoff_angle = getdouble( fp);
			if (cutoff_angle <=  0.0 || cutoff_angle > 90.0)
				bailout("read_mod.c: must have 0.0 < cutoff_angle <= 90.0 deg\n");
			mod->photo.radar[ilaw].hybrid.qs.cos_cutoff = cos( cutoff_angle*D2R);
			if (readparam( fp, &mod->photo.radar[ilaw].hybrid.diff.R))
				++nfpar;
			if (par->action == FIT && par->vary_radalb
					&& mod->photo.radar[ilaw].hybrid.diff.R.state == 'c'
					&& mod->photo.radar[ilaw].hybrid.diff.R.val != 0.0)
				bailout("read_mod.c: must let nonzero R_diff float if vary_radalb = yes\n");
			mod->photo.radar[ilaw].hybrid.diff.R_save = mod->photo.radar[ilaw].hybrid.diff.R.val;
			if (readparam( fp, &mod->photo.radar[ilaw].hybrid.diff.C))
				++nfpar;
		}
		else if (!strcmp( str, "hagfors+cosine")) {
			mod->photo.radtype[ilaw] = HAGFORS_COSINE;
			if (readparam( fp, &mod->photo.radar[ilaw].hybrid.qs.R))
				++nfpar;
			if (par->action == FIT && par->vary_radalb
					&& mod->photo.radar[ilaw].hybrid.qs.R.state == 'c'
					&& mod->photo.radar[ilaw].hybrid.qs.R.val != 0.0)
				bailout("read_mod.c: must let nonzero R_qs float if vary_radalb = yes\n");
			mod->photo.radar[ilaw].hybrid.qs.R_save = mod->photo.radar[ilaw].hybrid.qs.R.val;
			if (readparam( fp, &mod->photo.radar[ilaw].hybrid.qs.C))
				++nfpar;
			cutoff_angle = getdouble( fp);
			if (cutoff_angle <=  0.0 || cutoff_angle > 90.0)
				bailout("read_mod.c: must have 0.0 < cutoff_angle <= 90.0 deg\n");
			mod->photo.radar[ilaw].hybrid.qs.cos_cutoff = cos( cutoff_angle*D2R);
			if (readparam( fp, &mod->photo.radar[ilaw].hybrid.diff.R))
				++nfpar;
			if (par->action == FIT && par->vary_radalb
					&& mod->photo.radar[ilaw].hybrid.diff.R.state == 'c'
					&& mod->photo.radar[ilaw].hybrid.diff.R.val != 0.0)
				bailout("read_mod.c: must let nonzero R_diff float if vary_radalb = yes\n");
			mod->photo.radar[ilaw].hybrid.diff.R_save = mod->photo.radar[ilaw].hybrid.diff.R.val;
			if (readparam( fp, &mod->photo.radar[ilaw].hybrid.diff.C))
				++nfpar;
		}
		else if (!strcmp( str, "cosine+cosine")) {
			mod->photo.radtype[ilaw] = COSINE_COSINE;
			if (readparam( fp, &mod->photo.radar[ilaw].hybrid.qs.R))
				++nfpar;
			if (par->action == FIT && par->vary_radalb
					&& mod->photo.radar[ilaw].hybrid.qs.R.state == 'c'
					&& mod->photo.radar[ilaw].hybrid.qs.R.val != 0.0)
				bailout("read_mod.c: must let nonzero R_qs float if vary_radalb = yes\n");
			mod->photo.radar[ilaw].hybrid.qs.R_save = mod->photo.radar[ilaw].hybrid.qs.R.val;
			if (readparam( fp, &mod->photo.radar[ilaw].hybrid.qs.C))
				++nfpar;
			cutoff_angle = getdouble( fp);
			if (cutoff_angle <=  0.0 || cutoff_angle > 90.0)
				bailout("read_mod.c: must have 0.0 < cutoff_angle <= 90.0 deg\n");
			mod->photo.radar[ilaw].hybrid.qs.cos_cutoff = cos( cutoff_angle*D2R);
			if (readparam( fp, &mod->photo.radar[ilaw].hybrid.diff.R))
				++nfpar;
			if (par->action == FIT && par->vary_radalb
					&& mod->photo.radar[ilaw].hybrid.diff.R.state == 'c'
					&& mod->photo.radar[ilaw].hybrid.diff.R.val != 0.0)
				bailout("read_mod.c: must let nonzero R_diff float if vary_radalb = yes\n");
			mod->photo.radar[ilaw].hybrid.diff.R_save = mod->photo.radar[ilaw].hybrid.diff.R.val;
			if (readparam( fp, &mod->photo.radar[ilaw].hybrid.diff.C))
				++nfpar;
		}
		else if (!strcmp( str, "harmcosine")) {
			mod->photo.radtype[ilaw] = HARMCOSINE_DIFF;

			mod->photo.radar[ilaw].harmcosine.R.nhar = getint( fp);
			L = mod->photo.radar[ilaw].harmcosine.R.nhar;
			
			/*=======================================================================*/
			/* if gpu processing via CUDA is enabled, allocate via CUDA function.
			* If it is not enabled, allocate via the standard C call.				 */

			if (CUDA){
				cudaCalloc((void**)&mod->photo.radar[ilaw].harmcosine.R.a,
					sizeof(struct param_t *), L + 1);
				cudaCalloc((void**)&mod->photo.radar[ilaw].harmcosine.R.b,
					sizeof(struct param_t *), L + 1);
				cudaCalloc((void**)&mod->photo.radar[ilaw].harmcosine.R.a_save,
					sizeof(struct param_t *), L + 1);
				cudaCalloc((void**)&mod->photo.radar[ilaw].harmcosine.R.b_save,
					sizeof(struct param_t *), L + 1);
			}
			else {
				mod->photo.radar[ilaw].harmcosine.R.a = (struct param_t **)
					calloc(L + 1, sizeof(struct param_t *));
				mod->photo.radar[ilaw].harmcosine.R.b = (struct param_t **)
					calloc(L + 1, sizeof(struct param_t *));
				mod->photo.radar[ilaw].harmcosine.R.a_save = (double **)
					calloc(L + 1, sizeof(struct param_t *));
				mod->photo.radar[ilaw].harmcosine.R.b_save = (double **)
					calloc(L + 1, sizeof(struct param_t *));
			}
			/*=======================================================================*/
						
			for (l=0; l<=L; l++) {
				
				/*=======================================================================*/
				/* if gpu processing via CUDA is enabled, allocate via CUDA function.
				* If it is not enabled, allocate via the standard C call.				 */

				if (CUDA){
					cudaCalloc((void**)&mod->photo.radar[ilaw].harmcosine.R.a[l],
						sizeof(struct param_t), l + 1);
					cudaCalloc((void**)&mod->photo.radar[ilaw].harmcosine.R.b[l],
						sizeof(struct param_t), l + 1);
					cudaCalloc((void**)&mod->photo.radar[ilaw].harmcosine.R.a_save[l],
						sizeof(struct param_t), l + 1);
					cudaCalloc((void**)&mod->photo.radar[ilaw].harmcosine.R.b_save[l],
						sizeof(struct param_t), l + 1);
				}
				else {
					mod->photo.radar[ilaw].harmcosine.R.a[l] = (struct param_t *)
						calloc(l + 1, sizeof(struct param_t));
					mod->photo.radar[ilaw].harmcosine.R.b[l] = (struct param_t *)
						calloc(l + 1, sizeof(struct param_t));
					mod->photo.radar[ilaw].harmcosine.R.a_save[l] = (double *)
						calloc(l + 1, sizeof(struct param_t));
					mod->photo.radar[ilaw].harmcosine.R.b_save[l] = (double *)
						calloc(l + 1, sizeof(struct param_t));
				}
				/*=======================================================================*/																	
				
				for (m=0; m<=l; m++) {
					if (readparam( fp, &mod->photo.radar[ilaw].harmcosine.R.a[l][m]))
						++nfpar;
					if (par->action == FIT && par->vary_radalb
							&& mod->photo.radar[ilaw].harmcosine.R.a[l][m].state == 'c'
							&& mod->photo.radar[ilaw].harmcosine.R.a[l][m].val != 0.0)
						bailout("read_mod.c: must let all nonzero albedo coefficients float if vary_radalb = yes\n");
					mod->photo.radar[ilaw].harmcosine.R.a_save[l][m]
					                                                = mod->photo.radar[ilaw].harmcosine.R.a[l][m].val;
					if (m > 0) {
						if (readparam( fp, &mod->photo.radar[ilaw].harmcosine.R.b[l][m]))
							++nfpar;
						if (par->action == FIT && par->vary_radalb
								&& mod->photo.radar[ilaw].harmcosine.R.b[l][m].state == 'c'
										&& mod->photo.radar[ilaw].harmcosine.R.b[l][m].val != 0.0)
							bailout("read_mod.c: must let all nonzero albedo coefficients float if vary_radalb = yes\n");
						mod->photo.radar[ilaw].harmcosine.R.b_save[l][m]
						                                                = mod->photo.radar[ilaw].harmcosine.R.b[l][m].val;
					}
				}
			}
			mod->photo.radar[ilaw].harmcosine.R.ntheta = -1;  /* dummy value */

			mod->photo.radar[ilaw].harmcosine.C.nhar = getint( fp);
			L = mod->photo.radar[ilaw].harmcosine.C.nhar;
			
			
			/*=======================================================================*/
			/* if gpu processing via CUDA is enabled, allocate via CUDA function.
			* If it is not enabled, allocate via the standard C call.				 */

			if (CUDA){
				cudaCalloc((void**)&mod->photo.radar[ilaw].harmcosine.C.a,
					sizeof(struct param_t *), L + 1);
				cudaCalloc((void**)&mod->photo.radar[ilaw].harmcosine.C.b,
					sizeof(struct param_t *), L + 1);				
			}
			else {
				mod->photo.radar[ilaw].harmcosine.C.a = (struct param_t **)
                                        		  calloc( L+1, sizeof( struct param_t *));
				mod->photo.radar[ilaw].harmcosine.C.b = (struct param_t **)
                                        		  calloc( L+1, sizeof( struct param_t *));
			}
			/*=======================================================================*/
									
			for (l=0; l<=L; l++) {
				/*=======================================================================*/
				/* if gpu processing via CUDA is enabled, allocate via CUDA function.
				* If it is not enabled, allocate via the standard C call.				 */

				if (CUDA){
					cudaCalloc((void**)&mod->photo.radar[ilaw].harmcosine.C.a[l] ,
						sizeof(struct param_t), l + 1);
					cudaCalloc((void**)&mod->photo.radar[ilaw].harmcosine.C.b[l],
						sizeof(struct param_t), l + 1);
				}
				else {
					mod->photo.radar[ilaw].harmcosine.C.a[l] = (struct param_t *)
						calloc(l + 1, sizeof(struct param_t));
					mod->photo.radar[ilaw].harmcosine.C.b[l] = (struct param_t *)
						calloc(l + 1, sizeof(struct param_t));
				}
				/*=======================================================================*/
								
				for (m=0; m<=l; m++) {
					if (readparam( fp, &mod->photo.radar[ilaw].harmcosine.C.a[l][m]))
						++nfpar;
					if (m > 0)
						if (readparam( fp, &mod->photo.radar[ilaw].harmcosine.C.b[l][m]))
							++nfpar;
				}
			}
			mod->photo.radar[ilaw].harmcosine.C.ntheta = -1;  /* dummy value */

			nc = mod->shape.ncomp;
			/*=======================================================================*/
			if (CUDA)
				cudaCalloc((void**)&mod->photo.radar[ilaw].harmcosine.local,
					sizeof(struct RC_t *), nc);
			else
				mod->photo.radar[ilaw].harmcosine.local = (struct RC_t **)
					calloc( nc, sizeof( struct RC_t *));
			/*=======================================================================*/

			for (c=0; c<nc; c++) {
				nf = mod->shape.comp[c].real.nf;
				/*=======================================================================*/
				if (CUDA)
					cudaCalloc((void**)&mod->photo.radar[ilaw].harmcosine.local[c],
						sizeof(struct RC_t), nf);
				else				
					mod->photo.radar[ilaw].harmcosine.local[c] = (struct RC_t *)
						calloc( nf, sizeof( struct RC_t));
				/*=======================================================================*/
			}
		}
		else if (!strcmp( str, "inhocosine")) {
			mod->photo.radtype[ilaw] = INHOCOSINE_DIFF;
			if (readparam( fp, &mod->photo.radar[ilaw].inhocosine.global.R))
				++nfpar;
			if (mod->photo.radar[ilaw].inhocosine.global.R.state == 'c' &&
					mod->photo.radar[ilaw].inhocosine.global.R.val != 0.0)
				globalflag = 1;
			else
				globalflag = 0;
			mod->photo.radar[ilaw].inhocosine.global.R_save
					= mod->photo.radar[ilaw].inhocosine.global.R.val;
			if (readparam( fp, &mod->photo.radar[ilaw].inhocosine.global.C))
				++nfpar;

			nc = getint( fp);
			if (nc != mod->shape.ncomp) {
				printf("%d components for inhocosine != %d model components\n",
						nc, mod->shape.ncomp);
				bailout("read_mod.c\n");
			}
			/*=======================================================================*/
			if (CUDA)
				cudaCalloc((void**)&mod->photo.radar[ilaw].inhocosine.local,
					sizeof(struct RC_t *), nc);
			else
				mod->photo.radar[ilaw].inhocosine.local = (struct RC_t **)
					calloc(nc, sizeof(struct RC_t *));
			/*=======================================================================*/
						
			for (c=0; c<nc; c++) {
				nf = getint( fp);
				if (nf != mod->shape.comp[c].real.nf) {
					printf("%d facets for inhocosine component %d != %d facets\n",
							nf, c, mod->shape.comp[c].real.nf);
					printf("           for realization of model component %d\n", c);
					bailout("read_mod.c\n");
				}
				/*=======================================================================*/
				if (CUDA)
					cudaCalloc((void**)&mod->photo.radar[ilaw].inhocosine.local[c],
						sizeof(struct RC_t), nf);
				else
					mod->photo.radar[ilaw].inhocosine.local[c] = (struct RC_t *)
						calloc(nf, sizeof(struct RC_t));
				/*=======================================================================*/				
				
				for (f=0; f<nf; f++) {
					if (readparam( fp, &mod->photo.radar[ilaw].inhocosine.local[c][f].R))
						++nfpar;
					if (par->action == FIT && par->vary_radalb &&
							((mod->photo.radar[ilaw].inhocosine.local[c][f].R.state == 'c') ||
									(mod->photo.radar[ilaw].inhocosine.local[c][f].R.state == '=' && globalflag)))
						bailout("read_mod.c: must let R float for all facets if vary_radalb = yes\n");
					mod->photo.radar[ilaw].inhocosine.local[c][f].R_save
							= mod->photo.radar[ilaw].inhocosine.local[c][f].R.val;
					if (readparam( fp, &mod->photo.radar[ilaw].inhocosine.local[c][f].C))
						++nfpar;
				}
			}
		}
		else
			bailout("read_mod.c: invalid radar scattering law\n");
	}

	/*  Read # of optical scattering laws and allocate memory  */
	mod->photo.noptlaws = getint( fp);

	/*=======================================================================*/
	if (CUDA) {

		if (mod->photo.noptlaws == 0) {
			cudaCalloc((void**)&mod->photo.opttype, sizeof(unsigned char), 2);
		}
		else {
			cudaCalloc((void**)&mod->photo.opttype, sizeof(unsigned char), mod->photo.noptlaws);
			cudaCalloc((void**)&mod->photo.optical, sizeof(union optscat_t),
					mod->photo.noptlaws);
		}
	}
	else {
		mod->photo.opttype = ucvector( 0, mod->photo.noptlaws-1);
		mod->photo.optical = (union optscat_t*) calloc(mod->photo.noptlaws,
				sizeof(union optscat_t));
	}
	/*=======================================================================*/
		
	/*  Read each optical scattering law  */

	for (ilaw=0; ilaw<mod->photo.noptlaws; ilaw++) {
		gettstr( fp, str);
		printf("# optical scattering law %d: %s\n", ilaw, str);
		fflush(stdout);
		if (!strcmp( str, "none"))
			mod->photo.opttype[ilaw] = NOLAW;
		else if (!strcmp( str, "geometrical")) {
			mod->photo.opttype[ilaw] = GEOMETRICAL;
			if (readparam( fp, &mod->photo.optical[ilaw].R.R))
				++nfpar;
			if (par->action == FIT && par->vary_optalb && mod->photo.optical[ilaw].R.R.state == 'c')
				bailout("read_mod.c: must let R float if vary_optalb = yes\n");
			mod->photo.optical[ilaw].R.R_save = mod->photo.optical[ilaw].R.R.val;
		}
		else if (!strcmp( str, "lambertian")) {
			mod->photo.opttype[ilaw] = LAMBERTLAW;
			if (readparam( fp, &mod->photo.optical[ilaw].R.R))
				++nfpar;
			if (par->action == FIT && par->vary_optalb && mod->photo.optical[ilaw].R.R.state == 'c')
				bailout("read_mod.c: must let R float if vary_optalb = yes\n");
			mod->photo.optical[ilaw].R.R_save = mod->photo.optical[ilaw].R.R.val;
		}
		else if (!strcmp( str, "lommel")) {
			mod->photo.opttype[ilaw] = LOMMEL;
			if (readparam( fp, &mod->photo.optical[ilaw].R.R))
				++nfpar;
			if (par->action == FIT && par->vary_optalb && mod->photo.optical[ilaw].R.R.state == 'c')
				bailout("read_mod.c: must let R float if vary_optalb = yes\n");
			mod->photo.optical[ilaw].R.R_save = mod->photo.optical[ilaw].R.R.val;
		}
		else if (!strcmp( str, "harmlambert") || !strcmp( str, "harmlommel")) {
			if (!strcmp( str, "harmlambert"))
				mod->photo.opttype[ilaw] = HARMLAMBERT;
			else
				mod->photo.opttype[ilaw] = HARMLOMMEL;

			mod->photo.optical[ilaw].harmR.R.nhar = getint( fp);
			L = mod->photo.optical[ilaw].harmR.R.nhar;
			
			/*=======================================================================*/
			if (CUDA) {
				cudaCalloc((void**)&mod->photo.optical[ilaw].harmR.R.a,
					sizeof(struct param_t *),	L + 1);
				cudaCalloc((void**)&mod->photo.optical[ilaw].harmR.R.b,
					sizeof(struct param_t *), L + 1);
				cudaCalloc((void**)&mod->photo.optical[ilaw].harmR.R.a_save,
					sizeof(struct param_t *), L + 1);
				cudaCalloc((void**)&mod->photo.optical[ilaw].harmR.R.b_save,
					sizeof(struct param_t *), L + 1);
			}
			else {
				mod->photo.optical[ilaw].harmR.R.a = (struct param_t **)
					calloc(L + 1, sizeof(struct param_t *));
				mod->photo.optical[ilaw].harmR.R.b = (struct param_t **)
					calloc(L + 1, sizeof(struct param_t *));
				mod->photo.optical[ilaw].harmR.R.a_save = (double **)
					calloc(L + 1, sizeof(struct param_t *));
				mod->photo.optical[ilaw].harmR.R.b_save = (double **)
					calloc(L + 1, sizeof(struct param_t *));
			}
			/*=======================================================================*/
						
			for (l=0; l<=L; l++) {
				
				/*=======================================================================*/
				if (CUDA) {
					cudaCalloc((void**)&mod->photo.optical[ilaw].harmR.R.a[l] ,
						sizeof(struct param_t), l + 1);
					cudaCalloc((void**)&mod->photo.optical[ilaw].harmR.R.b[l],
						sizeof(struct param_t), l + 1);
					cudaCalloc((void**)&mod->photo.optical[ilaw].harmR.R.a_save[l],
						sizeof(struct param_t), l + 1);
					cudaCalloc((void**)&mod->photo.optical[ilaw].harmR.R.b_save[l],
						sizeof(struct param_t), l + 1);
				}
				else {
					mod->photo.optical[ilaw].harmR.R.a[l] = (struct param_t *)
						calloc(l + 1, sizeof(struct param_t));
					mod->photo.optical[ilaw].harmR.R.b[l] = (struct param_t *)
						calloc(l + 1, sizeof(struct param_t));
					mod->photo.optical[ilaw].harmR.R.a_save[l] = (double *)
						calloc(l + 1, sizeof(struct param_t));
					mod->photo.optical[ilaw].harmR.R.b_save[l] = (double *)
						calloc(l + 1, sizeof(struct param_t));
				}
				/*=======================================================================*/
								
				for (m=0; m<=l; m++) {
					if (readparam( fp, &mod->photo.optical[ilaw].harmR.R.a[l][m]))
						++nfpar;
					if (par->action == FIT && par->vary_optalb
							&& mod->photo.optical[ilaw].harmR.R.a[l][m].state == 'c'
							&& mod->photo.optical[ilaw].harmR.R.a[l][m].val != 0.0)
						bailout("read_mod.c: must let all nonzero albedo coefficients float if vary_optalb = yes\n");
					mod->photo.optical[ilaw].harmR.R.a_save[l][m]
					                                             = mod->photo.optical[ilaw].harmR.R.a[l][m].val;
					if (m > 0) {
						if (readparam( fp, &mod->photo.optical[ilaw].harmR.R.b[l][m]))
							++nfpar;
						if (par->action == FIT && par->vary_optalb
								&& mod->photo.optical[ilaw].harmR.R.b[l][m].state == 'c'
										&& mod->photo.optical[ilaw].harmR.R.b[l][m].val != 0.0)
							bailout("read_mod.c: must let all nonzero albedo coefficients float if vary_optalb = yes\n");
						mod->photo.optical[ilaw].harmR.R.b_save[l][m]
						                                             = mod->photo.optical[ilaw].harmR.R.b[l][m].val;
					}
				}
			}
			mod->photo.optical[ilaw].harmR.R.ntheta = -1;  /* dummy value */

			nc = mod->shape.ncomp;
			
			/*=======================================================================*/
			if (CUDA) 
				cudaCalloc((void**)&mod->photo.optical[ilaw].harmR.local,
					sizeof(struct R_t *), nc);				
			else 
				mod->photo.optical[ilaw].harmR.local = (struct R_t **)
					calloc(nc, sizeof(struct R_t *));
			/*=======================================================================*/
						
			for (c=0; c<nc; c++) {
				nf = mod->shape.comp[c].real.nf;
				/*=======================================================================*/
				if (CUDA)
					cudaCalloc((void**)&mod->photo.optical[ilaw].harmR.local[c],
						sizeof(struct R_t), nf);
				else
					mod->photo.optical[ilaw].harmR.local[c] = (struct R_t *)
						calloc( nf, sizeof( struct R_t));
				/*=======================================================================*/
			}
		}
		else if (!strcmp( str, "inholambert") || !strcmp( str, "inholommel")) {
			if (!strcmp( str, "inholambert"))
				mod->photo.opttype[ilaw] = INHOLAMBERT;
			else
				mod->photo.opttype[ilaw] = INHOLOMMEL;
			if (readparam( fp, &mod->photo.optical[ilaw].inhoR.global.R))
				++nfpar;
			if (mod->photo.optical[ilaw].inhoR.global.R.state == 'c' &&
					mod->photo.optical[ilaw].inhoR.global.R.val != 0.0)
				globalflag = 1;
			else
				globalflag = 0;
			mod->photo.optical[ilaw].inhoR.global.R_save
					= mod->photo.optical[ilaw].inhoR.global.R.val;
			nc = getint( fp);
			if (nc != mod->shape.ncomp) {
				printf("%d components for %s != %d model components\n",
						nc, str, mod->shape.ncomp);
				bailout("read_mod.c\n");
			}
			
			/*=======================================================================*/
			if (CUDA)
				cudaCalloc((void**)&mod->photo.optical[ilaw].inhoR.local,
				sizeof(struct R_t *), nc);
			else
				mod->photo.optical[ilaw].inhoR.local = (struct R_t **)
					calloc(nc, sizeof(struct R_t *));
			/*=======================================================================*/
					
			for (c=0; c<nc; c++) {
				nf = getint( fp);
				if (nf != mod->shape.comp[c].real.nf) {
					printf("%d facets for %s component %d != %d facets\n",
							nf, str, c, mod->shape.comp[c].real.nf);
					printf("           for realization of model component %d\n", c);
					bailout("read_mod.c\n");
				}

				/*=======================================================================*/
				if (CUDA)
					cudaCalloc((void**)&mod->photo.optical[ilaw].inhoR.local[c],
					sizeof(struct R_t), nf);
				else
					mod->photo.optical[ilaw].inhoR.local[c] = (struct R_t *)
						calloc(nf, sizeof(struct R_t));
				/*=======================================================================*/
								
				for (f=0; f<nf; f++) {
					if (readparam( fp, &mod->photo.optical[ilaw].inhoR.local[c][f].R))
						++nfpar;
					if (par->action == FIT && par->vary_optalb &&
							((mod->photo.optical[ilaw].inhoR.local[c][f].R.state == 'c') ||
									(mod->photo.optical[ilaw].inhoR.local[c][f].R.state == '=' && globalflag)))
						bailout("read_mod.c: must let R float for all facets if vary_optalb = yes\n");
					mod->photo.optical[ilaw].inhoR.local[c][f].R_save
							= mod->photo.optical[ilaw].inhoR.local[c][f].R.val;
				}
			}
		}
		else if (!strcmp( str, "hapke")) {
			mod->photo.opttype[ilaw] = HAPKE;
			if (readparam( fp, &mod->photo.optical[ilaw].hapke.w))
				++nfpar;
			if (par->action == FIT && par->vary_optalb
					&& mod->photo.optical[ilaw].hapke.w.state == 'c')
				bailout("read_mod.c: must let w float if vary_optalb = yes\n");
			mod->photo.optical[ilaw].hapke.w_save = mod->photo.optical[ilaw].hapke.w.val;
			if (readparam( fp, &mod->photo.optical[ilaw].hapke.h))
				++nfpar;
			if (readparam( fp, &mod->photo.optical[ilaw].hapke.B0))
				++nfpar;
			if (readparam( fp, &mod->photo.optical[ilaw].hapke.g))
				++nfpar;
			if (readparam( fp, &mod->photo.optical[ilaw].hapke.theta))
				++nfpar;
			mod->photo.optical[ilaw].hapke.theta.val *= D2R;
		}
		else if (!strcmp( str, "harmhapke")) {
			mod->photo.opttype[ilaw] = HARMHAPKE;

			mod->photo.optical[ilaw].harmhapke.w.nhar = getint( fp);
			L = mod->photo.optical[ilaw].harmhapke.w.nhar;
			
			/*=======================================================================*/
			if (CUDA) {
				cudaCalloc((void**)&mod->photo.optical[ilaw].harmhapke.w.a,
					sizeof(struct param_t *), L + 1);
				cudaCalloc((void**)&mod->photo.optical[ilaw].harmhapke.w.b,
					sizeof(struct param_t *), L + 1);
				cudaCalloc((void**)&mod->photo.optical[ilaw].harmhapke.w.a_save,
					sizeof(struct param_t *), L + 1);
				cudaCalloc((void**)&mod->photo.optical[ilaw].harmhapke.w.b_save,
					sizeof(struct param_t *), L + 1);
			}
			else {
				mod->photo.optical[ilaw].harmhapke.w.a = (struct param_t **)
					calloc( L+1, sizeof( struct param_t *));
				mod->photo.optical[ilaw].harmhapke.w.b = (struct param_t **)
					calloc( L+1, sizeof( struct param_t *));
				mod->photo.optical[ilaw].harmhapke.w.a_save = (double **)
                    calloc( L+1, sizeof( struct param_t *));
				mod->photo.optical[ilaw].harmhapke.w.b_save = (double **)
                    calloc( L+1, sizeof( struct param_t *));
			}
			/*=======================================================================*/
							
			for (l=0; l<=L; l++) {
				/*=======================================================================*/
				if (CUDA) {
					cudaCalloc((void**)&mod->photo.optical[ilaw].harmhapke.w.a[l],
						sizeof(struct param_t), l + 1);
					cudaCalloc((void**)&mod->photo.optical[ilaw].harmhapke.w.b[l],
						sizeof(struct param_t), l + 1);
					cudaCalloc((void**)&mod->photo.optical[ilaw].harmhapke.w.a_save[l],
						sizeof(struct param_t), l + 1);
					cudaCalloc((void**)&mod->photo.optical[ilaw].harmhapke.w.b_save[l],
						sizeof(struct param_t), l + 1);
				}
				else {
					mod->photo.optical[ilaw].harmhapke.w.a[l] = (struct param_t *)
						calloc( l+1, sizeof( struct param_t));
					mod->photo.optical[ilaw].harmhapke.w.b[l] = (struct param_t *)
						calloc( l+1, sizeof( struct param_t));
					mod->photo.optical[ilaw].harmhapke.w.a_save[l] = (double *)
						calloc( l+1, sizeof( struct param_t));
					mod->photo.optical[ilaw].harmhapke.w.b_save[l] = (double *)
						calloc( l+1, sizeof( struct param_t));
				}
				/*=======================================================================*/
								
				for (m=0; m<=l; m++) {
					if (readparam( fp, &mod->photo.optical[ilaw].harmhapke.w.a[l][m]))
						++nfpar;
					if (par->action == FIT && par->vary_optalb
							&& mod->photo.optical[ilaw].harmhapke.w.a[l][m].state == 'c'
							&& mod->photo.optical[ilaw].harmhapke.w.a[l][m].val != 0.0)
						bailout("read_mod.c: must let all nonzero albedo coefficients float if vary_optalb = yes\n");
					mod->photo.optical[ilaw].harmhapke.w.a_save[l][m]
					                                                 = mod->photo.optical[ilaw].harmhapke.w.a[l][m].val;
					if (m > 0) {
						if (readparam( fp, &mod->photo.optical[ilaw].harmhapke.w.b[l][m]))
							++nfpar;
						if (par->action == FIT && par->vary_optalb
								&& mod->photo.optical[ilaw].harmhapke.w.b[l][m].state == 'c'
										&& mod->photo.optical[ilaw].harmhapke.w.b[l][m].val != 0.0)
							bailout("read_mod.c: must let all nonzero albedo coefficients float if vary_optalb = yes\n");
						mod->photo.optical[ilaw].harmhapke.w.b_save[l][m]
						                                                 = mod->photo.optical[ilaw].harmhapke.w.b[l][m].val;
					}
				}
			}
			mod->photo.optical[ilaw].harmhapke.w.ntheta = -1;  /* dummy value */

			mod->photo.optical[ilaw].harmhapke.h.nhar = getint( fp);
			L = mod->photo.optical[ilaw].harmhapke.h.nhar;
			
			/*=======================================================================*/
			if (CUDA) {
				cudaCalloc((void**)&mod->photo.optical[ilaw].harmhapke.h.a,
					sizeof(struct param_t *), L + 1);
				cudaCalloc((void**)&mod->photo.optical[ilaw].harmhapke.h.b,
					sizeof(struct param_t *), L + 1);				
			}
			else {
				mod->photo.optical[ilaw].harmhapke.h.a = (struct param_t **)
					calloc( L+1, sizeof( struct param_t *));
				mod->photo.optical[ilaw].harmhapke.h.b = (struct param_t **)
					calloc( L+1, sizeof( struct param_t *));
			}
			/*=======================================================================*/
					
			for (l=0; l<=L; l++) {
				/*=======================================================================*/
				if (CUDA) {
					cudaCalloc((void**)&mod->photo.optical[ilaw].harmhapke.h.a[l],
						sizeof(struct param_t), l + 1);
					cudaCalloc((void**)&mod->photo.optical[ilaw].harmhapke.h.b[l],
						sizeof(struct param_t), l + 1);
				}
				else {
					mod->photo.optical[ilaw].harmhapke.h.a[l] = (struct param_t *)
						calloc( l+1, sizeof( struct param_t));
					mod->photo.optical[ilaw].harmhapke.h.b[l] = (struct param_t *)
						calloc( l+1, sizeof( struct param_t));
				}
				/*=======================================================================*/
								
				for (m=0; m<=l; m++) {
					if (readparam( fp, &mod->photo.optical[ilaw].harmhapke.h.a[l][m]))
						++nfpar;
					if (m > 0)
						if (readparam( fp, &mod->photo.optical[ilaw].harmhapke.h.b[l][m]))
							++nfpar;
				}
			}
			mod->photo.optical[ilaw].harmhapke.h.ntheta = -1;  /* dummy value */

			mod->photo.optical[ilaw].harmhapke.B0.nhar = getint( fp);
			L = mod->photo.optical[ilaw].harmhapke.B0.nhar;
			
			/*=======================================================================*/
			if (CUDA) {
				cudaCalloc((void**)&mod->photo.optical[ilaw].harmhapke.B0.a,
					sizeof(struct param_t *), L + 1);
				cudaCalloc((void**)&mod->photo.optical[ilaw].harmhapke.B0.b,
					sizeof(struct param_t *), L + 1);
			}
			else {
				mod->photo.optical[ilaw].harmhapke.B0.a = (struct param_t **)
					calloc( L+1, sizeof( struct param_t *));
				mod->photo.optical[ilaw].harmhapke.B0.b = (struct param_t **)
					calloc( L+1, sizeof( struct param_t *));
			}
			/*=======================================================================*/
	
			for (l=0; l<=L; l++) {
				/*=======================================================================*/
				if (CUDA) {
					cudaCalloc((void**)&mod->photo.optical[ilaw].harmhapke.B0.a[l],
						sizeof(struct param_t), l + 1);
					cudaCalloc((void**)&mod->photo.optical[ilaw].harmhapke.B0.b[l],
						sizeof(struct param_t), l + 1);
				}
				else {
					mod->photo.optical[ilaw].harmhapke.B0.a[l] = (struct param_t *)
						calloc( l+1, sizeof( struct param_t));
					mod->photo.optical[ilaw].harmhapke.B0.b[l] = (struct param_t *)
						calloc( l+1, sizeof( struct param_t));					
				}
				/*=======================================================================*/
				
				for (m=0; m<=l; m++) {
					if (readparam( fp, &mod->photo.optical[ilaw].harmhapke.B0.a[l][m]))
						++nfpar;
					if (m > 0)
						if (readparam( fp, &mod->photo.optical[ilaw].harmhapke.B0.b[l][m]))
							++nfpar;
				}
			}
			mod->photo.optical[ilaw].harmhapke.B0.ntheta = -1;  /* dummy value */

			mod->photo.optical[ilaw].harmhapke.g.nhar = getint( fp);
			L = mod->photo.optical[ilaw].harmhapke.g.nhar;
			
			/*=======================================================================*/
			if (CUDA) {
				cudaCalloc((void**)&mod->photo.optical[ilaw].harmhapke.g.a,
					sizeof(struct param_t *), L + 1);
				cudaCalloc((void**)&mod->photo.optical[ilaw].harmhapke.g.b,
					sizeof(struct param_t *), L + 1);
			}
			else {
				mod->photo.optical[ilaw].harmhapke.g.a = (struct param_t **)
					calloc( L+1, sizeof( struct param_t *));
				mod->photo.optical[ilaw].harmhapke.g.b = (struct param_t **)
					calloc( L+1, sizeof( struct param_t *));
			}
			/*=======================================================================*/
						
			for (l=0; l<=L; l++) {
				/*=======================================================================*/
				if (CUDA) {
					cudaCalloc((void**)&mod->photo.optical[ilaw].harmhapke.g.a[l],
						sizeof(struct param_t), l + 1);
					cudaCalloc((void**)&mod->photo.optical[ilaw].harmhapke.g.b[l],
						sizeof(struct param_t), l + 1);
				}
				else {
					mod->photo.optical[ilaw].harmhapke.g.a[l] = (struct param_t *)
						calloc( l+1, sizeof( struct param_t));
					mod->photo.optical[ilaw].harmhapke.g.b[l] = (struct param_t *)
						calloc( l+1, sizeof( struct param_t));
				}
				/*=======================================================================*/
				
				for (m=0; m<=l; m++) {
					if (readparam( fp, &mod->photo.optical[ilaw].harmhapke.g.a[l][m]))
						++nfpar;
					if (m > 0)
						if (readparam( fp, &mod->photo.optical[ilaw].harmhapke.g.b[l][m]))
							++nfpar;
				}
			}
			mod->photo.optical[ilaw].harmhapke.g.ntheta = -1;  /* dummy value */

			mod->photo.optical[ilaw].harmhapke.theta.nhar = getint( fp);
			L = mod->photo.optical[ilaw].harmhapke.theta.nhar;
			
			/*=======================================================================*/
			if (CUDA) {
				cudaCalloc((void**)&mod->photo.optical[ilaw].harmhapke.theta.a,
					sizeof(struct param_t *), L + 1);
				cudaCalloc((void**)&mod->photo.optical[ilaw].harmhapke.theta.b,
					sizeof(struct param_t *), L + 1);
			}
			else {
				mod->photo.optical[ilaw].harmhapke.theta.a = (struct param_t **)
					calloc( L+1, sizeof( struct param_t *));
				mod->photo.optical[ilaw].harmhapke.theta.b = (struct param_t **)
					calloc( L+1, sizeof( struct param_t *));
			}
			/*=======================================================================*/
						
			for (l=0; l<=L; l++) {
				/*=======================================================================*/
				if (CUDA) {
					cudaCalloc((void**)&mod->photo.optical[ilaw].harmhapke.theta.a[l],
						sizeof(struct param_t), l + 1);
					cudaCalloc((void**)&mod->photo.optical[ilaw].harmhapke.theta.b[l],
						sizeof(struct param_t), l + 1);
				}
				else {
					mod->photo.optical[ilaw].harmhapke.theta.a[l] = (struct param_t *)
						calloc( l+1, sizeof( struct param_t));
					mod->photo.optical[ilaw].harmhapke.theta.b[l] = (struct param_t *)
						calloc( l+1, sizeof( struct param_t));
				}
				/*=======================================================================*/
								
				for (m=0; m<=l; m++) {
					if (readparam( fp, &mod->photo.optical[ilaw].harmhapke.theta.a[l][m]))
						++nfpar;
					mod->photo.optical[ilaw].harmhapke.theta.a[l][m].val *= D2R;
					if (m > 0) {
						if (readparam( fp, &mod->photo.optical[ilaw].harmhapke.theta.b[l][m]))
							++nfpar;
						mod->photo.optical[ilaw].harmhapke.theta.b[l][m].val *= D2R;
					}
				}
			}
			mod->photo.optical[ilaw].harmhapke.theta.ntheta = -1;  /* dummy value */

			nc = mod->shape.ncomp;
			/*=======================================================================*/
			if (CUDA) 
				cudaCalloc((void**)&mod->photo.optical[ilaw].harmhapke.local,
					sizeof(struct hapke_t *), nc);
			else 
				mod->photo.optical[ilaw].harmhapke.local = (struct hapke_t **)
					calloc(nc, sizeof(struct hapke_t *));
			/*=======================================================================*/
			
			for (c=0; c<nc; c++) {
				nf = mod->shape.comp[c].real.nf;
				/*=======================================================================*/
				if (CUDA)
					cudaCalloc((void**)&mod->photo.optical[ilaw].harmhapke.local[c],
						sizeof(struct hapke_t), nf);
				else
					mod->photo.optical[ilaw].harmhapke.local[c] = (struct hapke_t *)
						calloc(nf, sizeof(struct hapke_t));
				/*=======================================================================*/				
			}
		}
		else if (!strcmp( str, "inhohapke")) {
			mod->photo.opttype[ilaw] = INHOHAPKE;
			if (readparam( fp, &mod->photo.optical[ilaw].inhohapke.global.w))
				++nfpar;
			if (mod->photo.optical[ilaw].inhohapke.global.w.state == 'c' &&
					mod->photo.optical[ilaw].inhohapke.global.w.val != 0.0)
				globalflag = 1;
			else
				globalflag = 0;
			mod->photo.optical[ilaw].inhohapke.global.w_save
					= mod->photo.optical[ilaw].inhohapke.global.w.val;
			if (readparam( fp, &mod->photo.optical[ilaw].inhohapke.global.h))
				++nfpar;
			if (readparam( fp, &mod->photo.optical[ilaw].inhohapke.global.B0))
				++nfpar;
			if (readparam( fp, &mod->photo.optical[ilaw].inhohapke.global.g))
				++nfpar;
			if (readparam( fp, &mod->photo.optical[ilaw].inhohapke.global.theta))
				++nfpar;
			mod->photo.optical[ilaw].inhohapke.global.theta.val *= D2R;

			nc = getint( fp);
			if (nc != mod->shape.ncomp) {
				printf("%d components for inhohapke != %d model components\n",
						nc, mod->shape.ncomp);
				bailout("read_mod.c\n");
			}
			
			/*=======================================================================*/
			if (CUDA)
				cudaCalloc((void**)&mod->photo.optical[ilaw].inhohapke.local,
				sizeof(struct hapke_t *), nc);
			else
				mod->photo.optical[ilaw].inhohapke.local = (struct hapke_t **)
					calloc(nc, sizeof(struct hapke_t *));
			/*=======================================================================*/
						
			for (c=0; c<nc; c++) {
				nf = getint( fp);
				if (nf != mod->shape.comp[c].real.nf) {
					printf("%d facets for inhohapke component %d != %d facets\n",
							nf, c, mod->shape.comp[c].real.nf);
					printf("           for realization of model component %d\n", c);
					bailout("read_mod.c\n");
				}
				/*=======================================================================*/
				if (CUDA)
					cudaCalloc((void**)&mod->photo.optical[ilaw].inhohapke.local[c],
					sizeof(struct hapke_t), nf);
				else
					mod->photo.optical[ilaw].inhohapke.local[c] = (struct hapke_t *)
						calloc(nf, sizeof(struct hapke_t));
				/*=======================================================================*/
								
				for (f=0; f<nf; f++) {
					if (readparam( fp, &mod->photo.optical[ilaw].inhohapke.local[c][f].w))
						++nfpar;
					if (par->action == FIT && par->vary_optalb &&
							((mod->photo.optical[ilaw].inhohapke.local[c][f].w.state == 'c') ||
									(mod->photo.optical[ilaw].inhohapke.local[c][f].w.state == '=' && globalflag)))
						bailout("read_mod.c: must let w float for all facets if vary_optalb = yes\n");
					mod->photo.optical[ilaw].inhohapke.local[c][f].w_save
							= mod->photo.optical[ilaw].inhohapke.local[c][f].w.val;
					if (readparam( fp, &mod->photo.optical[ilaw].inhohapke.local[c][f].h))
						++nfpar;
					if (readparam( fp, &mod->photo.optical[ilaw].inhohapke.local[c][f].B0))
						++nfpar;
					if (readparam( fp, &mod->photo.optical[ilaw].inhohapke.local[c][f].g))
						++nfpar;
					if (readparam( fp, &mod->photo.optical[ilaw].inhohapke.local[c][f].theta))
						++nfpar;
					mod->photo.optical[ilaw].inhohapke.local[c][f].theta.val *= D2R;
				}
			}
		}
		else if (!strcmp( str, "kaasalainen")) {
			mod->photo.opttype[ilaw] = KAASALAINEN;
			if (readparam( fp, &mod->photo.optical[ilaw].kaas.R))
				++nfpar;
			if (par->action == FIT && par->vary_optalb
					&& mod->photo.optical[ilaw].kaas.R.state == 'c')
				bailout("read_mod.c: must let R float if vary_optalb = yes\n");
			mod->photo.optical[ilaw].kaas.R_save = mod->photo.optical[ilaw].kaas.R.val;
			if (readparam( fp, &mod->photo.optical[ilaw].kaas.wt))
				++nfpar;
			if (readparam( fp, &mod->photo.optical[ilaw].kaas.A0))
				++nfpar;
			if (readparam( fp, &mod->photo.optical[ilaw].kaas.D))
				++nfpar;
			if (readparam( fp, &mod->photo.optical[ilaw].kaas.k))
				++nfpar;
			mod->photo.optical[ilaw].kaas.D.val *= D2R;
			mod->photo.optical[ilaw].kaas.k.val /= D2R;
		}
		else if (!strcmp( str, "harmkaas")) {
			mod->photo.opttype[ilaw] = HARMKAAS;

			mod->photo.optical[ilaw].harmkaas.R.nhar = getint( fp);
			L = mod->photo.optical[ilaw].harmkaas.R.nhar;
			
			/*=======================================================================*/
			if (CUDA){
				cudaCalloc((void**)&mod->photo.optical[ilaw].harmkaas.R.a,
					sizeof(struct param_t *), L + 1);
				cudaCalloc((void**)&mod->photo.optical[ilaw].harmkaas.R.b,
					sizeof(struct param_t *), L + 1);
				cudaCalloc((void**)&mod->photo.optical[ilaw].harmkaas.R.a_save,
					sizeof(struct param_t *), L + 1);
				cudaCalloc((void**)&mod->photo.optical[ilaw].harmkaas.R.a_save,
					sizeof(struct param_t *), L + 1);
			}
			else {
				mod->photo.optical[ilaw].harmkaas.R.a = (struct param_t **)
					calloc(L + 1, sizeof(struct param_t *));
				mod->photo.optical[ilaw].harmkaas.R.b = (struct param_t **)
					calloc(L + 1, sizeof(struct param_t *));
				mod->photo.optical[ilaw].harmkaas.R.a_save = (double **)
					calloc(L + 1, sizeof(struct param_t *));
				mod->photo.optical[ilaw].harmkaas.R.b_save = (double **)
					calloc(L + 1, sizeof(struct param_t *));
			}
			/*=======================================================================*/
					
			for (l=0; l<=L; l++) {
				/*=======================================================================*/
				if (CUDA){
					cudaCalloc((void**)&mod->photo.optical[ilaw].harmkaas.R.a[l],
						sizeof(struct param_t), l + 1);
					cudaCalloc((void**)&mod->photo.optical[ilaw].harmkaas.R.b[l],
						sizeof(struct param_t), l + 1);
					cudaCalloc((void**)&mod->photo.optical[ilaw].harmkaas.R.a_save[l],
						sizeof(struct param_t), l + 1);
					cudaCalloc((void**)&mod->photo.optical[ilaw].harmkaas.R.a_save[l],
						sizeof(struct param_t), l + 1);
				}
				else {
					mod->photo.optical[ilaw].harmkaas.R.a[l] = (struct param_t *)
						calloc(l + 1, sizeof(struct param_t));
					mod->photo.optical[ilaw].harmkaas.R.b[l] = (struct param_t *)
						calloc(l + 1, sizeof(struct param_t));
					mod->photo.optical[ilaw].harmkaas.R.a_save[l] = (double *)
						calloc(l + 1, sizeof(struct param_t));
					mod->photo.optical[ilaw].harmkaas.R.b_save[l] = (double *)
						calloc(l + 1, sizeof(struct param_t));
				}
				/*=======================================================================*/
							
				for (m=0; m<=l; m++) {
					if (readparam( fp, &mod->photo.optical[ilaw].harmkaas.R.a[l][m]))
						++nfpar;
					if (par->action == FIT && par->vary_optalb
							&& mod->photo.optical[ilaw].harmkaas.R.a[l][m].state == 'c'
							&& mod->photo.optical[ilaw].harmkaas.R.a[l][m].val != 0.0)
						bailout("read_mod.c: must let all nonzero albedo coefficients float if vary_optalb = yes\n");
					mod->photo.optical[ilaw].harmkaas.R.a_save[l][m]
					                                                = mod->photo.optical[ilaw].harmkaas.R.a[l][m].val;
					if (m > 0) {
						if (readparam( fp, &mod->photo.optical[ilaw].harmkaas.R.b[l][m]))
							++nfpar;
						if (par->action == FIT && par->vary_optalb
								&& mod->photo.optical[ilaw].harmkaas.R.b[l][m].state == 'c'
										&& mod->photo.optical[ilaw].harmkaas.R.b[l][m].val != 0.0)
							bailout("read_mod.c: must let all nonzero albedo coefficients float if vary_optalb = yes\n");
						mod->photo.optical[ilaw].harmkaas.R.b_save[l][m]
						                                                = mod->photo.optical[ilaw].harmkaas.R.b[l][m].val;
					}
				}
			}
			mod->photo.optical[ilaw].harmkaas.R.ntheta = -1;  /* dummy value */

			mod->photo.optical[ilaw].harmkaas.wt.nhar = getint( fp);
			L = mod->photo.optical[ilaw].harmkaas.wt.nhar;
			
			/*=======================================================================*/
			if (CUDA){
				cudaCalloc((void**)&mod->photo.optical[ilaw].harmkaas.wt.a,
					sizeof(struct param_t *), L + 1);
				cudaCalloc((void**)&mod->photo.optical[ilaw].harmkaas.wt.b,
					sizeof(struct param_t *), L + 1);				
			}
			else {
				mod->photo.optical[ilaw].harmkaas.wt.a = (struct param_t **)
					calloc(L + 1, sizeof(struct param_t *));
				mod->photo.optical[ilaw].harmkaas.wt.b = (struct param_t **)
					calloc(L + 1, sizeof(struct param_t *));
			}
			/*=======================================================================*/
					
			for (l=0; l<=L; l++) {
				/*=======================================================================*/
				if (CUDA){
					cudaCalloc((void**)&mod->photo.optical[ilaw].harmkaas.wt.a[l],
						sizeof(struct param_t), l + 1);
					cudaCalloc((void**)&mod->photo.optical[ilaw].harmkaas.wt.b[l],
						sizeof(struct param_t), l + 1);
				}
				else {
					mod->photo.optical[ilaw].harmkaas.wt.a[l] = (struct param_t *)
						calloc(l + 1, sizeof(struct param_t));
					mod->photo.optical[ilaw].harmkaas.wt.b[l] = (struct param_t *)
						calloc(l + 1, sizeof(struct param_t));
				}
				/*=======================================================================*/
				
				for (m=0; m<=l; m++) {
					if (readparam( fp, &mod->photo.optical[ilaw].harmkaas.wt.a[l][m]))
						++nfpar;
					if (m > 0)
						if (readparam( fp, &mod->photo.optical[ilaw].harmkaas.wt.b[l][m]))
							++nfpar;
				}
			}
			mod->photo.optical[ilaw].harmkaas.wt.ntheta = -1;  /* dummy value */

			mod->photo.optical[ilaw].harmkaas.A0.nhar = getint( fp);
			L = mod->photo.optical[ilaw].harmkaas.A0.nhar;
			
			/*=======================================================================*/
			if (CUDA){
				cudaCalloc((void**)&mod->photo.optical[ilaw].harmkaas.A0.a,
					sizeof(struct param_t *), L + 1);
				cudaCalloc((void**)&mod->photo.optical[ilaw].harmkaas.A0.b,
					sizeof(struct param_t *), L + 1);
			}
			else {
				mod->photo.optical[ilaw].harmkaas.A0.a = (struct param_t **)
					calloc(L + 1, sizeof(struct param_t *));
				mod->photo.optical[ilaw].harmkaas.A0.b = (struct param_t **)
					calloc(L + 1, sizeof(struct param_t *));
			}
			/*=======================================================================*/
						
			for (l=0; l<=L; l++) {
				/*=======================================================================*/
				if (CUDA){
					cudaCalloc((void**)&mod->photo.optical[ilaw].harmkaas.A0.a[l],
						sizeof(struct param_t), l + 1);
					cudaCalloc((void**)&mod->photo.optical[ilaw].harmkaas.A0.b[l],
						sizeof(struct param_t), l + 1);
				}
				else {
					mod->photo.optical[ilaw].harmkaas.A0.a[l] = (struct param_t *)
						calloc(l + 1, sizeof(struct param_t));
					mod->photo.optical[ilaw].harmkaas.A0.b[l] = (struct param_t *)
						calloc(l + 1, sizeof(struct param_t));
				}
				/*=======================================================================*/
								
				for (m=0; m<=l; m++) {
					if (readparam( fp, &mod->photo.optical[ilaw].harmkaas.A0.a[l][m]))
						++nfpar;
					if (m > 0)
						if (readparam( fp, &mod->photo.optical[ilaw].harmkaas.A0.b[l][m]))
							++nfpar;
				}
			}
			mod->photo.optical[ilaw].harmkaas.A0.ntheta = -1;  /* dummy value */

			mod->photo.optical[ilaw].harmkaas.D.nhar = getint( fp);
			L = mod->photo.optical[ilaw].harmkaas.D.nhar;
			
			/*=======================================================================*/
			if (CUDA){
				cudaCalloc((void**)&mod->photo.optical[ilaw].harmkaas.D.a,
					sizeof(struct param_t *), L + 1);
				cudaCalloc((void**)&mod->photo.optical[ilaw].harmkaas.D.b,
					sizeof(struct param_t *), L + 1);
			}
			else {
				mod->photo.optical[ilaw].harmkaas.D.a = (struct param_t **)
					calloc(L + 1, sizeof(struct param_t *));
				mod->photo.optical[ilaw].harmkaas.D.b = (struct param_t **)
					calloc(L + 1, sizeof(struct param_t *));
			}
			/*=======================================================================*/
						
			for (l=0; l<=L; l++) {
				/*=======================================================================*/
				if (CUDA){
					cudaCalloc((void**)&mod->photo.optical[ilaw].harmkaas.D.a[l],
						sizeof(struct param_t), l + 1);
					cudaCalloc((void**)&mod->photo.optical[ilaw].harmkaas.D.b[l],
						sizeof(struct param_t), l + 1);
				}
				else {
					mod->photo.optical[ilaw].harmkaas.D.a[l] = (struct param_t *)
						calloc(l + 1, sizeof(struct param_t));
					mod->photo.optical[ilaw].harmkaas.D.b[l] = (struct param_t *)
						calloc(l + 1, sizeof(struct param_t));
				}
				/*=======================================================================*/
								
				for (m=0; m<=l; m++) {
					if (readparam( fp, &mod->photo.optical[ilaw].harmkaas.D.a[l][m]))
						++nfpar;
					mod->photo.optical[ilaw].harmkaas.D.a[l][m].val *= D2R;
					if (m > 0) {
						if (readparam( fp, &mod->photo.optical[ilaw].harmkaas.D.b[l][m]))
							++nfpar;
						mod->photo.optical[ilaw].harmkaas.D.b[l][m].val *= D2R;
					}
				}
			}
			mod->photo.optical[ilaw].harmkaas.D.ntheta = -1;  /* dummy value */

			mod->photo.optical[ilaw].harmkaas.k.nhar = getint( fp);
			L = mod->photo.optical[ilaw].harmkaas.k.nhar;
			
			/*=======================================================================*/
			if (CUDA){
				cudaCalloc((void**)&mod->photo.optical[ilaw].harmkaas.k.a,
					sizeof(struct param_t *), L + 1);
				cudaCalloc((void**)&mod->photo.optical[ilaw].harmkaas.k.b,
					sizeof(struct param_t *), L + 1);
			}
			else {
				mod->photo.optical[ilaw].harmkaas.k.a = (struct param_t **)
					calloc(L + 1, sizeof(struct param_t *));
				mod->photo.optical[ilaw].harmkaas.k.b = (struct param_t **)
					calloc(L + 1, sizeof(struct param_t *));
			}
			/*=======================================================================*/
			
			for (l=0; l<=L; l++) {
				/*=======================================================================*/
				if (CUDA){
					cudaCalloc((void**)&mod->photo.optical[ilaw].harmkaas.k.a[l],
						sizeof(struct param_t), l + 1);
					cudaCalloc((void**)&mod->photo.optical[ilaw].harmkaas.k.b[l],
						sizeof(struct param_t), l + 1);
				}
				else {
					mod->photo.optical[ilaw].harmkaas.k.a[l] = (struct param_t *)
						calloc(l + 1, sizeof(struct param_t));
					mod->photo.optical[ilaw].harmkaas.k.b[l] = (struct param_t *)
						calloc(l + 1, sizeof(struct param_t));
				}
				/*=======================================================================*/
								
				for (m=0; m<=l; m++) {
					if (readparam( fp, &mod->photo.optical[ilaw].harmkaas.k.a[l][m]))
						++nfpar;
					mod->photo.optical[ilaw].harmkaas.k.a[l][m].val /= D2R;
					if (m > 0) {
						if (readparam( fp, &mod->photo.optical[ilaw].harmkaas.k.b[l][m]))
							++nfpar;
						mod->photo.optical[ilaw].harmkaas.k.b[l][m].val /= D2R;
					}
				}
			}
			mod->photo.optical[ilaw].harmkaas.k.ntheta = -1;  /* dummy value */

			nc = mod->shape.ncomp;
			/*=======================================================================*/
			if (CUDA)
				cudaCalloc((void**)&mod->photo.optical[ilaw].harmkaas.local,
					sizeof(struct kaas_t *), nc);				
			else
				mod->photo.optical[ilaw].harmkaas.local = (struct kaas_t **)
					calloc(nc, sizeof(struct kaas_t *));
			/*=======================================================================*/
						
			for (c=0; c<nc; c++) {
				nf = mod->shape.comp[c].real.nf;
				/*=======================================================================*/
				if (CUDA)
					cudaCalloc((void**)&mod->photo.optical[ilaw].harmkaas.local[c],
					sizeof(struct kaas_t), nf);
				else
					mod->photo.optical[ilaw].harmkaas.local[c] = (struct kaas_t *)
						calloc(nf, sizeof(struct kaas_t));
				/*=======================================================================*/				
			}
		}
		else if (!strcmp( str, "inhokaas")) {
			mod->photo.opttype[ilaw] = INHOKAAS;
			if (readparam( fp, &mod->photo.optical[ilaw].inhokaas.global.R))
				++nfpar;
			if (mod->photo.optical[ilaw].inhokaas.global.R.state == 'c' &&
					mod->photo.optical[ilaw].inhokaas.global.R.val != 0.0)
				globalflag = 1;
			else
				globalflag = 0;
			mod->photo.optical[ilaw].inhokaas.global.R_save
					= mod->photo.optical[ilaw].inhokaas.global.R.val;
			if (readparam( fp, &mod->photo.optical[ilaw].inhokaas.global.wt))
				++nfpar;
			if (readparam( fp, &mod->photo.optical[ilaw].inhokaas.global.A0))
				++nfpar;
			if (readparam( fp, &mod->photo.optical[ilaw].inhokaas.global.D))
				++nfpar;
			if (readparam( fp, &mod->photo.optical[ilaw].inhokaas.global.k))
				++nfpar;
			mod->photo.optical[ilaw].inhokaas.global.D.val *= D2R;
			mod->photo.optical[ilaw].inhokaas.global.k.val /= D2R;

			nc = getint( fp);
			if (nc != mod->shape.ncomp) {
				printf("%d components for inhokaas != %d model components\n",
						nc, mod->shape.ncomp);
				bailout("read_mod.c\n");
			}
			
			/*=======================================================================*/
			if (CUDA)
				cudaCalloc((void**)&mod->photo.optical[ilaw].inhokaas.local,
				sizeof(struct kaas_t *), nc);
			else
				mod->photo.optical[ilaw].inhokaas.local = (struct kaas_t **)
					calloc(nc, sizeof(struct kaas_t *));
			/*=======================================================================*/
			
			for (c=0; c<nc; c++) {
				nf = getint( fp);
				if (nf != mod->shape.comp[c].real.nf) {
					printf("%d facets for inhokaas component %d != %d facets\n",
							nf, c, mod->shape.comp[c].real.nf);
					printf("           for realization of model component %d\n", c);
					bailout("read_mod.c\n");
				}
				/*=======================================================================*/
				if (CUDA)
					cudaCalloc((void**)&mod->photo.optical[ilaw].inhokaas.local[c],
					sizeof(struct kaas_t), nf);
				else
					mod->photo.optical[ilaw].inhokaas.local[c] = (struct kaas_t *)
						calloc(nf, sizeof(struct kaas_t));
				/*=======================================================================*/
								
				for (f=0; f<nf; f++) {
					if (readparam( fp, &mod->photo.optical[ilaw].inhokaas.local[c][f].R))
						++nfpar;
					if (par->action == FIT && par->vary_optalb &&
							((mod->photo.optical[ilaw].inhokaas.local[c][f].R.state == 'c') ||
									(mod->photo.optical[ilaw].inhokaas.local[c][f].R.state == '=' && globalflag)))
						bailout("read_mod.c: must let R float for all facets if vary_optalb = yes\n");
					mod->photo.optical[ilaw].inhokaas.local[c][f].R_save
							= mod->photo.optical[ilaw].inhokaas.local[c][f].R.val;
					if (readparam( fp, &mod->photo.optical[ilaw].inhokaas.local[c][f].wt))
						++nfpar;
					if (readparam( fp, &mod->photo.optical[ilaw].inhokaas.local[c][f].A0))
						++nfpar;
					if (readparam( fp, &mod->photo.optical[ilaw].inhokaas.local[c][f].D))
						++nfpar;
					if (readparam( fp, &mod->photo.optical[ilaw].inhokaas.local[c][f].k))
						++nfpar;
					mod->photo.optical[ilaw].inhokaas.local[c][f].D.val *= D2R;
					mod->photo.optical[ilaw].inhokaas.local[c][f].k.val /= D2R;
				}
			}
		}
		else
			bailout("read_mod.c: invalid optical scattering law\n");
	}

	return nfpar;
}

int read_spin( FILE *fp, struct mod_t *mod)
{
	char str[MAXLEN];
	int nfpar=0, i, cd[6], n;

	/*  First compute mod->spin.t0, the reference epoch (JD) for
      the angles, spin vector, and principal moments of inertia
      in the "spin" portion of the mod file.                      */

	for (i=0; i<=5; i++)
		cd[i] = getint( fp);
	cal2jd( cd[0], cd[1], cd[2], cd[3], cd[4], cd[5], &mod->spin.t0);
	printf("# initial JD of spin state: %f\n", mod->spin.t0);
	fflush(stdout);

	/*  Read in the angle and spin offsets (two sets of Euler angles)  */

	for (i=0; i<=2; i++)
		if (readparam( fp, &mod->spin.angle[i]))
			++nfpar;

	for (i=0; i<=2; i++)
		if (readparam( fp, &mod->spin.omega[i]))
			++nfpar;

	/*  Read the three principal moments of inertia -- the parameters that are used
      in integrating Euler's equations.  These numbers in fact are only PROPORTIONAL
      to the three principal moments: Euler's equations only depend on their RATIOS,
      so shape has no power to adjust their magnitude in an absolute sense.  If you
      increase all three starting values by a factor of 1000, and increase
      inertia_step by a factor of 1000, the three moments you obtain at the end of
      the fit will simply be 1000 times greater than they would have been otherwise.  */

	for (i=0; i<=2; i++)
		switch (readparam( fp, &mod->spin.inertia[i])) {
		case 1:
			++nfpar;
			break;
		case 0:
			break;
		case -1:
			fprintf( stderr, "ERROR: end of mod file reached, moments of inertia are missing\n");
			bailout("routine read_spin in read_mod.c\n");
			break;
		}

	/*  Read in change in spin rate parameters (YORP acceleration components)  */

	for (i=0; i<=2; i++)
		switch (readparam( fp, &mod->spin.omegadot[i])) {
		case 1:
			++nfpar;
			break;
		case 0:
			break;
		case -1:
			fprintf( stderr, "ERROR: end of mod file reached, YORP components are missing\n");
			bailout("routine read_spin in read_mod.c\n");
			break;
		}

	/* Read in parameters for libration.          */

	switch (readparam( fp, &mod->spin.lib_amp)) {
	case 1: ++nfpar;
	break;
	case 0: break;
	case -1:
		fprintf( stderr, "ERROR: end of mod file reached, libration parameter {libration amplitude} missing from mod file\n");
		bailout("routine read_spin in read_mod.c\n");
		break;
	}

	switch (readparam( fp, &mod->spin.lib_freq)) {
	case 1: ++nfpar;
	break;
	case 0: break;
	case -1:
		fprintf(stderr, "ERROR: end of mod file reached, libration parameter {libration frequency} missing from mod file\n");
		bailout("routine read_spin in read_mod.c\n");
		break;
	}

	switch (readparam( fp, &mod->spin.lib_phase)) {
	case 1: ++nfpar;
	break;
	case 0: break;
	case -1:
		fprintf( stderr, "ERROR: end of mod file reached, libration parameter {libration phase} missing from mod file\n");
		bailout("routine read_spin in read_mod.c\n");
		break;
	}

	/*  Read the number of spin impulses, the epochs of those impulses, and the impulses themselves  */

	if (gettstr( fp, str)) {
		mod->spin.n_impulse = string_to_int( str);
		if (mod->spin.n_impulse > MAXIMP) {
			fprintf( stderr, "Too many spin impulses, must increase MAXIMP\n");
			bailout("routine read_spin in read_mod.c\n");
		}
		for (n=0; n<mod->spin.n_impulse; n++) {
			for (i=0; i<=5; i++)
				cd[i] = getint( fp);
			cal2jd( cd[0], cd[1], cd[2], cd[3], cd[4], cd[5], &mod->spin.t_impulse[n]);
			if (n > 0 && mod->spin.t_impulse[n] <= mod->spin.t_impulse[n-1])
				bailout("read_mod.c: spin impulses must be given in increasing chronological order\n");
			for (i=0; i<=2; i++)
				if (readparam( fp, &mod->spin.impulse[n][i]))
					++nfpar;
		}
	} else {
		fprintf( stderr, "ERROR: end of mod file reached, number of spin impulses is missing\n");
		bailout("routine read_spin in read_mod.c\n");
	}
	for (n=mod->spin.n_impulse; n<MAXIMP; n++) {
		mod->spin.t_impulse[n] = -HUGENUMBER;
		for (i=0; i<=2; i++) {
			mod->spin.impulse[n][i].state = 'c';
			mod->spin.impulse[n][i].val = 0.0;
		}
	}

	/*  Iff the first two spin components are held fixed at zero, the first two YORP components
      are held fixed at zero, and the first two components of all spin impulses are held fixed
      at zero, flag this as a principal-axis rotator so that we don't have to waste time
      integrating Euler's equations to get the spin state at each epoch.  Note that this flag
      is NOT updated as a fit evolves.                                                          */

	mod->spin.pa = 0;
	if ( mod->spin.omega[0].val == 0.0 && mod->spin.omega[0].state == 'c' &&
			mod->spin.omega[1].val == 0.0 && mod->spin.omega[1].state == 'c'    ) {
		mod->spin.pa = 1;
		n = 0;
		while (n < mod->spin.n_impulse && mod->spin.pa) {
			if ( (mod->spin.impulse[n][0].val != 0.0) || (mod->spin.impulse[n][0].state == 'f') ||
					(mod->spin.impulse[n][1].val != 0.0) || (mod->spin.impulse[n][1].state == 'f')    )
				mod->spin.pa = 0;
			n++;
		}
	}

	if ( mod->spin.pa &&
			(mod->spin.omegadot[0].val != 0 || mod->spin.omegadot[0].state == 'f' ||
					mod->spin.omegadot[1].val != 0 || mod->spin.omegadot[1].state == 'f'    )) {
		fprintf( stderr, "ERROR: for PA rotator, YORP can only be applied to {spin 2}!\n");
		fprintf( stderr, "       -- set YORP paramaters {spin 0 dot} and {spin 1 dot} to c 0.000000\n");
		bailout("read_mod.c\n");
	} else if ( !mod->spin.pa &&
			(mod->spin.omegadot[0].val != 0 || mod->spin.omegadot[0].state == 'f' ||
					mod->spin.omegadot[1].val != 0 || mod->spin.omegadot[1].state == 'f' ||
					mod->spin.omegadot[2].val != 0 || mod->spin.omegadot[2].state == 'f' ||
					mod->spin.lib_amp.val != 0     || mod->spin.lib_amp.state == 'f')) {
		fprintf( stderr, "ERROR: cannot apply NPA rotation and YORP/Librations together at this time!\n");
		bailout("read_mod.c\n");
	}

	if (mod->spin.pa)
		printf("# assuming PA rotation\n");
	else
		printf("# assuming NPA rotation\n");
	fflush(stdout);

	/*  Deal with spin vectors that are "over the pole":
      |ecliptic pole latitude| > 90 deg

      To do this, shift the second Euler angle (= 90 - pole latitude)
      into the range [0, 360) deg; if it's now > 180, fold it around
      180 and add 180 to the first Euler angle (= 90 + pole longitude)
      and to the third Euler angle (orientation about rotation axis)    */

	mod->spin.angle[1].val -= 360.0*floor(mod->spin.angle[1].val/360.0);
	if (mod->spin.angle[1].val > 180.0) {
		mod->spin.angle[1].val = 360.0 - mod->spin.angle[1].val;
		mod->spin.angle[0].val += 180.0;
		mod->spin.angle[2].val += 180.0;
	}

	/*  Shift Euler angles into the range [0, 360) deg, then convert
      Euler angles to radians and spin vector components to rad/day  */

	for (i=0; i<=2; i++) {
		mod->spin.angle[i].val -= 360.0*floor(mod->spin.angle[i].val/360.0);
		mod->spin.angle[i].val *= D2R;
		mod->spin.omega[i].val *= D2R;
		mod->spin.omegadot[i].val *= D2R;
		for (n=0; n<mod->spin.n_impulse; n++)
			mod->spin.impulse[n][i].val *= D2R;
	}

	mod->spin.lib_amp.val *=D2R;
	mod->spin.lib_freq.val *=D2R;
	mod->spin.lib_phase.val *=D2R;

	return nfpar;
}

void setupreal( struct mod_t *mod)
{
	int c, nt, **iv, *np;

	/* Initialize variable to avoid compilation warning  */

	nt = 0;

	for (c=0; c<mod->shape.ncomp; c++) { /* for each component */
		if (mod->shape.comp[c].type == VERTEX) {
			mod->shape.comp[c].real = mod->shape.comp[c].desc.ver;
		} else {
			if (mod->shape.comp[c].type == ELLIPSE)
				nt = mod->shape.comp[c].desc.ell.ntheta;
			else if (mod->shape.comp[c].type == OVOID)
				nt = mod->shape.comp[c].desc.ovoid.ntheta;
			else
				nt = mod->shape.comp[c].desc.har.ntheta;
			np = ivector( 0, nt);
			iv = imatrix( 0, nt, 0, 2*nt);
			setuprealver( mod, c, iv, nt, np);
			setuprealfac( mod, c, iv, nt, np);
			free_imatrix( iv, 0, nt, 0, 2*nt);
			free_ivector( np, 0, nt);
		}
	}
}

void setuprealver( struct mod_t *mod, int c, int **iv, int nt, int *np)
{
	int t, p, nv=0, v=0, L=-1, l, m;
	double pi, theta, phi, costheta;
	double **nlm=NULL, **plm=NULL;

	/*  Figure out how many vertices are required  */

	pi = 4*atan(1.0);
	for (t=0; t<=nt; t++) {
		theta = (t*pi)/nt;
		np[t] = iround(2*nt*sin(theta));
		if (np[t] == 0)
			np[t] = 1;
		nv += np[t];
	}

	/*  For harmonic models, get spherical harmonic normalization coefficients  */

	if (mod->shape.comp[c].type == HARMONIC) {
		L = mod->shape.comp[c].desc.har.nhar;
		nlm = matrix( 0, L, 0, L);
		plm = matrix( 0, L, 0, L);
		for (l=0; l<=L; l++)
			for (m=0; m<=l; m++)
				nlm[l][m] = sqrt( (2*l+1) * exp(gammln(l-m+1.0) - gammln(l+m+1.0)) );
	}

	/*  Set up the vertices  */

	mod->shape.comp[c].real.nv = nv;
	
	/*=======================================================================*/
	if (CUDA)
		cudaCalloc((void**)&mod->shape.comp[c].real.v, 
			sizeof(struct vertex_t), nv);
	else
		mod->shape.comp[c].real.v = calloc(nv, sizeof(struct vertex_t));
	/*=======================================================================*/
		
	printf("# component %d has %d vertices in realization\n",
			c, mod->shape.comp[c].real.nv);
	fflush(stdout);

	v = 0;
	for (t=0; t<=nt; t++) {
		theta = (t*pi)/nt;
		for (p=0; p<np[t]; p++) {
			phi = (p*2*pi)/np[t];
			mod->shape.comp[c].real.v[v].a[0] =
					mod->shape.comp[c].real.v[v].a[1] =
							mod->shape.comp[c].real.v[v].a[2] = 0.0;
			mod->shape.comp[c].real.v[v].u[0] = sin(theta)*cos(phi);
			mod->shape.comp[c].real.v[v].u[1] = sin(theta)*sin(phi);
			mod->shape.comp[c].real.v[v].u[2] = cos(theta);
			mod->shape.comp[c].real.v[v].r.state = 'f';
			mod->shape.comp[c].real.v[v].r.val = 0.0;
			iv[t][p] = v;

			/*  For harmonic models, get the functions afactor[l][m] and bfactor[l][m]
          which multiply the a and b spherical harmonic coefficients.  They are
          normalized such that for any value of l (0 <= l <= L), the integral
          over the sphere of (afactor[l][0])^2 is 4*pi, while the integral of
          (afactor[l][m])^2 or (bfactor[l][m])^2 is 2*pi for 1 <= m <= l.         */

			if (mod->shape.comp[c].type == HARMONIC) {
				costheta = mod->shape.comp[c].real.v[v].u[2];
				for (l=0; l<=L; l++)
					for (m=0; m<=l; m++)
						plm[l][m] = nlm[l][m]*plgndr( l, m, costheta);

				if (CUDA) {
					/* Outer loop allocation */
					cudaCalloc((void**)&mod->shape.comp[c].real.v[v].afactor,
						sizeof(double*), L+1);
					/* Now inner loop allocation */
					for (int z=0; z<=L; z++)
						cudaCalloc((void**)&mod->shape.comp[c].real.v[v].afactor[z],
								sizeof(double), L+1);
				}
				else
					mod->shape.comp[c].real.v[v].afactor = matrix( 0, L, 0, L);

				if (L > 0) {
					if (CUDA) {
						/* Outer loop allocation */
						cudaCalloc((void**)&mod->shape.comp[c].real.v[v].bfactor,
								sizeof(double*), L+1);
						/* Now inner loop allocation */
						for (int z=0; z<=L; z++)
							cudaCalloc((void**)&mod->shape.comp[c].real.v[v].bfactor[z],
									sizeof(double), L+1);
					}
					else
						mod->shape.comp[c].real.v[v].bfactor = matrix( 1, L, 1, L);
				}
				for (l=0; l<=L; l++) {
					mod->shape.comp[c].real.v[v].afactor[l][0] = plm[l][0];
					for (m=1; m<=l; m++) {
						mod->shape.comp[c].real.v[v].afactor[l][m] = cos(m*phi)*plm[l][m];
						mod->shape.comp[c].real.v[v].bfactor[l][m] = sin(m*phi)*plm[l][m];
					}
				}
			}
			++v;
		}
	}
	if (mod->shape.comp[c].type == HARMONIC) {
		free_matrix( nlm, 0, L, 0, L);
		free_matrix( plm, 0, L, 0, L);
	}
}

void setuprealfac( struct mod_t *mod, int c, int **iv, int nt, int *np)
{
	int t, done, vs[3], p0, p1, p0t, p1t, nf=0, f=0, k, off1, off0;
	double B0, s1, pi, phi1, phi0;

	pi = 4*atan(1.0);
	/* count the facets */
	for (t=0; t<nt; t++) {
		off0 = off1 = p0 = p1 = 0;
		B0 = (2*pi)/np[t];
		s1 = (2*pi)/np[t+1];
		vs[0] = iv[t][0];
		vs[1] = iv[t+1][0];
		done = 0;
		while (!done) {
			p0t = p0 + 1;                                     /* potential next vertex on "top" */
			p1t = p1 + 1;                                     /* potential next vertex on "bottom" */
			phi0 = (p0t + off0)*B0;
			phi1 = (p1t + off1)*s1;
			if ((phi0 <= phi1) && (t != 0)) {                 /* if top is closer than bottom */
				if (p0t == np[t]) {                             /* and is not the north pole */
					p0t = 0;
					off0 = np[t];
				}
				vs[2] = iv[t][p0 = p0t];                        /* make that next vertex */
				vs[0] = vs[2];                                  /* set up next facet */
				++nf;
			}
			else if ((phi0 >= phi1) && (t != nt)) {           /* if bottom is closer than top */
				if (p1t == np[t+1]) {                           /* and is not south pole */
					p1t = 0;
					off1 = np[t+1];
				}
				vs[2] = iv[t+1][p1 = p1t];                      /* other use bottom */
				vs[1] = vs[2];                                  /* set up next facet */
				++nf;
			}
			if ((p0 == 0) && (p1 == 0))
				done = 1;
		}
	}
	mod->shape.comp[c].real.nf = nf;
	/*=======================================================================*/
	if (CUDA)
		cudaCalloc((void**)&mod->shape.comp[c].real.f,
			sizeof(struct facet_t), nf);
	else
		mod->shape.comp[c].real.f = calloc(nf, sizeof(struct facet_t));
	/*=======================================================================*/
		
	/* assign the facets */
	for (t=0; t<nt; t++) {
		off0 = off1 = p0 = p1 = 0;
		B0 = (2*pi)/np[t];
		s1 = (2*pi)/np[t+1];
		vs[0] = iv[t][0];
		vs[1] = iv[t+1][0];
		done = 0;
		while (!done) {
			p0t = p0 + 1;                                     /* potential next vertex on "top" */
			p1t = p1 + 1;                                     /* potential next vertex on "bottom" */
			phi0 = (p0t + off0)*B0;
			phi1 = (p1t + off1)*s1;
			if ((phi0 <= phi1) && (t != 0)) {                 /* if top is closer than bottom */
				if (p0t == np[t]) {                             /* and is not the north pole */
					p0t = 0;
					off0 = np[t];
				}
				vs[2] = iv[t][p0 = p0t];                        /* make that next vertex */
				for (k=0; k<=2; k++)
					mod->shape.comp[c].real.f[f].v[k] = vs[k];
				++f;
				vs[0] = vs[2];                                  /* set up next facet */
			}
			else if ((phi0 >= phi1) && (t != nt)) {           /* if bottom is closer than top */
				if (p1t == np[t+1]) {                           /* and is not south pole */
					p1t = 0;
					off1 = np[t+1];
				}
				vs[2] = iv[t+1][p1 = p1t];                      /* other use bottom */
				for (k=0; k<=2; k++)
					mod->shape.comp[c].real.f[f].v[k] = vs[k];
				++f;
				vs[1] = vs[2];                                  /* set up next facet */
			}
			if ((p0 == 0) && (p1 == 0))
				done = 1;
		}
	}
	printf("# component %d has %d facets in realization\n",
			c,mod->shape.comp[c].real.nf);
	fflush(stdout);
}

void setupsides( struct vertices_t *vt)
{
	int f, ns=0, v[3], i, new, s;

	vt->ns = (3*vt->nf)/2;
	
	/*=======================================================================*/
	if (CUDA)
		cudaCalloc((void**)&vt->s, sizeof(struct side_t), vt->ns);
	else
		vt->s = (struct side_t *) calloc(vt->ns, sizeof(struct side_t));
	/*=======================================================================*/
		
	for (f=0; f<vt->nf; f++) {          /* go through every facet */
		for (i=0; i<=2; i++)                /* make list of 3 vertices */
			v[i] = vt->f[f].v[i];
		if (v[0] < 0 || v[0] >= vt->nv || v[1] < 0 || v[1] >= vt->nv
				|| v[2] < 0 || v[2] >= vt->nv) {
			fprintf( stderr, "Facet %d (vertices %d %d %d) has out-of-bounds vertex numbers\n",
					f, v[0], v[1], v[2]);
			bailout("routine setupsides in read_mod.c\n");
		} else if (v[0] == v[1] || v[1] == v[2] || v[2] == v[0]) {
			fprintf( stderr, "Facet %d (vertices %d %d %d) does not have three different corners\n",
					f, v[0], v[1], v[2]);
			bailout("routine setupsides in read_mod.c\n");
		}
		while (!((v[0] < v[1]) && (v[1] < v[2]))) {  /* sort into ascending order */
			if (v[1] < v[0]) {
				i = v[0];
				v[0] = v[1];
				v[1] = i;
			}
			if (v[2] < v[1]) {
				i = v[1];
				v[1] = v[2];
				v[2] = i;
			}
		}

		new = 1;                                            /* assume v[0]-v[1] is a new side */
		for (s=0; s<ns; s++) {                              /* compare to current sides */
			if ((v[0] == vt->s[s].v[0]) && (v[1] == vt->s[s].v[1]))
				new = 0;
		}
		if (new) {
			vt->s[s].v[0] = v[0];
			vt->s[s].v[1] = v[1];
			++ns;
		}

		new = 1;                                            /* assume v[0]-v[2] is a new side */
		for (s=0; s<ns; s++) {                              /* compare to current sides */
			if ((v[0] == vt->s[s].v[0]) && (v[2] == vt->s[s].v[1]))
				new = 0;
		}
		if (new) {
			vt->s[s].v[0] = v[0];
			vt->s[s].v[1] = v[2];
			++ns;
		}

		new = 1;                                            /* assume v[1]-v[2] is a new side */
		for (s=0; s<ns; s++) {                              /* compare to current sides */
			if ((v[1] == vt->s[s].v[0]) && (v[2] == vt->s[s].v[1]))
				new = 0;
		}
		if (new) {
			vt->s[s].v[0] = v[1];
			vt->s[s].v[1] = v[2];
			++ns;
		}
	}

	if (ns != vt->ns) {
		fprintf( stderr, "Should find %d sides for component with %d vertices, found %d instead\n",
				vt->ns, vt->nv, ns);
		bailout("routine setupsides in read_mod.c\n");
	}

	for (s=0; s<vt->ns; s++)
		vt->s[s].f[0] = vt->s[s].f[1] = -1;

	for (f=0; f<vt->nf; f++) {          /* go through every facet */
		vt->f[f].s[0] = vt->f[f].s[1] = vt->f[f].s[2] = -1;
		for (i=0; i<=2; i++)                                        /* make list of 3 vertices */
			v[i] = vt->f[f].v[i];
		while (!((v[0] < v[1]) && (v[1] < v[2]))) {  /* sort into ascending order */
			if (v[1] < v[0]) {
				i = v[0];
				v[0] = v[1];
				v[1] = i;
			}
			if (v[2] < v[1]) {
				i = v[1];
				v[1] = v[2];
				v[2] = i;
			}
		}
		for (s=0; s<vt->ns; s++) {        /* check every side until 3 found */
			if ( ((vt->s[s].v[0] == v[0]) && (vt->s[s].v[1] == v[1]))
					|| ((vt->s[s].v[0] == v[0]) && (vt->s[s].v[1] == v[2]))
					|| ((vt->s[s].v[0] == v[1]) && (vt->s[s].v[1] == v[2])) ) {
				if (vt->s[s].f[0] < 0)
					vt->s[s].f[0] = f;
				else
					vt->s[s].f[1] = f;
				if (vt->f[f].s[0] < 0)
					vt->f[f].s[0] = s;
				else if (vt->f[f].s[1] < 0)
					vt->f[f].s[1] = s;
				else if (vt->f[f].s[2] < 0)
					vt->f[f].s[2] = s;
			}
		}
	}
}

void setupvertices( struct vertices_t *vt)
{
	int f, s, v, *nas, *naf;

	int maxnas=0, maxnaf=0;

	/* Create temp vectors for the # of attached sides and # of attached facets
	 * for each vertex.  Length of the temp vectors is [nv]	 */
	nas = ivector( 0, vt->nv);     /* number of sides  attached to each vertex */
	naf = ivector( 0, vt->nv);     /* number of facets attached to each vertex */

	/* Step through each vertex & initialize each vertex' naf and nas to zero */
	for (v=0; v<vt->nv; v++)
		vt->v[v].naf = vt->v[v].nas = 0;

	/* Step through each side and increase nas for both vertices that make up
	 * the side */
	for (s=0; s<vt->ns; s++) {
		++vt->v[vt->s[s].v[0]].nas;
		++vt->v[vt->s[s].v[1]].nas;
	}

	/* Step through each vertex */
	for (v=0; v<vt->nv; v++) {

		/* Set vertex' naf number equal to the nas calculated in previous loop */
		vt->v[v].naf = vt->v[v].nas;

		/* Create a vector each for attached facets and attached sides for
		 * each vertex */
		if (CUDA) {
			cudaCalloc((void**)&vt->v[v].as, sizeof(int), vt->v[v].nas);
			cudaCalloc((void**)&vt->v[v].af, sizeof(int), vt->v[v].naf);
		} else {
			vt->v[v].as = ivector( 0, vt->v[v].nas-1);   /* list of sides  attached to each vertex */
			vt->v[v].af = ivector( 0, vt->v[v].naf-1);   /* list of facets attached to each vertex */
		}

		/* Copy each vertex' nas and naf figures into the nas and naf vectors
		 * created at the beginning of this function for temporary storage	 */
		nas[v] = vt->v[v].nas;
		naf[v] = vt->v[v].naf;

		maxnas = MAX(maxnas, vt->v[v].nas);
		maxnaf = MAX(maxnaf, vt->v[v].naf);

		/* Now reset each vertex' naf and nas figures to zero */
		vt->v[v].nas = vt->v[v].naf = 0;
	}

	/* Step through each side and fill in the list of attached sides and # of
	 * attached sides for each sides two vertices */
	for (s=0; s<vt->ns; s++) {
		/* vt->v[x].as[y] = s */
		/* x = vt->s[s].v[0]   - int index of 1st vertex of current side
		 * y = vt->v[x].nas++  - the # of attached sides, increase after read
		 * 	 		 */
		vt->v[vt->s[s].v[0]].as[vt->v[vt->s[s].v[0]].nas++] = s;
		vt->v[vt->s[s].v[1]].as[vt->v[vt->s[s].v[1]].nas++] = s;
	}

	/* Step through each facet and fill in the list of attached facets and # of
	 * attached facets for each of the three vertices that make up each facet*/
	for (f=0; f<vt->nf; f++)
		for (v=0; v<=2; v++)
			/* vt->v[x].af[y] = f
			 *
			 * x = vt->f[f].v[v] - int index of vertex v attached to facet
			 * y = vt->v[x]  	 - # of att. facets, increase after read
			 * */
			vt->v[vt->f[f].v[v]].af[vt->v[vt->f[f].v[v]].naf++] = f;

	/* Step through each vertex and compare the just calculated nas and naf
	 * for each vertex against the naf and nas arrays copied earlier from
	 * vt, which was then reset and recalculated.*/
	for (v=0; v<vt->nv; v++) {
		if (nas[v] != vt->v[v].nas) {
			printf("%d != %d\n", nas[v], vt->v[v].nas);
			fflush(stdout);
		}
		if (naf[v] != vt->v[v].naf) {
			printf("%d != %d\n", naf[v], vt->v[v].naf);
			fflush(stdout);
		}
		maxnas = MAX(maxnas, vt->v[v].nas);
		maxnaf = MAX(maxnaf, vt->v[v].naf);
	}

	free_ivector( nas, 0, vt->nv);
	free_ivector( naf, 0, vt->nv);
}
