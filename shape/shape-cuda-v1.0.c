/*
 ============================================================================
 Name        : shape-cuda.cu
 Author      : Matthias Engels
 Version     : v0.75
 Copyright   : This work was done by Matthias Engels in the pursuit of his doctorate degree in Electrical Engineering.
 Description : S. Hudson's shape code, adapted to CUDA architecture
 ============================================================================
 */
#include "../shape/head.h"

int CUDA   = 0;		/* Use CUDA code or run CPU code 	*/
int TIMING = 0;		/* Time certain kernel executions	*/
int GPU0   = 1;		/* Which GPU will run code 			*/
int GPU1   = 0;
int maxThreadsPerBlock = 0;
int HMT	= 0;
int HMT_threads = 4;
int EXP = 0;

int main(int argc, char *argv[])
{
	/* The following 3 lines are for accurately timing multi-threaded code */
	struct timespec start, finish;
	double elapsed;
	clock_gettime(CLOCK_MONOTONIC, &start);
	long x;	/* For gpuid read-in */

	/* Declare variables */
	char progname[MAXLEN], errormessage[MAXLEN], localtimestring[MAXLEN];
	char *slashptr;
	struct par_t par, par1;				// the parameter structure
	struct mod_t mod, mod1, mod2, mod3;	// the model structure
	struct dat_t dat, dat1, dat2, dat3;	// the data structure
	struct rusage usage;				// the usage structure
	struct par_t *dev_par, *dev_par1;
	struct mod_t *dev_mod, *dev_mod1;
	struct dat_t *dev_dat, *dev_dat1;

	printf("CUDA Toolkit 10.0 in use.");

	/* Get program name (minus the path)  */
	slashptr = strrchr(argv[0], '/');
	if (slashptr)
		strcpy(progname, slashptr+1);
	else
		strcpy(progname, argv[0]);

	/*  initialize  */
	init( argc, argv, progname);

	if (argc >= 6) {
		if (strcmp(argv[4], "-GPU")==0) {
			CUDA = 1;
			x = strtol(argv[5], NULL, 10);
			GPU0 = (int)x;
		}
		else if (strcmp(argv[4], "-HMT")==0) {
			CUDA = 0;
			HMT = 1;
			x = strtol(argv[5], NULL, 10);
			HMT_threads = (int) x;
		}
		else
			CUDA = 0;
	}
	if (CUDA) {
		printf("Shape-CUDA-v1.2(A) running\n");
		printf("Now with even more face-melting concurrency.\n");
		/* Check available CUDA devices, if any, before proceeding */
		CUDACount(1);
		maxThreadsPerBlock = 256;
		gpuErrchk(cudaSetDevice(GPU0));
	}

	/* Read the par file, get the action, and make sure actions other than
	 * "fit" do NOT use parallel processing.   */
	read_par( argv[1], &par);

	/*  Record the names of the mod and obs files  */
	if (par.action == ORBIT) {
		if (argc == 5) {
			sprintf( mod.name,  "%s", argv[2]);
			sprintf( mod2.name, "%s", argv[3]);
			sprintf( dat.name,  "%s", argv[4]);  /* use same obs file twice */
			sprintf( dat2.name, "%s", argv[4]);
		} else if (argc == 6) {
			sprintf( mod.name,  "%s", argv[2]);
			sprintf( mod2.name, "%s", argv[3]);
			sprintf( mod3.name, "%s", argv[4]);
			sprintf( dat.name,  "%s", argv[5]);  /* use same obs file three times */
			sprintf( dat2.name, "%s", argv[5]);
			sprintf( dat3.name, "%s", argv[5]);
		}
	} else {
		if (argc > 2) {
			sprintf( mod.name, "%s", argv[2]);
			sprintf(mod1.name, "%s", argv[2]);
		}
		if (argc > 3) {
			sprintf( dat.name, "%s", argv[3]);
			sprintf(dat1.name, "%s", argv[3]);
		}
	}

	/*  Carry out the desired action - 'fit' only for now */
	switch (par.action) {
	case FORMAT:
		if (argc != 4) {
			sprintf(errormessage, "usage - %s formatpar modfile obsfile\n", progname);
			bailout(errormessage);
		}
		read_mod( &par, &mod);
		read_dat( &par, &mod, &dat);
		realize_mod( &par, &mod);
		realize_spin( &par, &mod, &dat);
		realize_photo( &par, &mod, 1.0, 1.0, 0);
		/* realize_delcor, realize_dopscale, and realize_xyoff were called in read_dat */
		calc_fits( &par, &mod, &dat);
		chi2( &par, &dat, 0);
		write_mod( &par, &mod);
		write_dat( &par, &dat);
		break;
	case MIRROR:
		if (argc != 3 && argc != 4) {
			sprintf(errormessage, "usage - %s mirrorpar modfile [obsfile]\n", progname);
			bailout(errormessage);
		}
		read_mod( &par, &mod);
		mirror_mod( &mod);
		realize_mod( &par, &mod);
		write_mod( &par, &mod);
		if (argc == 4) {
			read_dat( &par, &mod, &dat);
			mirror_dat( &dat);
			realize_spin( &par, &mod, &dat);
			write_dat( &par, &dat);
		}
		break;
	case WRITE:
		if (argc != 4) {
			sprintf(errormessage, "usage - %s writepar modfile obsfile\n", progname);
			bailout(errormessage);
		}
		read_mod( &par, &mod);
		read_dat( &par, &mod, &dat);
		realize_mod( &par, &mod);
		realize_spin( &par, &mod, &dat);
		realize_photo( &par, &mod, 1.0, 1.0, 0);
		/* realize_delcor, realize_dopscale, and realize_xyoff were called in read_dat */
		calc_fits( &par, &mod, &dat);
		show_deldoplim( &dat);
		chi2( &par, &dat, 1);
		if (par.write_obs)
			write_dat( &par, &dat);
		break;
	case FIT:

		par.nfpar = 0;
		if (CUDA) {
			par.nfpar += read_mod( &par, &mod);
			par.nfpar += read_dat_gpu( &par, &mod, &dat);
		}
		else {
			par.nfpar += read_mod( &par, &mod);
			par.nfpar += read_dat( &par, &mod, &dat);
		}
		mkparlist( &par, &mod, &dat);

		/* Make CUDA device copies of par, mod, dat (these copies reside
		 * in device memory and are inaccessible by the host (CPU) code */
		if (CUDA) {

			cudaCalloc1((void**)&dev_par, sizeof(struct par_t), 1);
			gpuErrchk(cudaMemcpy(dev_par, &par, sizeof(struct par_t),
					cudaMemcpyHostToDevice));
			cudaCalloc1((void**)&dev_mod, sizeof(struct mod_t), 1);
			gpuErrchk(cudaMemcpy(dev_mod, &mod, sizeof(struct mod_t),
					cudaMemcpyHostToDevice));
			cudaCalloc1((void**)&dev_dat, sizeof(struct dat_t), 1);
			gpuErrchk(cudaMemcpy(dev_dat, &dat, sizeof(struct dat_t),
					cudaMemcpyHostToDevice));
			bestfit_gpu(dev_par,dev_mod,dev_dat, &par,&mod,&dat);
		}
		else if (HMT)
			bestfit_hmt(&par, &mod, &dat);
		else
			bestfit( &par, &mod, &dat);

		break;
	case FACETS:
		if (argc != 3) {
			sprintf(errormessage, "usage - %s facetspar modfile\n", progname);
			bailout(errormessage);
		}
		read_mod( &par, &mod);
		realize_mod( &par, &mod);
		write_mod( &par, &mod);
		break;
	case SAMPLE:
		if (argc != 3) {
			sprintf(errormessage, "usage - %s samplepar modfile\n", progname);
			bailout(errormessage);
		}
		read_mod( &par, &mod);
		realize_mod( &par, &mod);
		sample_mod( &par, &mod);
		break;
	case COVAR:
		if (argc != 4) {
			sprintf(errormessage, "usage - %s covarpar modfile obsfile\n", progname);
			bailout(errormessage);
		}
		par.nfpar = 0;
		par.nfpar += read_mod( &par, &mod);
		par.nfpar += read_dat( &par, &mod, &dat);
		mkparlist( &par, &mod, &dat);
		covar( &par, &mod, &dat);
		break;
	case WAVEFRONT:
		if (argc != 3) {
			sprintf(errormessage, "usage - %s wavefrontpar modfile\n", progname);
			bailout(errormessage);
		}
		read_mod( &par, &mod);
		realize_mod( &par, &mod);
		if (mod.shape.ncomp == 1)
			write_wf( &mod);
		else
			merge_comps( &par, &mod);
		break;
	case REFSHAPE:
		if (argc != 3) {
			sprintf(errormessage, "usage - %s refshapepar modfile\n", progname);
			bailout(errormessage);
		}
		read_mod( &par, &mod);
		realize_mod( &par, &mod);
		ref_mod( &mod);
		write_mod( &par, &mod);
		break;
	case MOMENTS:
		if (argc != 3 && argc != 4) {
			sprintf(errormessage, "usage - %s momentspar modfile [obsfile]\n", progname);
			bailout(errormessage);
		}
		if (par.mark_unseen) {
			if (argc != 4) {
				printf("\nNeed obs file for moments action if 'mark_unseen' is turned on\n\n");
				fflush(stdout);
				sprintf(errormessage, "usage - %s momentspar modfile obsfile\n", progname);
				bailout(errormessage);
			}
			read_mod( &par, &mod);
			read_dat( &par, &mod, &dat);
			realize_mod( &par, &mod);
			realize_spin( &par, &mod, &dat);
			realize_photo( &par, &mod, 1.0, 1.0, 0);
			/* realize_delcor, realize_dopscale, and realize_xyoff were called in read_dat */
			calc_fits( &par, &mod, &dat);
		} else {
			read_mod( &par, &mod);
			realize_mod( &par, &mod);
			realize_photo( &par, &mod, 1.0, 1.0, 0);
		}
		show_moments( &par, &mod);
		if (!par.mark_unseen && argc != 3) {
			printf("WARNING: obs file not used for moments action if 'mark_unseen' is turned off\n\n");
			fflush(stdout);
		}
		break;
	case DELCORINIT:
		if (argc != 4) {
			sprintf(errormessage, "usage - %s delcorinitpar modfile obsfile\n", progname);
			bailout(errormessage);
		}
		read_mod( &par, &mod);
		read_dat( &par, &mod, &dat);
		realize_mod( &par, &mod);
		realize_spin( &par, &mod, &dat);
		realize_photo( &par, &mod, 1.0, 1.0, 0);
		/* realize_delcor, realize_dopscale, and realize_xyoff were called in read_dat */
		calc_fits( &par, &mod, &dat);
		delcorinit( &par, &mod, &dat);
		calc_fits( &par, &mod, &dat);
		chi2( &par, &dat, 0);
		write_mod( &par, &mod);
		write_dat( &par, &dat);
		break;
	case CONVEXHULL:
		if (argc != 3) {
			sprintf(errormessage, "usage - %s convexpar modfile\n", progname);
			bailout(errormessage);
		}
		read_mod( &par, &mod);
		realize_mod( &par, &mod);
		convex_hull( &par, &mod);
		break;
	case VIEW:
		if (argc != 3 && argc != 4) {
			sprintf(errormessage, "usage - %s viewpar modfile [obsfile]\n", progname);
			bailout(errormessage);
		}
		if (par.mark_unseen) {
			if (argc != 4) {
				printf("\nNeed obs file for view action if 'mark_unseen' is turned on\n\n");
				fflush(stdout);
				sprintf(errormessage, "usage - %s viewpar modfile obsfile\n", progname);
				bailout(errormessage);
			}
			read_mod( &par, &mod);
			read_dat( &par, &mod, &dat);
			realize_mod( &par, &mod);
			realize_spin( &par, &mod, &dat);
			realize_photo( &par, &mod, 1.0, 1.0, 0);
			/* realize_delcor, realize_dopscale, and realize_xyoff were called in read_dat */
			calc_fits( &par, &mod, &dat);
		} else {
			read_mod( &par, &mod);
			realize_mod( &par, &mod);
			realize_photo( &par, &mod, 1.0, 1.0, 0);
		}
		view_mod( &par, &mod);
		if (!par.mark_unseen && argc != 3) {
			printf("\nWARNING: obs file not used for view action if 'mark_unseen' is turned off\n\n");
			fflush(stdout);
		}
		break;
	case AREA:
		if (argc != 3) {
			sprintf(errormessage, "usage - %s areapar modfile\n", progname);
			bailout(errormessage);
		}
		read_mod( &par, &mod);
		realize_mod( &par, &mod);
		proj_area( &par, &mod);
		break;
	case PHOTOFACETS:
		if (argc != 3) {
			sprintf(errormessage, "usage - %s photofacetspar modfile\n", progname);
			bailout(errormessage);
		}
		read_mod( &par, &mod);
		realize_mod( &par, &mod);
		realize_photo( &par, &mod, 1.0, 1.0, 0);
		write_mod( &par, &mod);
		break;
	case PHOTOHARM:
		if (argc != 3) {
			sprintf(errormessage, "usage - %s photoharmpar modfile\n", progname);
			bailout(errormessage);
		}
		read_mod( &par, &mod);
		realize_mod( &par, &mod);
		realize_photo( &par, &mod, 1.0, 1.0, 0);
		write_mod( &par, &mod);
		break;
	case SLICE:
		if (argc != 3 && argc != 4) {
			sprintf(errormessage, "usage - %s slicepar modfile [obsfile]\n", progname);
			bailout(errormessage);
		}
		if (par.mark_unseen) {
			if (argc != 4) {
				printf("\nNeed obs file for slice action if 'mark_unseen' is turned on\n\n");
				fflush(stdout);
				sprintf(errormessage, "usage - %s slicepar modfile obsfile\n", progname);
				bailout(errormessage);
			}
			read_mod( &par, &mod);
			read_dat( &par, &mod, &dat);
			realize_mod( &par, &mod);
			realize_spin( &par, &mod, &dat);
			realize_photo( &par, &mod, 1.0, 1.0, 0);
			/* realize_delcor, realize_dopscale, and realize_xyoff were called in read_dat */
			calc_fits( &par, &mod, &dat);
		} else {
			read_mod( &par, &mod);
			realize_mod( &par, &mod);
			realize_photo( &par, &mod, 1.0, 1.0, 0);
		}
		slice( &par, &mod);
		if (!par.mark_unseen && argc != 3) {
			printf("\nWARNING: obs file not used for slice action if 'mark_unseen' is turned off\n\n");
			fflush(stdout);
		}
		break;
	case ORBIT:
		if (par.is_triple) {
			if (argc != 6) {
				printf("\nNeed third mod file for orbit action if par file specifies a triple system\n\n");
				fflush(stdout);
				sprintf(errormessage, "usage - %s orbitpar modfile1 modfile2 modfile3 obsfile\n", progname);
				bailout(errormessage);
			}
			read_mod( &par, &mod);
			read_mod( &par, &mod2);
			read_mod( &par, &mod3);
			read_dat( &par, &mod, &dat);
			read_dat( &par, &mod2, &dat2);
			read_dat( &par, &mod3, &dat3);
			realize_mod( &par, &mod);
			realize_mod( &par, &mod2);
			realize_mod( &par, &mod3);
			/* call realize_spin three times for same obs file, to get spin vector
	             and coordinate transformation matrices at each epoch for each model */
			realize_spin( &par, &mod,  &dat);
			realize_spin( &par, &mod2, &dat2);
			realize_spin( &par, &mod3, &dat3);
			realize_photo( &par, &mod,  1.0, 1.0, 0);
			realize_photo( &par, &mod2, 1.0, 1.0, 0);
			realize_photo( &par, &mod3, 1.0, 1.0, 0);
			/* realize_delcor, realize_dopscale, and realize_xyoff were called in read_dat */
			calc_orbit( &par, &mod, &mod2, &mod3, &dat, &dat2, &dat3);
		} else {
			if (argc != 5) {
				printf("\nNeed only two mod files for orbit action if par file specifies a binary system\n\n");
				fflush(stdout);
				sprintf(errormessage, "usage - %s orbitpar modfile1 modfile2 obsfile\n", progname);
				bailout(errormessage);
			}
			read_mod( &par, &mod);
			read_mod( &par, &mod2);
			read_dat( &par, &mod, &dat);
			read_dat( &par, &mod2, &dat2);
			realize_mod( &par, &mod);
			realize_mod( &par, &mod2);
			/* call realize_spin two times for same obs file, to get spin vector
	             and coordinate transformation matrices at each epoch for each model */
			realize_spin( &par, &mod,  &dat);
			realize_spin( &par, &mod2, &dat2);
			realize_photo( &par, &mod,  1.0, 1.0, 0);
			realize_photo( &par, &mod2, 1.0, 1.0, 0);
			/* realize_delcor, realize_dopscale, and realize_xyoff were called in read_dat */
			calc_orbit( &par, &mod, &mod2, &mod2, &dat, &dat2, &dat2);
		}
		show_deldoplim( &dat);
		chi2( &par, &dat, 1);
		break;
	case SPLIT:
		if (argc != 3) {
			sprintf(errormessage, "usage - %s splitpar modfile\n", progname);
			bailout(errormessage);
		}
		read_mod( &par, &mod);
		realize_mod( &par, &mod);
		split_mod( &par, &mod);
		break;
	case MAP:
		if (argc != 4) {
			sprintf(errormessage, "usage - %s mappar modfile obsfile\n", progname);
			bailout(errormessage);
		}
		read_mod( &par, &mod);
		read_dat( &par, &mod, &dat);
		realize_mod( &par, &mod);
		realize_spin( &par, &mod, &dat);
		realize_photo( &par, &mod, 1.0, 1.0, 0);
		/* realize_delcor, realize_dopscale, and realize_xyoff were called in read_dat */
		map_radar( &par, &mod, &dat);
		break;

	default:
		printf("\npar.action: %i\n\n", par.action);
		bailout("shape.c: undefined action in main switch statement (ln 96).\n");
	}

	/*  List user and system CPU usage and sign off */
	if (!getrusage (RUSAGE_SELF, &usage))
		printf("\n# cpu usage (sec): %f user %f system\n",
				usage.ru_utime.tv_sec + 1.e-6 * usage.ru_utime.tv_usec,
				usage.ru_stime.tv_sec + 1.e-6 * usage.ru_stime.tv_usec);
	else
		printf("# cpu usage (sec): %f user %f system\n", -9.99, -9.99);
	printf("# ready to exit\n");
	timestring(localtimestring, MAXLEN, "");
	printf("#\n");
	printf("# ending time %s\n", localtimestring);


	clock_gettime(CLOCK_MONOTONIC, &finish);

	elapsed = (finish.tv_sec - start.tv_sec);
	elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

	printf("\n\n###   Elapsed time from start to finish: %g    ###\n\n", elapsed);
	if (CUDA)
		printf("Full double-precision (more accurate) shape-cuda performed with one GPU\n");
	return 0;
}

