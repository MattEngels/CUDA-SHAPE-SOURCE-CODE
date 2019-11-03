/*****************************************************************************************
                                                                               read_par.c

Reads a general parameter file into struct par_t par.

Modified 2015 July 7 by SN:
    Bug fix: "sun_appmag" parameter was incorrectly numbered

Modified 2014 August 22 by SN:
    Add lib_amp_step, lib_amp_tol and lib_amp_abstol for librations.
    Add similar variables for lib_freq and lib_phase.

Modified 2014 February 25 by CM:
    Add "delcor_verbose" parameter

Modified 2014 February 19 by CM:
    If "mask_tol" isn't specified, set it to the POS pixel width if "pos_width" and
        "pos_pixels" have been specified; otherwise set it to 0.0 as before
    For the map action, require that "pos_pixels" and "pos_width" are specified
    Bug fix: for the map action with map_mode = facets, don't have to specify the map_set
        and map_frame parameters

Modified 2013 July 28 by CM:
    Add "write_highlight" "write_comp" and "write_facet" parameters

Modified 2013 July 14 by CM:
    Add "term_maxiter" parameter

Modified 2013 July 5 by CM:
    Add "pa3tilt" penalty function

Modified 2013 June 25 by CM:
    If setting an angle parameter to a default value, print that value to the screen in
        degrees rather than radians

Modified 2013 April 14 by CM:
    Add "listpos_deldop" "listpos_opt" and "listpos_path" parameters
        (forgot to upload this change until 2013 June 7)

Modified 2012 December 5 by CM:
    Add "chi2fit0_thresh" parameter

Modified 2012 June 13 by CM:
    Add "objfunc_start" parameter

Modified 2012 March 24 by CM:
    Add "vary_dopscale" "dopscale_min" and "dopscale_max" parameters

Modified 2012 March 5 by CM:
    Add "pa_highlight" "pa_comp" "pa_facet" "view_highlight" "view_comp" and "view_facet"
        parameters
    Rename MAXMAPFACETS defined constant as MAXCHOSENFACETS
    Add "pa_bodycoords" parameter

Modified 2012 February 1 by CM:
    Change name of "getline" routine to "readline"

Modified 2011 August 22 by CM:
    Add "impulse" penalty function
    Add "int_abstol" parameter

Modified 2010 September 1 by CM:
    Forgot to give prototype for "read_intvecpar" routine
    Was incorrectly passing arguments in some calls to read_intvecpar

Modified 2010 August 25 by CM:
    Change "map_forward" parameter to "map_mode" and give it three options
        rather than two
    Add "map_comp" "map_facet" and "map_verbose" parameters
    Add "remove_comments" routine

Modified 2010 August 10 by CM:
    Add "radfitmin" and "radobsmin" parameters

Modified 2010 June 15 by CM:
    Remove "rescale" action and associated "scalefactor" parameter
    Add "map" action
    Add "map_set" "map_frame" "map_forward" "map_dellim" "map_doplim"
        "map_xposlim" "map_yposlim" "map_posfactor" "map_fitcutoff" and
        "map_poscutoff" parameters
    Add "read_intvecpar" routine

Modified 2010 May 20 by CM:
    Add "split" action
    Add "split_plane" parameter (which determines "split_norm" and
        "split_const")
    Add NACTIONS defined constant

Modified 2010 April 27 by CM:
    Add "bifurcation" penalty function
    Add "rad_rho_min" and "rad_rho_max" parameters
    Add "noncosine" penalty function

Modified 2009 July 5 by CM:
    Add "npar_update" parameter

Modified 2009 April 27 by CM:
    Initialize "sun_appmag" parameter to the SOLAR_VMAG defined constant

Modified 2009 April 20 by CM:
    Fix small bug in initializing delcor_step and delcor_abstol components

Modified 2009 April 10 by CM:
    Add "slice_posfactor" and "view_posfactor" parameters

Modified 2009 April 3 by CM:
    Add "warn_badradar" "plot_angmom" "pierce_spinvec" and "pierce_angmom"
        parameters
    Slight fix to some error messages

Modified 2008 August 10 by CM:
    Change "delcor_step" and "delcor_abstol" parameters to be vectors
        rather than scalars

Modified 2008 July 11 by CM:
    Change parameters for the "orbit" action to allow for a triple system
    Change the routines that read parameter vectors so that they don't
        read beyond the current line

Modified 2007 September 20 by CM:
    Add "write_chi2fit0" parameter
    Streamline code by creating "read_yesnopar" "initialize_par"
        "adjust_par" and similar routines

Modified 2007 August 31 by CM:
    Add "avoid_badpos" "bad_objfactor" and "maskdir" parameters
    Don't include <stdio.h> (it's already included by "head.h")

Modified 2007 August 19 by CM:
    Replace local Arecibo version that was accidentally uploaded
        (and that doesn't compile elsewhere)

Modified 2007 August 15 by CM:
    Add "term_badmodel" parameter

Modified 2007 August 11 by CM:
    Eliminate unused variable
    Add "radfitmax" and "radobsmax" and "write_posfactor" parameters

Modified 2007 August 4 by CM:
    Add "orbit" action
    Add "semimajor_axis" "r_pericenter" "eccentricity" "t_pericenter"
        "long_asc_node" "inclination" "arg_pericenter" "binary_gravparam"
        "orbit_period" "orbit_reflex" and "orbit_posfactor" parameters

Modified 2007 February 21 by CM:
    Add "slice" action
    Add "slice_long" "slice_lat" "slice_offset" "slice_viewlong"
        "slice_viewlat" "slice_sunlong" "slice_sunlat" "slice_scatlaw"
        "slice_shadows" "slice_planefrac" "slice_skyfactor"
        "slice_dimfactor" "slice_read" and "slice_dointerp" parameters
    Add "maxrdev" and "maxellipdev" penalty functions

Modified 2006 October 1 by CM:
    Add "ratio_step" "ratio_tol" and "ratio_abstol" parameters
    Add "vary_delcor0" "vary_radalb" and "vary_optalb" parameters

Modified 2006 June 18 by CM:
    Add 24 parameters which are user-specified upper and lower limits to
        photometric parameters: rad_R_min, rad_R_max, rad_C_min, opt_R_max,
        etc.

Modified 2006 April 6 by PT:
    Add "spindot_step", "spindot_tol", and "spindot_abstol" for changing
        spin rates

Modified 2006 March 10 by CM:
    Print a warning if the "pos_scope" parameter is set to "local" for the
        "fit" action

Modified 2005 October 6 by CM:
    Add "sun_appmag" parameter

Modified 2005 September 18 by CM:
    Add "view_shadows" "view_sunlong" and "view_sunlat" parameters

Modified 2005 September 11 by CM:
    Added new "photoharm" action and various parameters needed for it:

Modified 2005 September 8 by CM:
    Added new "photofacets" action and various parameters needed for it:
        radfacets, optfacets, rad_global_R_state, opt_local_wt_state, etc.

Modified 2005 August 1 by CM:
    Add "pa_scatlaw" and "view_scatlaw" and "sky_radlaw" and "sky_optlaw"
        parameters

Modified 2005 July 21 by CM:
    Add "exclude_seen" parameter
    Add "thetadel" penalty function for use with the "inhohapke" optical
        scattering law
    Add "rad_c_del" and "rad_c_var" penalty functions for use with the
        "inhocosine" radar scattering law
    Check that no parameter or penalty function is specified more than once
        in the parameter file

Modified 2005 July 3 by CM:
    Added "lcrv_writeall" parameter

Modified 2005 July 2 by CM and JLM:
    Validate all parameters: set default values if appropriate, or else
        exit if crucial parameters such as "pos_pixels" and "pos_width"
        and "term_prec" were not specified.  Previously this validation
        was done for many but not all parameters.

Modified 2005 June 24 by CM:
    Added "mark_unseen" and "mincosine_seen" parameters

Modified 2005 June 22 by PAT:
    Fixed typo in endian parameter initialization

Modified 2005 June 20 by CM:
    Added "endian" parameter

Modified 2005 March 16 by CM:
    Stop treating "wavefront" action as different from others
    Recognize that "covar" action requires parameter step sizes

Modified 2005 March 6 by CM:
    Added "list_posetcal" parameter

Modified 2005 February 24 by CM:
    Removed reference to nonexistent "mpiload" action
    Moved two checks for required parameters from main program to here
    Added "xyoff_step" "xyoff_tol" "xyoff_abstol" fitting parameters;
        use default values if not specified in the par file.

Modified 2005 February 21 by CM:
    Add "cubicconv" option for "dd_scaling" and "poset_scaling" parameters
    Add "poset_resample" and "image_rebin" parameters

Modified 2005 February 6 by CM:
    Add "delta_rotphase" parameter

Modified 2005 January 19 by CM:
    Add "bilinear" option for "dd_scaling" parameter
    Add "poset_scaling" and "poset_resid" parameters
    Set default values for "dd_scaling" "dd_resid" "poset_scaling" and
        "poset_resid"

Modified 2005 January 9 by CM:
    Close parameter file when finished reading
    For parallel processing, suppress most screen output for nodes
        other than root

Modified 2004 December 20 by CM:
    Added new "area_latincr" and "area_longincr" parameters

Modified 2004 November 30 by CM:
    Added new "listfit" and "listfit_path" parameters

Modified 2004 November 23 by CM:
    Added new "view" action and "view_long" and "view_lat" parameters

Modified 2004 October 29 by CM:
    Added new "first_fitpar" parameter

Modified 2004 October 7 by CM:
    Quit if an unknown parameter is read in

Modified 2004 August 13 by CM:
    Add six new parameters, absolute tolerances for fitting:
        "length_abstol" "angle_abstol"   "spin_abstol"
        "photo_abstol"  "inertia_abstol" "delcor_abstol"

Modified 2004 July 31 by CM:
    Added new "delcor_read" and "delcor_file" parameters

Modified 2004 May 8 by CM:
    Added new "flattening" penalty
    Added new "rescale" action and "scalefactor" parameter

Modified 2004 April 25 by CM:
    Added new "inertiadev_uni" and "nonpa_uni" penalties for PA rotators

Modified 2004 March 27 by CM:
    Added new "plot_spinvec" and "plot_subradar" and "plot_com"
        and "plot_pa" and "write_obs" parameters

Modified 2004 March 6 by CM:
    Added new "convexhull" action and the new "convex_file" parameter

Modified 2004 February 16 by CM:
    Added new "delcorthresh" and "jdoffset" and "perform_ltc" parameters

Modified 2003 July 30 by CM:
    Added new "dd_maxsides" parameter

Modified 2003 May 16 by CM:
    Added new "listres" parameter

Modified 2003 May 14 by CM:
    Added new "delcorinit" action

Modified 2003 May 9 by CM:
    Added new "scalefitobs" parameter

Modified 2003 May 7 by CM:
    Added new "sinc2width" parameter

Modified 2003 May 1 by CM:
    Initialize all "step" and "tol" parameters to negative values;
    don't set photo_step = length_step (and so on) unless length_step
    (and so on) has been set in the parameter file

Modified 2003 April 30 by CM:
    Initialize "theta_steps" parameter to 0, so that it can actually
    be used for something (i.e., to override "ntheta" if specified)

Modified 2003 April 29 by CM:
    Added new "nsinc2" parameter

Modified 2003 April 23 by CM:
    Added new "moments" action and six new parameters:
        pa_ephfile, pa_ephformat, pa_startepoch, pa_stopepoch,
        pa_angleoff, pa_omegaoff
        (Note that the last two are *vectors*)

Modified 2003 April 3 by CM:
    Allow new "photo_step" and "photo_tol" values for fitting photometric
        parameters; for backwards compatibilty, set them to length_step
        and length_tol if values were not specified in the parameter file
    Allow new "inertia_step" and "inertia_tol" values for fitting moments
        of inertia, and "delcor_step" and "delcor_tol" for fitting delay
        correction polynomial coefficients; for backwards compatibilty,
        set these to spin_step (in deg/day) and spin_tol if values were
        not specified in the parameter file

*****************************************************************************************/

#include "head.h"

#define NPARAMS 248
#define NPENALTIES 25
#define NACTIONS 20

void initialize_par( struct par_t *par);
void read_yesnopar( FILE *fp, unsigned char *parval, char *parname,
                    int par_was_specified[], int n);
int read_yesnovecpar( FILE *fp, unsigned char parval[], char *parname, int n_elements,
                      int par_was_specified[], int n);
void read_statepar( FILE *fp, char *parval, char *parname, int equals_allowed,
                    int par_was_specified[], int n);
void read_optionspar( FILE *fp, unsigned char *parval, char *parname,
                      const char *options[], unsigned char parvals[], int noptions,
                      int par_was_specified[], int n);
void read_stringpar( FILE *fp, char *parval, char *parname,
                     int par_was_specified[], int n);
void read_intpar( FILE *fp, int *parval, char *parname,
                  int par_was_specified[], int n);
int read_intvecpar( FILE *fp, int parval[], char *parname, int n_elements,
                    int par_was_specified[], int n);
void read_jdpar( FILE *fp, double *parval, char *parname,
                 int par_was_specified[], int n);
int read_jdvecpar( FILE *fp, double parval[], char *parname, int n_elements,
                   int par_was_specified[], int n);
void read_minmaxpar( FILE *fp, double *parval, char *parname, int minmaxflag,
                     int par_was_specified[], int n);
void read_doublepar( FILE *fp, double *parval, char *parname,
                     int par_was_specified[], int n);
int read_doublevecpar( FILE *fp, double parval[], char *parname, int n_elements,
                       int par_was_specified[], int n);
void remove_comments( char *str);
void adjust_par( struct par_t *par);
void read_penalties( FILE *fp, struct par_t *par);
void assign_penalty( int pentype_val, int *pentype, char *penname, double penweight,
                     int pen_was_specified[], int n);
void check_if_already_read( int was_already_read[], int ntot, char *str, int n);
void check_triple( int n_read, int n_max, int *triple);


void read_par( char *name, struct par_t *par)
{
  FILE *fp;
  char str[MAXLEN];
  int i, par_was_specified[NPARAMS], n_read;
  double maxangle_seen, split_plane[4], normconst;

  /*  Initialize all parameters prior to reading in values from the parameter file  */

  initialize_par( par);

  /*  Initialize an array that shows whether or not each
      parameter has been specified so far in the parameter file  */

  for (i=0; i<NPARAMS; i++)
    par_was_specified[i] = 0;

  /*  Open the parameter file and check that it begins with the PARAMETERS keyword  */

  printf("#\n# reading parameters from file: %s\n", name);
  fflush(stdout);

  FOPEN( fp, name, "r");
  gettstr( fp, str);
  if (strcmp( str, "PARAMETERS"))
    bailout("read_par.c: expected keyword PARAMETERS\n");

  /*  Main loop:                                */
  /*                                            */
  /*  Read the parameters one by one until you  */
  /*  hit the penalty section (or else EOF)     */

  while (strcmp( str, "PENALTIES") && !feof(fp)) {
    gettstr( fp, str);

    if (!strcmp( str, "action")) {
        const char *options[] =
                         {"fit",         "write",      "format",    "mirror",   "facets",
                          "sample",      "covar",      "wavefront", "refshape", "moments",
                          "delcorinit",  "convexhull", "map",       "view",     "area",
                          "photofacets", "photoharm",  "slice",     "orbit",    "split"};
        unsigned char parvals[] =
                         { FIT,           WRITE,        FORMAT,      MIRROR,     FACETS,
                           SAMPLE,        COVAR,        WAVEFRONT,   REFSHAPE,   MOMENTS,
                           DELCORINIT,    CONVEXHULL,   MAP,         VIEW,       AREA,
                           PHOTOFACETS,   PHOTOHARM,    SLICE,       ORBIT,      SPLIT};
        read_optionspar( fp, &par->action, str, options, parvals, NACTIONS, par_was_specified, 0);
    }
    else if (!strcmp( str, "speckle")) {
        read_yesnopar( fp, &par->speckle, str, par_was_specified, 1);
    }
    else if (!strcmp( str, "pos_smooth")) {
        read_yesnopar( fp, &par->pos_smooth, str, par_was_specified, 2);
    }
    else if (!strcmp( str, "read_node")) {
        read_yesnopar( fp, &par->read_node, str, par_was_specified, 3);
    }
    else if (!strcmp( str, "pos_scope")) {
        const char *options[]   = {"global", "local"};
        unsigned char parvals[] = { GLOBAL,   LOCAL };
        read_optionspar( fp, &par->pos_scope, str, options, parvals, 2, par_was_specified, 4);
    }
    else if (!strcmp( str, "dd_scaling")) {
        const char *options[]   = {"none", "block", "bilinear", "bicubic", "cubicconv"};
        unsigned char parvals[] = { NONE,   BLOCK,   BILINEAR,   BICUBIC,   CUBICCONV };
        read_optionspar( fp, &par->dd_scaling, str, options, parvals, 5, par_was_specified, 5);
    }
    else if (!strcmp( str, "int_method")) {
        const char *options[]   = {"brute", "periodic"};
        unsigned char parvals[] = { BRUTE,   PERIODIC };
        read_optionspar( fp, &par->int_method, str, options, parvals, 2, par_was_specified, 6);
    }
    else if (!strcmp( str, "lcrv_pos")) {
        read_yesnopar( fp, &par->lcrv_pos, str, par_was_specified, 7);
    }
    else if (!strcmp( str, "theta_steps")) {
        read_intpar( fp, &par->theta_steps, str, par_was_specified, 8);
    }
    else if (!strcmp( str, "ver_samps")) {
        read_intpar( fp, &par->ver_samps, str, par_was_specified, 9);
    }
    else if (!strcmp( str, "pos_pixels")) {
        read_intpar( fp, &par->pos_pixels, str, par_was_specified, 10);
    }
    else if (!strcmp( str, "pos_width")) {
        read_doublepar( fp, &par->pos_width, str, par_was_specified, 11);
    }
    else if (!strcmp( str, "dd_gamma")) {
        read_doublepar( fp, &par->dd_gamma, str, par_was_specified, 12);
    }
    else if (!strcmp( str, "mask_tol")) {
        read_doublepar( fp, &par->mask_tol, str, par_was_specified, 13);
    }
    else if (!strcmp( str, "length_step")) {
        read_doublepar( fp, &par->length_step, str, par_was_specified, 14);
    }
    else if (!strcmp( str, "length_tol")) {
        read_doublepar( fp, &par->length_tol, str, par_was_specified, 15);
    }
    else if (!strcmp( str, "length_abstol")) {
        read_doublepar( fp, &par->length_abstol, str, par_was_specified, 16);
    }
    else if (!strcmp( str, "angle_step")) {
        read_doublepar( fp, &par->angle_step, str, par_was_specified, 17);
        par->angle_step *= D2R;
    }
    else if (!strcmp( str, "angle_tol")) {
        read_doublepar( fp, &par->angle_tol, str, par_was_specified, 18);
    }
    else if (!strcmp( str, "angle_abstol")) {
        read_doublepar( fp, &par->angle_abstol, str, par_was_specified, 19);
        par->angle_abstol *= D2R;
    }
    else if (!strcmp( str, "spin_step")) {
        read_doublepar( fp, &par->spin_step, str, par_was_specified, 20);
        par->spin_step *= D2R;
    }
    else if (!strcmp( str, "spin_tol")) {
        read_doublepar( fp, &par->spin_tol, str, par_was_specified, 21);
    }
    else if (!strcmp( str, "spin_abstol")) {
        read_doublepar( fp, &par->spin_abstol, str, par_was_specified, 22);
        par->spin_abstol *= D2R;
    }
    else if (!strcmp( str, "spindot_step")) {
        read_doublepar( fp, &par->spindot_step, str, par_was_specified, 23);
        par->spindot_step *= D2R;
    }
    else if (!strcmp( str, "spindot_tol")) {
        read_doublepar( fp, &par->spindot_tol, str, par_was_specified, 24);
    }
    else if (!strcmp( str, "spindot_abstol")) {
        read_doublepar( fp, &par->spindot_abstol, str, par_was_specified, 25);
        par->spindot_abstol *= D2R;
    }
    else if (!strcmp( str, "photo_step")) {
        read_doublepar( fp, &par->photo_step, str, par_was_specified, 26);
    }
    else if (!strcmp( str, "photo_tol")) {
        read_doublepar( fp, &par->photo_tol, str, par_was_specified, 27);
    }
    else if (!strcmp( str, "photo_abstol")) {
        read_doublepar( fp, &par->photo_abstol, str, par_was_specified, 28);
    }
    else if (!strcmp( str, "inertia_step")) {
        read_doublepar( fp, &par->inertia_step, str, par_was_specified, 29);
    }
    else if (!strcmp( str, "inertia_tol")) {
        read_doublepar( fp, &par->inertia_tol, str, par_was_specified, 30);
    }
    else if (!strcmp( str, "inertia_abstol")) {
        read_doublepar( fp, &par->inertia_abstol, str, par_was_specified, 31);
    }
    else if (!strcmp( str, "delcor_step")) {
        n_read = read_doublevecpar( fp, par->delcor_step, str, MAXDELCORPAR,
                                    par_was_specified, 32);
        for (i=n_read; i<MAXDELCORPAR; i++)
          par->delcor_step[i] = par->delcor_step[0];
    }
    else if (!strcmp( str, "delcor_tol")) {
        read_doublepar( fp, &par->delcor_tol, str, par_was_specified, 33);
    }
    else if (!strcmp( str, "delcor_abstol")) {
        n_read = read_doublevecpar( fp, par->delcor_abstol, str, MAXDELCORPAR,
                                    par_was_specified, 34);
        for (i=n_read; i<MAXDELCORPAR; i++)
          par->delcor_abstol[i] = par->delcor_abstol[0];
    }
    else if (!strcmp( str, "xyoff_step")) {
        read_doublepar( fp, &par->xyoff_step, str, par_was_specified, 35);
    }
    else if (!strcmp( str, "xyoff_tol")) {
        read_doublepar( fp, &par->xyoff_tol, str, par_was_specified, 36);
    }
    else if (!strcmp( str, "xyoff_abstol")) {
        read_doublepar( fp, &par->xyoff_abstol, str, par_was_specified, 37);
    }
    else if (!strcmp( str, "term_prec")) {
        read_doublepar( fp, &par->term_prec, str, par_was_specified, 38);
    }
    else if (!strcmp( str, "dd_resid")) {
        read_doublepar( fp, &par->dd_resid, str, par_was_specified, 39);
    }
    else if (!strcmp( str, "optposmax")) {
        read_doublepar( fp, &par->optposmax, str, par_was_specified, 40);
    }
    else if (!strcmp( str, "radposmax")) {
        read_doublepar( fp, &par->radposmax, str, par_was_specified, 41);
    }
    else if (!strcmp( str, "pa_ephfile")) {
        read_stringpar( fp, par->pa_ephfile, str, par_was_specified, 42);
    }
    else if (!strcmp( str, "pa_ephformat")) {
        const char *options[]   = {"horizons", "datataking"};
        unsigned char parvals[] = { HORIZONS,   DATATAKING };
        read_optionspar( fp, &par->pa_ephformat, str, options, parvals, 2, par_was_specified, 43);
    }
    else if (!strcmp( str, "pa_startepoch")) {
        read_jdpar( fp, &par->pa_startepoch, str, par_was_specified, 44);
    }
    else if (!strcmp( str, "pa_stopepoch")) {
        read_jdpar( fp, &par->pa_stopepoch, str, par_was_specified, 45);
    }
    else if (!strcmp( str, "pa_angleoff")) {
        n_read = read_doublevecpar( fp, par->pa_angleoff, str, 3, par_was_specified, 46);
        if (n_read != 3)
          bailout("read_par.c: pa_angleoff must have 3 components\n");
    }
    else if (!strcmp( str, "pa_omegaoff")) {
        n_read = read_doublevecpar( fp, par->pa_omegaoff, str, 3, par_was_specified, 47);
        if (n_read != 3)
          bailout("read_par.c: pa_omegaoff must have 3 components\n");
    }
    else if (!strcmp( str, "nsinc2")) {
        read_intpar( fp, &par->nsinc2, str, par_was_specified, 48);
    }
    else if (!strcmp( str, "sinc2width")) {
        read_intpar( fp, &par->sinc2width, str, par_was_specified, 49);
    }
    else if (!strcmp( str, "scalefitobs")) {
        const char *options[]   = {"separate",       "maxfitobs",     "fit",     "obs"    };
        unsigned char parvals[] = {SCALE_SEPARATELY, SCALE_MAXFITOBS, SCALE_FIT, SCALE_OBS};
        read_optionspar( fp, &par->scalefitobs, str, options, parvals, 4, par_was_specified, 50);
    }
    else if (!strcmp( str, "listres")) {
        read_yesnopar( fp, &par->listres, str, par_was_specified, 51);
    }
    else if (!strcmp( str, "dd_maxsides")) {
        check_if_already_read( par_was_specified, NPARAMS, str, 52);
        gettstr( fp, str);
        if (!strcmp( str, "right-top")) {
            par->dd_clockwiserot = 0;
            par->dd_xflip = 0;
            par->dd_yflip = 0;
        } else if (!strcmp( str, "right-bottom")) {
            par->dd_clockwiserot = 0;
            par->dd_xflip = 0;
            par->dd_yflip = 1;
        } else if (!strcmp( str, "left-top")) {
            par->dd_clockwiserot = 0;
            par->dd_xflip = 1;
            par->dd_yflip = 0;
        } else if (!strcmp( str, "left-bottom")) {
            par->dd_clockwiserot = 0;
            par->dd_xflip = 1;
            par->dd_yflip = 1;
        } else if (!strcmp( str, "top-right")) {
            par->dd_clockwiserot = 1;
            par->dd_xflip = 0;
            par->dd_yflip = 1;
        } else if (!strcmp( str, "top-left")) {
            par->dd_clockwiserot = 1;
            par->dd_xflip = 1;
            par->dd_yflip = 1;
        } else if (!strcmp( str, "bottom-right")) {
            par->dd_clockwiserot = 1;
            par->dd_xflip = 0;
            par->dd_yflip = 0;
        } else if (!strcmp( str, "bottom-left")) {
            par->dd_clockwiserot = 1;
            par->dd_xflip = 1;
            par->dd_yflip = 0;
        } else
            bailout("read_par.c: unrecognized value for dd_maxsides\n");
        printf("# dd_maxsides %s\n", str);
    }
    else if (!strcmp( str, "delcorthresh")) {
        read_doublepar( fp, &par->delcorthresh, str, par_was_specified, 53);
    }
    else if (!strcmp( str, "jdoffset")) {
        read_jdpar( fp, &par->jdoffset, str, par_was_specified, 54);
    }
    else if (!strcmp( str, "perform_ltc")) {
        read_yesnopar( fp, &par->perform_ltc, str, par_was_specified, 55);
    }
    else if (!strcmp( str, "convex_file")) {
        read_stringpar( fp, par->convex_file, str, par_was_specified, 56);
    }
    else if (!strcmp( str, "plot_spinvec")) {
        read_yesnopar( fp, &par->plot_spinvec, str, par_was_specified, 57);
    }
    else if (!strcmp( str, "plot_subradar")) {
        read_yesnopar( fp, &par->plot_subradar, str, par_was_specified, 58);
    }
    else if (!strcmp( str, "plot_com")) {
        read_yesnopar( fp, &par->plot_com, str, par_was_specified, 59);
    }
    else if (!strcmp( str, "plot_pa")) {
        n_read = read_yesnovecpar( fp, par->plot_pa, str, 3, par_was_specified, 60);
        if (n_read != 3)
          bailout("read_par.c: plot_pa must have 3 components\n");
    }
    else if (!strcmp( str, "write_obs")) {
        read_yesnopar( fp, &par->write_obs, str, par_was_specified, 61);
    }
    else if (!strcmp( str, "delcor_read")) {
        read_yesnopar( fp, &par->delcor_read, str, par_was_specified, 62);
    }
    else if (!strcmp( str, "delcor_file")) {
        read_stringpar( fp, par->delcor_file, str, par_was_specified, 63);
    }
    else if (!strcmp( str, "first_fitpar")) {
        read_intpar( fp, &par->first_fitpar, str, par_was_specified, 64);
    }
    else if (!strcmp( str, "view_long")) {
        read_doublepar( fp, &par->view_long, str, par_was_specified, 65);
        par->view_long *= D2R;
    }
    else if (!strcmp( str, "view_lat")) {
        read_doublepar( fp, &par->view_lat, str, par_was_specified, 66);
        par->view_lat *= D2R;
    }
    else if (!strcmp( str, "listfit")) {
        read_yesnopar( fp, &par->listfit, str, par_was_specified, 67);
    }
    else if (!strcmp( str, "listfit_path")) {
        read_stringpar( fp, par->listfit_path, str, par_was_specified, 68);
    }
    else if (!strcmp( str, "area_latincr")) {
        read_doublepar( fp, &par->area_latincr, str, par_was_specified, 69);
        par->area_latincr *= D2R;
    }
    else if (!strcmp( str, "area_longincr")) {
        read_doublepar( fp, &par->area_longincr, str, par_was_specified, 70);
        par->area_longincr *= D2R;
    }
    else if (!strcmp( str, "poset_scaling")) {
        const char *options[]   = {"none", "block", "bilinear", "bicubic", "cubicconv"};
        unsigned char parvals[] = { NONE,   BLOCK,   BILINEAR,   BICUBIC,   CUBICCONV };
        read_optionspar( fp, &par->poset_scaling, str, options, parvals, 5, par_was_specified, 71);
    }
    else if (!strcmp( str, "poset_resample")) {
        const char *options[]   = {"block", "bilinear", "bicubic", "cubicconv"};
        unsigned char parvals[] = { BLOCK,   BILINEAR,   BICUBIC,   CUBICCONV };
        read_optionspar( fp, &par->poset_resample, str, options, parvals, 4, par_was_specified, 72);
    }
    else if (!strcmp( str, "image_rebin")) {
        read_yesnopar( fp, &par->image_rebin, str, par_was_specified, 73);
    }
    else if (!strcmp( str, "poset_resid")) {
        read_doublepar( fp, &par->poset_resid, str, par_was_specified, 74);
    }
    else if (!strcmp( str, "list_posetcal")) {
        read_yesnopar( fp, &par->list_posetcal, str, par_was_specified, 75);
    }
    else if (!strcmp( str, "delta_rotphase")) {
        read_doublepar( fp, &par->delta_rotphase, str, par_was_specified, 76);
        par->delta_rotphase *= D2R;
    }
    else if (!strcmp( str, "endian")) {
        const char *options[]   = {"big",           "little"};
        unsigned char parvals[] = {BIG_ENDIAN_DATA, LITTLE_ENDIAN_DATA};
        read_optionspar( fp, &par->endian, str, options, parvals, 2, par_was_specified, 77);
    }
    else if (!strcmp( str, "mark_unseen")) {
        read_yesnopar( fp, &par->mark_unseen, str, par_was_specified, 78);
    }
    else if (!strcmp( str, "maxangle_seen")) {
        read_doublepar( fp, &maxangle_seen, str, par_was_specified, 79);
        if (maxangle_seen < 0.0 || maxangle_seen > 90.0)
          bailout("read_par.c: must have 0.0 <= maxangle_seen <= 90.0 deg\n");
        par->mincosine_seen = cos(maxangle_seen*D2R);
    }
    else if (!strcmp( str, "lcrv_writeall")) {
        read_yesnopar( fp, &par->lcrv_writeall, str, par_was_specified, 80);
    }
    else if (!strcmp( str, "exclude_seen")) {
        read_intpar( fp, &par->exclude_seen, str, par_was_specified, 81);
    }
    else if (!strcmp( str, "pa_scatlaw")) {
        const char *options[]   = {"optical",   "radar",   "lambertian"};
        unsigned char parvals[] = {OPTICALVIEW, RADARVIEW, LAMBERTVIEW };
        read_optionspar( fp, &par->pa_scatlaw, str, options, parvals, 3, par_was_specified, 82);
    }
    else if (!strcmp( str, "view_scatlaw")) {
        const char *options[]   = {"optical",   "radar",   "lambertian"};
        unsigned char parvals[] = {OPTICALVIEW, RADARVIEW, LAMBERTVIEW };
        read_optionspar( fp, &par->view_scatlaw, str, options, parvals, 3, par_was_specified, 83);
    }
    else if (!strcmp( str, "sky_radlaw")) {
        const char *options[]   = {"radar",   "lambertian"};
        unsigned char parvals[] = {RADARVIEW, LAMBERTVIEW };
        read_optionspar( fp, &par->sky_radlaw, str, options, parvals, 2, par_was_specified, 84);
    }
    else if (!strcmp( str, "sky_optlaw")) {
        const char *options[]   = {"optical",   "lambertian"};
        unsigned char parvals[] = {OPTICALVIEW, LAMBERTVIEW };
        read_optionspar( fp, &par->sky_optlaw, str, options, parvals, 2, par_was_specified, 85);
    }
    else if (!strcmp( str, "radfacets")) {
        read_yesnopar( fp, &par->radfacets, str, par_was_specified, 86);
    }
    else if (!strcmp( str, "rad_global_R_state")) {
        read_statepar( fp, &par->rad_global_R_state, str, 0, par_was_specified, 87);
    }
    else if (!strcmp( str, "rad_global_C_state")) {
        read_statepar( fp, &par->rad_global_C_state, str, 0, par_was_specified, 88);
    }
    else if (!strcmp( str, "rad_local_R_state")) {
        read_statepar( fp, &par->rad_local_R_state, str, 1, par_was_specified, 89);
    }
    else if (!strcmp( str, "rad_local_C_state")) {
        read_statepar( fp, &par->rad_local_C_state, str, 1, par_was_specified, 90);
    }
    else if (!strcmp( str, "optfacets")) {
        read_yesnopar( fp, &par->optfacets, str, par_was_specified, 91);
    }
    else if (!strcmp( str, "opt_global_R_state")) {
        read_statepar( fp, &par->opt_global_R_state, str, 0, par_was_specified, 92);
    }
    else if (!strcmp( str, "opt_global_wt_state")) {
        read_statepar( fp, &par->opt_global_wt_state, str, 0, par_was_specified, 93);
    }
    else if (!strcmp( str, "opt_global_A0_state")) {
        read_statepar( fp, &par->opt_global_A0_state, str, 0, par_was_specified, 94);
    }
    else if (!strcmp( str, "opt_global_D_state")) {
        read_statepar( fp, &par->opt_global_D_state, str, 0, par_was_specified, 95);
    }
    else if (!strcmp( str, "opt_global_k_state")) {
        read_statepar( fp, &par->opt_global_k_state, str, 0, par_was_specified, 96);
    }
    else if (!strcmp( str, "opt_global_w_state")) {
        read_statepar( fp, &par->opt_global_w_state, str, 0, par_was_specified, 97);
    }
    else if (!strcmp( str, "opt_global_h_state")) {
        read_statepar( fp, &par->opt_global_h_state, str, 0, par_was_specified, 98);
    }
    else if (!strcmp( str, "opt_global_B0_state")) {
        read_statepar( fp, &par->opt_global_B0_state, str, 0, par_was_specified, 99);
    }
    else if (!strcmp( str, "opt_global_g_state")) {
        read_statepar( fp, &par->opt_global_g_state, str, 0, par_was_specified, 100);
    }
    else if (!strcmp( str, "opt_global_theta_state")) {
        read_statepar( fp, &par->opt_global_theta_state, str, 0, par_was_specified, 101);
    }
    else if (!strcmp( str, "opt_local_R_state")) {
        read_statepar( fp, &par->opt_local_R_state, str, 1, par_was_specified, 102);
    }
    else if (!strcmp( str, "opt_local_wt_state")) {
        read_statepar( fp, &par->opt_local_wt_state, str, 1, par_was_specified, 103);
    }
    else if (!strcmp( str, "opt_local_A0_state")) {
        read_statepar( fp, &par->opt_local_A0_state, str, 1, par_was_specified, 104);
    }
    else if (!strcmp( str, "opt_local_D_state")) {
        read_statepar( fp, &par->opt_local_D_state, str, 1, par_was_specified, 105);
    }
    else if (!strcmp( str, "opt_local_k_state")) {
        read_statepar( fp, &par->opt_local_k_state, str, 1, par_was_specified, 106);
    }
    else if (!strcmp( str, "opt_local_w_state")) {
        read_statepar( fp, &par->opt_local_w_state, str, 1, par_was_specified, 107);
    }
    else if (!strcmp( str, "opt_local_h_state")) {
        read_statepar( fp, &par->opt_local_h_state, str, 1, par_was_specified, 108);
    }
    else if (!strcmp( str, "opt_local_B0_state")) {
        read_statepar( fp, &par->opt_local_B0_state, str, 1, par_was_specified, 109);
    }
    else if (!strcmp( str, "opt_local_g_state")) {
        read_statepar( fp, &par->opt_local_g_state, str, 1, par_was_specified, 110);
    }
    else if (!strcmp( str, "opt_local_theta_state")) {
        read_statepar( fp, &par->opt_local_theta_state, str, 1, par_was_specified, 111);
    }
    else if (!strcmp( str, "radharm")) {
        read_yesnopar( fp, &par->radharm, str, par_was_specified, 112);
    }
    else if (!strcmp( str, "rad_R_nharm")) {
        read_intpar( fp, &par->rad_R_nharm, str, par_was_specified, 113);
    }
    else if (!strcmp( str, "rad_C_nharm")) {
        read_intpar( fp, &par->rad_C_nharm, str, par_was_specified, 114);
    }
    else if (!strcmp( str, "optharm")) {
        read_yesnopar( fp, &par->optharm, str, par_was_specified, 115);
    }
    else if (!strcmp( str, "opt_R_nharm")) {
        read_intpar( fp, &par->opt_R_nharm, str, par_was_specified, 116);
    }
    else if (!strcmp( str, "opt_wt_nharm")) {
        read_intpar( fp, &par->opt_wt_nharm, str, par_was_specified, 117);
    }
    else if (!strcmp( str, "opt_A0_nharm")) {
        read_intpar( fp, &par->opt_A0_nharm, str, par_was_specified, 118);
    }
    else if (!strcmp( str, "opt_D_nharm")) {
        read_intpar( fp, &par->opt_D_nharm, str, par_was_specified, 119);
    }
    else if (!strcmp( str, "opt_k_nharm")) {
        read_intpar( fp, &par->opt_k_nharm, str, par_was_specified, 120);
    }
    else if (!strcmp( str, "opt_w_nharm")) {
        read_intpar( fp, &par->opt_w_nharm, str, par_was_specified, 121);
    }
    else if (!strcmp( str, "opt_h_nharm")) {
        read_intpar( fp, &par->opt_h_nharm, str, par_was_specified, 122);
    }
    else if (!strcmp( str, "opt_B0_nharm")) {
        read_intpar( fp, &par->opt_B0_nharm, str, par_was_specified, 123);
    }
    else if (!strcmp( str, "opt_g_nharm")) {
        read_intpar( fp, &par->opt_g_nharm, str, par_was_specified, 124);
    }
    else if (!strcmp( str, "opt_theta_nharm")) {
        read_intpar( fp, &par->opt_theta_nharm, str, par_was_specified, 125);
    }
    else if (!strcmp( str, "view_shadows")) {
        read_yesnopar( fp, &par->view_shadows, str, par_was_specified, 126);
    }
    else if (!strcmp( str, "view_sunlong")) {
        read_doublepar( fp, &par->view_sunlong, str, par_was_specified, 127);
        par->view_sunlong *= D2R;
    }
    else if (!strcmp( str, "view_sunlat")) {
        read_doublepar( fp, &par->view_sunlat, str, par_was_specified, 128);
        par->view_sunlat *= D2R;
    }
    else if (!strcmp( str, "sun_appmag")) {
    	n_read = read_doublevecpar( fp, (*par).sun_appmag, str, MAXSUNMAGS,
    			par_was_specified, 129);
    	for (i=n_read; i<MAXSUNMAGS; i++)
    		(*par).sun_appmag[i] = (*par).sun_appmag[0];
    }
    else if (!strcmp( str, "rad_R_min")) {
        read_minmaxpar( fp, &par->rad_R_min, str, 0, par_was_specified, 130);
    }
    else if (!strcmp( str, "rad_R_max")) {
        read_minmaxpar( fp, &par->rad_R_max, str, 1, par_was_specified, 131);
    }
    else if (!strcmp( str, "rad_C_min")) {
        read_minmaxpar( fp, &par->rad_C_min, str, 0, par_was_specified, 132);
    }
    else if (!strcmp( str, "rad_C_max")) {
        read_minmaxpar( fp, &par->rad_C_max, str, 1, par_was_specified, 133);
    }
    else if (!strcmp( str, "opt_R_min")) {
        read_minmaxpar( fp, &par->opt_R_min, str, 0, par_was_specified, 134);
    }
    else if (!strcmp( str, "opt_R_max")) {
        read_minmaxpar( fp, &par->opt_R_max, str, 1, par_was_specified, 135);
    }
    else if (!strcmp( str, "opt_wt_min")) {
        read_minmaxpar( fp, &par->opt_wt_min, str, 0, par_was_specified, 136);
    }
    else if (!strcmp( str, "opt_wt_max")) {
        read_minmaxpar( fp, &par->opt_wt_max, str, 1, par_was_specified, 137);
    }
    else if (!strcmp( str, "opt_A0_min")) {
        read_minmaxpar( fp, &par->opt_A0_min, str, 0, par_was_specified, 138);
    }
    else if (!strcmp( str, "opt_A0_max")) {
        read_minmaxpar( fp, &par->opt_A0_max, str, 1, par_was_specified, 139);
    }
    else if (!strcmp( str, "opt_D_min")) {
        read_minmaxpar( fp, &par->opt_D_min, str, 0, par_was_specified, 140);
    }
    else if (!strcmp( str, "opt_D_max")) {
        read_minmaxpar( fp, &par->opt_D_max, str, 1, par_was_specified, 141);
    }
    else if (!strcmp( str, "opt_k_min")) {
        read_minmaxpar( fp, &par->opt_k_min, str, 0, par_was_specified, 142);
    }
    else if (!strcmp( str, "opt_k_max")) {
        read_minmaxpar( fp, &par->opt_k_max, str, 1, par_was_specified, 143);
    }
    else if (!strcmp( str, "opt_w_min")) {
        read_minmaxpar( fp, &par->opt_w_min, str, 0, par_was_specified, 144);
    }
    else if (!strcmp( str, "opt_w_max")) {
        read_minmaxpar( fp, &par->opt_w_max, str, 1, par_was_specified, 145);
    }
    else if (!strcmp( str, "opt_h_min")) {
        read_minmaxpar( fp, &par->opt_h_min, str, 0, par_was_specified, 146);
    }
    else if (!strcmp( str, "opt_h_max")) {
        read_minmaxpar( fp, &par->opt_h_max, str, 1, par_was_specified, 147);
    }
    else if (!strcmp( str, "opt_B0_min")) {
        read_minmaxpar( fp, &par->opt_B0_min, str, 0, par_was_specified, 148);
    }
    else if (!strcmp( str, "opt_B0_max")) {
        read_minmaxpar( fp, &par->opt_B0_max, str, 1, par_was_specified, 149);
    }
    else if (!strcmp( str, "opt_g_min")) {
        read_minmaxpar( fp, &par->opt_g_min, str, 0, par_was_specified, 150);
    }
    else if (!strcmp( str, "opt_g_max")) {
        read_minmaxpar( fp, &par->opt_g_max, str, 1, par_was_specified, 151);
    }
    else if (!strcmp( str, "opt_theta_min")) {
        read_minmaxpar( fp, &par->opt_theta_min, str, 0, par_was_specified, 152);
    }
    else if (!strcmp( str, "opt_theta_max")) {
        read_minmaxpar( fp, &par->opt_theta_max, str, 1, par_was_specified, 153);
    }
    else if (!strcmp( str, "ratio_step")) {
        read_doublepar( fp, &par->ratio_step, str, par_was_specified, 154);
    }
    else if (!strcmp( str, "ratio_tol")) {
        read_doublepar( fp, &par->ratio_tol, str, par_was_specified, 155);
    }
    else if (!strcmp( str, "ratio_abstol")) {
        read_doublepar( fp, &par->ratio_abstol, str, par_was_specified, 156);
    }
    else if (!strcmp( str, "vary_delcor0")) {
        const char *options[]   = {"none",    "no",      "size",    "all",    "yes"   };
        unsigned char parvals[] = {VARY_NONE, VARY_NONE, VARY_SIZE, VARY_ALL, VARY_ALL};
        read_optionspar( fp, &par->vary_delcor0, str, options, parvals, 5, par_was_specified, 157);
    }
    else if (!strcmp( str, "vary_radalb")) {
        const char *options[]   = {"none",    "no",      "size",    "all",    "yes"   };
        unsigned char parvals[] = {VARY_NONE, VARY_NONE, VARY_SIZE, VARY_ALL, VARY_ALL};
        read_optionspar( fp, &par->vary_radalb, str, options, parvals, 5, par_was_specified, 158);
    }
    else if (!strcmp( str, "vary_optalb")) {
        const char *options[]   = {"none",    "no",      "size",    "all",    "yes"   };
        unsigned char parvals[] = {VARY_NONE, VARY_NONE, VARY_SIZE, VARY_ALL, VARY_ALL};
        read_optionspar( fp, &par->vary_optalb, str, options, parvals, 5, par_was_specified, 159);
    }
    else if (!strcmp( str, "slice_long")) {
        read_doublepar( fp, &par->slice_long, str, par_was_specified, 160);
        par->slice_long *= D2R;
    }
    else if (!strcmp( str, "slice_lat")) {
        read_doublepar( fp, &par->slice_lat, str, par_was_specified, 161);
        par->slice_lat *= D2R;
    }
    else if (!strcmp( str, "slice_offset")) {
        read_doublepar( fp, &par->slice_offset, str, par_was_specified, 162);
    }
    else if (!strcmp( str, "slice_viewlong")) {
        read_doublepar( fp, &par->slice_viewlong, str, par_was_specified, 163);
        par->slice_viewlong *= D2R;
    }
    else if (!strcmp( str, "slice_viewlat")) {
        read_doublepar( fp, &par->slice_viewlat, str, par_was_specified, 164);
        par->slice_viewlat *= D2R;
    }
    else if (!strcmp( str, "slice_sunlong")) {
        read_doublepar( fp, &par->slice_sunlong, str, par_was_specified, 165);
        par->slice_sunlong *= D2R;
    }
    else if (!strcmp( str, "slice_sunlat")) {
        read_doublepar( fp, &par->slice_sunlat, str, par_was_specified, 166);
        par->slice_sunlat *= D2R;
    }
    else if (!strcmp( str, "slice_scatlaw")) {
        const char *options[]   = {"optical",   "radar",   "lambertian"};
        unsigned char parvals[] = {OPTICALVIEW, RADARVIEW, LAMBERTVIEW };
        read_optionspar( fp, &par->slice_scatlaw, str, options, parvals, 3, par_was_specified, 167);
    }
    else if (!strcmp( str, "slice_shadows")) {
        read_yesnopar( fp, &par->slice_shadows, str, par_was_specified, 168);
    }
    else if (!strcmp( str, "slice_planefrac")) {
        read_doublepar( fp, &par->slice_planefrac, str, par_was_specified, 169);
    }
    else if (!strcmp( str, "slice_skyfactor")) {
        read_doublepar( fp, &par->slice_skyfactor, str, par_was_specified, 170);
    }
    else if (!strcmp( str, "slice_dimfactor")) {
        read_doublepar( fp, &par->slice_dimfactor, str, par_was_specified, 171);
    }
    else if (!strcmp( str, "slice_read")) {
        read_yesnopar( fp, &par->slice_read, str, par_was_specified, 172);
    }
    else if (!strcmp( str, "slice_dointerp")) {
        read_yesnopar( fp, &par->slice_dointerp, str, par_was_specified, 173);
    }
    else if (!strcmp( str, "semimajor_axis")) {
        n_read = read_doublevecpar( fp, par->semimajor_axis, str, 2, par_was_specified, 174);
        check_triple( n_read, 2, &par->is_triple);
    }
    else if (!strcmp( str, "r_pericenter")) {
        n_read = read_doublevecpar( fp, par->r_pericenter, str, 2, par_was_specified, 175);
        check_triple( n_read, 2, &par->is_triple);
    }
    else if (!strcmp( str, "eccentricity")) {
        n_read = read_doublevecpar( fp, par->eccentricity, str, 2, par_was_specified, 176);
        check_triple( n_read, 2, &par->is_triple);
    }
    else if (!strcmp( str, "t_pericenter")) {
        n_read = read_jdvecpar( fp, par->t_pericenter, str, 2, par_was_specified, 177);
        check_triple( n_read, 2, &par->is_triple);
    }
    else if (!strcmp( str, "long_asc_node")) {
        n_read = read_doublevecpar( fp, par->long_asc_node, str, 2, par_was_specified, 178);
        check_triple( n_read, 2, &par->is_triple);
        for (i=0; i<n_read; i++)
          par->long_asc_node[i] *= D2R;
    }
    else if (!strcmp( str, "inclination")) {
        n_read = read_doublevecpar( fp, par->inclination, str, 2, par_was_specified, 179);
        check_triple( n_read, 2, &par->is_triple);
        for (i=0; i<n_read; i++)
          par->inclination[i] *= D2R;
    }
    else if (!strcmp( str, "arg_pericenter")) {
        n_read = read_doublevecpar( fp, par->arg_pericenter, str, 2, par_was_specified, 180);
        check_triple( n_read, 2, &par->is_triple);
        for (i=0; i<n_read; i++)
          par->arg_pericenter[i] *= D2R;
    }
    else if (!strcmp( str, "binary_totmass")) {
        n_read = read_doublevecpar( fp, par->binary_gravparam, str, 2, par_was_specified, 181);
        check_triple( n_read, 2, &par->is_triple);
        for (i=0; i<n_read; i++)
          par->binary_gravparam[i] *= GRAVCONST;
    }
    else if (!strcmp( str, "orbit_period")) {
        n_read = read_doublevecpar( fp, par->orbit_period, str, 2, par_was_specified, 182);
        check_triple( n_read, 2, &par->is_triple);
    }
    else if (!strcmp( str, "orbit_reflex")) {
        read_yesnopar( fp, &par->orbit_reflex, str, par_was_specified, 183);
    }
    else if (!strcmp( str, "orbit_posfactor")) {
        n_read = read_doublevecpar( fp, par->orbit_posfactor, str, 3, par_was_specified, 184);
        if (n_read == 1)
          bailout("read_par.c: orbit_posfactor must have 2 or 3 components\n");
        check_triple( n_read, 3, &par->is_triple);
    }
    else if (!strcmp( str, "write_posfactor")) {
        read_doublepar( fp, &par->write_posfactor, str, par_was_specified, 185);
    }
    else if (!strcmp( str, "radfitmax")) {
        read_doublepar( fp, &par->radfitmax, str, par_was_specified, 186);
    }
    else if (!strcmp( str, "radobsmax")) {
        read_doublepar( fp, &par->radobsmax, str, par_was_specified, 187);
    }
    else if (!strcmp( str, "term_badmodel")) {
        read_yesnopar( fp, &par->term_badmodel, str, par_was_specified, 188);
    }
    else if (!strcmp( str, "avoid_badpos")) {
        read_yesnopar( fp, &par->avoid_badpos, str, par_was_specified, 189);
    }
    else if (!strcmp( str, "bad_objfactor")) {
        read_doublepar( fp, &par->bad_objfactor, str, par_was_specified, 190);
    }
    else if (!strcmp( str, "maskdir")) {
        read_stringpar( fp, par->maskdir, str, par_was_specified, 191);
        if (!strcmp( par->maskdir, "\"\"") || !strcmp( par->maskdir, "\'\'"))
          strcpy(par->maskdir, "");
    }
    else if (!strcmp( str, "write_chi2fit0")) {
        read_yesnopar( fp, &par->write_chi2fit0, str, par_was_specified, 192);
    }
    else if (!strcmp( str, "warn_badradar")) {
        read_yesnopar( fp, &par->warn_badradar, str, par_was_specified, 193);
    }
    else if (!strcmp( str, "plot_angmom")) {
        read_yesnopar( fp, &par->plot_angmom, str, par_was_specified, 194);
    }
    else if (!strcmp( str, "pierce_spinvec")) {
        read_yesnopar( fp, &par->pierce_spinvec, str, par_was_specified, 195);
    }
    else if (!strcmp( str, "pierce_angmom")) {
        read_yesnopar( fp, &par->pierce_angmom, str, par_was_specified, 196);
    }
    else if (!strcmp( str, "slice_posfactor")) {
        read_doublepar( fp, &par->slice_posfactor, str, par_was_specified, 197);
    }
    else if (!strcmp( str, "view_posfactor")) {
        read_doublepar( fp, &par->view_posfactor, str, par_was_specified, 198);
    }
    else if (!strcmp( str, "npar_update")) {
        read_intpar( fp, &par->npar_update, str, par_was_specified, 199);
    }
    else if (!strcmp( str, "rad_rho_min")) {
        read_minmaxpar( fp, &par->rad_rho_min, str, 0, par_was_specified, 200);
    }
    else if (!strcmp( str, "rad_rho_max")) {
        read_minmaxpar( fp, &par->rad_rho_max, str, 1, par_was_specified, 201);
    }
    else if (!strcmp( str, "split_plane")) {
        n_read = read_doublevecpar( fp, split_plane, str, 4, par_was_specified, 202);
        if (n_read != 4)
          bailout("read_par.c: split_plane must have 4 components\n");
        normconst = sqrt(split_plane[0]*split_plane[0] + split_plane[1]*split_plane[1]
                                                       + split_plane[2]*split_plane[2]);
        for (i=0; i<=2; i++)
          par->split_norm[i] = split_plane[i] / normconst;
        par->split_const = split_plane[3] / normconst;
    }
    else if (!strcmp( str, "map_set")) {
        read_intpar( fp, &par->map_set, str, par_was_specified, 203);
    }
    else if (!strcmp( str, "map_frame")) {
        read_intpar( fp, &par->map_frame, str, par_was_specified, 204);
    }
    else if (!strcmp( str, "map_mode")) {
        const char *options[]   = {"deldop",       "pos",       "facets"      };
        unsigned char parvals[] = {MAPMODE_DELDOP, MAPMODE_POS, MAPMODE_FACETS};
        read_optionspar( fp, &par->map_mode, str, options, parvals, 3, par_was_specified, 205);
    }
    else if (!strcmp( str, "map_dellim")) {
        n_read = read_intvecpar( fp, par->map_dellim, str, 2, par_was_specified, 206);
        if (n_read < 1 || n_read > 2)
          bailout("read_par.c: map_dellim must have 1 or 2 components\n");
        else if (n_read == 1)
          par->map_dellim[1] = par->map_dellim[0];
    }
    else if (!strcmp( str, "map_doplim")) {
        n_read = read_intvecpar( fp, par->map_doplim, str, 2, par_was_specified, 207);
        if (n_read < 1 || n_read > 2)
          bailout("read_par.c: map_doplim must have 1 or 2 components\n");
        else if (n_read == 1)
          par->map_doplim[1] = par->map_doplim[0];
    }
    else if (!strcmp( str, "map_xposlim")) {
        n_read = read_intvecpar( fp, par->map_xposlim, str, 2, par_was_specified, 208);
        if (n_read < 1 || n_read > 2)
          bailout("read_par.c: map_xposlim must have 1 or 2 components\n");
        else if (n_read == 1)
          par->map_xposlim[1] = par->map_xposlim[0];
    }
    else if (!strcmp( str, "map_yposlim")) {
        n_read = read_intvecpar( fp, par->map_yposlim, str, 2, par_was_specified, 209);
        if (n_read < 1 || n_read > 2)
          bailout("read_par.c: map_yposlim must have 1 or 2 components\n");
        else if (n_read == 1)
          par->map_yposlim[1] = par->map_yposlim[0];
    }
    else if (!strcmp( str, "map_posfactor")) {
        read_doublepar( fp, &par->map_posfactor, str, par_was_specified, 210);
    }
    else if (!strcmp( str, "map_fitcutoff")) {
        read_doublepar( fp, &par->map_fitcutoff, str, par_was_specified, 211);
    }
    else if (!strcmp( str, "map_poscutoff")) {
        read_doublepar( fp, &par->map_poscutoff, str, par_was_specified, 212);
    }
    else if (!strcmp( str, "map_comp")) {
        n_read = read_intvecpar( fp, par->map_comp, str, MAXCHOSENFACETS,
                                 par_was_specified, 213);
    }
    else if (!strcmp( str, "map_facet")) {
        n_read = read_intvecpar( fp, par->map_facet, str, MAXCHOSENFACETS,
                                 par_was_specified, 214);
    }
    else if (!strcmp( str, "map_verbose")) {
        read_yesnopar( fp, &par->map_verbose, str, par_was_specified, 215);
    }
    else if (!strcmp( str, "radfitmin")) {
        read_doublepar( fp, &par->radfitmin, str, par_was_specified, 216);
    }
    else if (!strcmp( str, "radobsmin")) {
        read_doublepar( fp, &par->radobsmin, str, par_was_specified, 217);
    }
    else if (!strcmp( str, "int_abstol")) {
        read_doublepar( fp, &par->int_abstol, str, par_was_specified, 218);
    }
    else if (!strcmp( str, "pa_bodycoords")) {
        read_yesnopar( fp, &par->pa_bodycoords, str, par_was_specified, 219);
    }
    else if (!strcmp( str, "pa_highlight")) {
        read_yesnopar( fp, &par->pa_highlight, str, par_was_specified, 220);
    }
    else if (!strcmp( str, "pa_comp")) {
        n_read = read_intvecpar( fp, par->pa_comp, str, MAXCHOSENFACETS,
                                 par_was_specified, 221);
    }
    else if (!strcmp( str, "pa_facet")) {
        n_read = read_intvecpar( fp, par->pa_facet, str, MAXCHOSENFACETS,
                                 par_was_specified, 222);
    }
    else if (!strcmp( str, "view_highlight")) {
        read_yesnopar( fp, &par->view_highlight, str, par_was_specified, 223);
    }
    else if (!strcmp( str, "view_comp")) {
        n_read = read_intvecpar( fp, par->view_comp, str, MAXCHOSENFACETS,
                                 par_was_specified, 224);
    }
    else if (!strcmp( str, "view_facet")) {
        n_read = read_intvecpar( fp, par->view_facet, str, MAXCHOSENFACETS,
                                 par_was_specified, 225);
    }
    else if (!strcmp( str, "vary_dopscale")) {
        const char *options[]   = {"none",    "no",      "spin",    "all",    "yes"   };
        unsigned char parvals[] = {VARY_NONE, VARY_NONE, VARY_SPIN, VARY_ALL, VARY_ALL};
        read_optionspar( fp, &par->vary_dopscale, str, options, parvals, 5,
                         par_was_specified, 226);
    }
    else if (!strcmp( str, "dopscale_min")) {
        read_minmaxpar( fp, &par->dopscale_min, str, 0, par_was_specified, 227);
    }
    else if (!strcmp( str, "dopscale_max")) {
        read_minmaxpar( fp, &par->dopscale_max, str, 1, par_was_specified, 228);
    }
    else if (!strcmp( str, "objfunc_start")) {
        read_doublepar( fp, &par->objfunc_start, str, par_was_specified, 229);
    }
    else if (!strcmp( str, "chi2fit0_thresh")) {
        read_doublepar( fp, &par->chi2fit0_thresh, str, par_was_specified, 230);
    }
    else if (!strcmp( str, "listpos_deldop")) {
        read_yesnopar( fp, &par->listpos_deldop, str, par_was_specified, 231);
    }
    else if (!strcmp( str, "listpos_opt")) {
        read_yesnopar( fp, &par->listpos_opt, str, par_was_specified, 232);
    }
    else if (!strcmp( str, "listpos_path")) {
        read_stringpar( fp, par->listpos_path, str, par_was_specified, 233);
    }
    else if (!strcmp( str, "term_maxiter")) {
        read_intpar( fp, &par->term_maxiter, str, par_was_specified, 234);
    }
    else if (!strcmp( str, "write_highlight")) {
        read_yesnopar( fp, &par->write_highlight, str, par_was_specified, 235);
    }
    else if (!strcmp( str, "write_comp")) {
        n_read = read_intvecpar( fp, par->write_comp, str, MAXCHOSENFACETS,
                                 par_was_specified, 236);
    }
    else if (!strcmp( str, "write_facet")) {
        n_read = read_intvecpar( fp, par->write_facet, str, MAXCHOSENFACETS,
                                 par_was_specified, 237);
    }
    else if (!strcmp( str, "delcor_verbose")) {
        read_yesnopar( fp, &par->delcor_verbose, str, par_was_specified, 238);
    }
    else if (!strcmp( str, "lib_amp_step")) {
        read_doublepar( fp, &par->lib_amp_step, str, par_was_specified, 239);
        par->lib_amp_step *= D2R;
    }
    else if (!strcmp( str, "lib_amp_tol")) {
        read_doublepar( fp, &par->lib_amp_tol, str, par_was_specified, 240);
    }
    else if (!strcmp( str, "lib_amp_abstol")) {
        read_doublepar( fp, &par->lib_amp_abstol, str, par_was_specified, 241);
        par->lib_amp_abstol *= D2R;
    }
    else if (!strcmp( str, "lib_freq_step")) {
        read_doublepar( fp, &par->lib_freq_step, str, par_was_specified, 242);
        par->lib_freq_step *= D2R;
    }
    else if (!strcmp( str, "lib_freq_tol")) {
        read_doublepar( fp, &par->lib_freq_tol, str, par_was_specified, 243);
    }
    else if (!strcmp( str, "lib_freq_abstol")) {
        read_doublepar( fp, &par->lib_freq_abstol, str, par_was_specified, 244);
        par->lib_freq_abstol *= D2R;
    }
    else if (!strcmp( str, "lib_phase_step")) {
        read_doublepar( fp, &par->lib_phase_step, str, par_was_specified, 245);
        par->lib_phase_step *= D2R;
    }
    else if (!strcmp( str, "lib_phase_tol")) {
        read_doublepar( fp, &par->lib_phase_tol, str, par_was_specified, 246);
    }
    else if (!strcmp( str, "lib_phase_abstol")) {
        read_doublepar( fp, &par->lib_phase_abstol, str, par_was_specified, 247);
        par->lib_phase_abstol *= D2R;
    }

    else if (strcmp( str, "PENALTIES")) {
        printf("????? %s ?????\n", str);
        bailout("read_par.c: don't recognize that parameter\n");
    }

  }  /* end while loop */

  /*  Check that all essential parameters have been input, and
      adjust the values of inessential parameters that were not
      input by setting them to their default values              */

  adjust_par( par);

  /*  Read in the penalty functions and penalty weights  */

  if (strcmp( str, "PENALTIES")) {
    printf("%s\n", str);
    bailout("read_par.c: expected keyword PENALTIES\n");
  }
  read_penalties( fp, par);

  /*  Close the parameter file  */

  fclose( fp);
  printf("# finished reading parameter file\n");
  fflush(stdout);
}


/******************************************************************/
/*  The following routine initializes all parameters              */
/******************************************************************/


void initialize_par( struct par_t *par)
{
  int i;

  /*  This block protects against these parameters being undefined  */

  par->speckle = 0;
  par->pos_smooth = 1;
  par->read_node = 0;
  par->pos_scope = GLOBAL;
  par->dd_scaling = NONE;
  par->int_method = PERIODIC;
  par->lcrv_pos = 0;
  par->theta_steps = 0;
  par->dd_gamma = 1.0;
  par->dd_resid = 5.0;
  par->optposmax = 0.0;
  par->radposmax = 0.0;
  par->pa_ephformat = HORIZONS;
  par->pa_startepoch = -1.0;
  par->pa_stopepoch = HUGENUMBER;
  for (i=0; i<3; i++) {
    par->pa_angleoff[i] = 0.0;
    par->pa_omegaoff[i] = 0.0;
  }
  par->scalefitobs = SCALE_SEPARATELY;
  par->listres = 0;
  par->jdoffset = 0.0;
  par->perform_ltc = 1;
  par->plot_spinvec = 0;
  par->plot_subradar = 0;
  par->plot_com = 0;
  for (i=0; i<3; i++)
    par->plot_pa[i] = 0;
  par->write_obs = 0;
  par->delcor_read = 0;
  par->first_fitpar = 0;
  par->view_long = 0.0;
  par->view_lat = PIE/2;
  par->view_shadows = 0;
  par->view_sunlong = 0.0;
  par->view_sunlat = PIE/2;
  par->listfit = 0;
  strcpy(par->listfit_path, "");
  par->area_latincr = 5.0*D2R;
  par->area_longincr = 5.0*D2R;
  par->poset_scaling = NONE;
  par->poset_resample = BILINEAR;
  par->image_rebin = 0;
  par->poset_resid = 5.0;
  par->list_posetcal = 0;
  par->delta_rotphase = 0.0;
  par->endian = BIG_ENDIAN_DATA;
  par->mark_unseen = 0;
  par->mincosine_seen = 0.0;
  par->lcrv_writeall = 0;
  par->exclude_seen = -1;
  par->pa_scatlaw = LAMBERTVIEW;
  par->view_scatlaw = LAMBERTVIEW;
  par->sky_radlaw = LAMBERTVIEW;
  par->sky_optlaw = OPTICALVIEW;
  par->radfacets = 1;
  par->rad_global_R_state = 'c';
  par->rad_global_C_state = 'c';
  par->rad_local_R_state = 'f';
  par->rad_local_C_state = 'f';
  par->optfacets = 1;
  par->opt_global_R_state = 'c';
  par->opt_global_wt_state = 'c';
  par->opt_global_A0_state = 'c';
  par->opt_global_D_state = 'c';
  par->opt_global_k_state = 'c';
  par->opt_global_w_state = 'c';
  par->opt_global_h_state = 'c';
  par->opt_global_B0_state = 'c';
  par->opt_global_g_state = 'c';
  par->opt_global_theta_state = 'c';
  par->opt_local_R_state = 'f';
  par->opt_local_wt_state = 'f';
  par->opt_local_A0_state = 'f';
  par->opt_local_D_state = 'f';
  par->opt_local_k_state = 'f';
  par->opt_local_w_state = 'f';
  par->opt_local_h_state = 'f';
  par->opt_local_B0_state = 'f';
  par->opt_local_g_state = 'f';
  par->opt_local_theta_state = 'f';
  par->radharm = 1;
  par->rad_R_nharm = 0;
  par->rad_C_nharm = 0;
  par->optharm = 1;
  par->opt_R_nharm = 0;
  par->opt_wt_nharm = 0;
  par->opt_A0_nharm = 0;
  par->opt_D_nharm = 0;
  par->opt_k_nharm = 0;
  par->opt_w_nharm = 0;
  par->opt_h_nharm = 0;
  par->opt_B0_nharm = 0;
  par->opt_g_nharm = 0;
  par->opt_theta_nharm = 0;
  for (i=0; i<MAXSUNMAGS; i++)
      (*par).sun_appmag[i] = SOLAR_VMAG;
  par->rad_R_min = 0.0;
  par->rad_R_max = HUGENUMBER;
  par->rad_C_min = 0.0;
  par->rad_C_max = HUGENUMBER;
  par->rad_rho_min = 0.0;
  par->rad_rho_max = HUGENUMBER;
  par->opt_R_min = 0.0;
  par->opt_R_max = HUGENUMBER;
  par->opt_wt_min = 0.0;
  par->opt_wt_max = 1.0;
  par->opt_A0_min = 0.0;
  par->opt_A0_max = HUGENUMBER;
  par->opt_D_min = 0.0;
  par->opt_D_max = HUGENUMBER;
  par->opt_k_min = -HUGENUMBER;
  par->opt_k_max = 0.0;
  par->opt_w_min = 0.0;
  par->opt_w_max = 1.0;
  par->opt_h_min = 0.0;
  par->opt_h_max = HUGENUMBER;
  par->opt_B0_min = 0.0;
  par->opt_B0_max = HUGENUMBER;
  par->opt_g_min = -1.0;
  par->opt_g_max = 1.0;
  par->opt_theta_min = 0.0;
  par->opt_theta_max = PIE/2;
  par->vary_delcor0 = VARY_NONE;
  par->vary_radalb = VARY_NONE;
  par->vary_optalb = VARY_NONE;
  par->slice_long = 0.0;
  par->slice_lat = PIE/2;
  par->slice_offset = 0.0;
  par->slice_sunlong = 0.0;
  par->slice_sunlat = PIE/2;
  par->slice_scatlaw = LAMBERTVIEW;
  par->slice_shadows = 0;
  par->slice_planefrac = 0.85;
  par->slice_skyfactor = 0.2;
  par->slice_dimfactor = 0.2;
  par->slice_read = 0;
  par->slice_dointerp = 1;
  par->orbit_reflex = 0;
  for (i=0; i<3; i++)
    par->orbit_posfactor[i] = 1.0;
  par->write_posfactor = 1.0;
  par->radfitmin = par->radfitmax = 0.0;
  par->radobsmin = par->radobsmax = 0.0;
  par->term_badmodel = 0;
  par->avoid_badpos = 0;
  par->bad_objfactor = 2.0;
  strcpy(par->maskdir, "");
  par->write_chi2fit0 = 0;
  par->warn_badradar = 0;
  par->plot_angmom = 0;
  par->pierce_spinvec = 1;
  par->pierce_angmom = 1;
  par->slice_posfactor = 1.0;
  par->view_posfactor = 1.0;
  par->npar_update = 20;
  par->map_mode = MAPMODE_DELDOP;
  par->map_posfactor = 1.0;
  par->map_fitcutoff = par->map_poscutoff = -9.99;
  for (i=0; i<MAXCHOSENFACETS; i++)
    par->map_comp[i] = 0;
  par->map_verbose = 0;
  par->int_abstol = 1.0e-7;
  par->pa_bodycoords = 0;
  par->pa_highlight = par->view_highlight = par->write_highlight = 0;
  for (i=0; i<MAXCHOSENFACETS; i++)
    par->pa_comp[i] = par->view_comp[i] = par->write_comp[i] = 0;
  par->vary_dopscale = VARY_NONE;
  par->dopscale_min = 0.0;
  par->dopscale_max = HUGENUMBER;
  par->chi2fit0_thresh = 0.0;
  par->listpos_deldop = par->listpos_opt = 0;
  strcpy(par->listpos_path, "");
  par->term_maxiter = 0;
  par->delcor_verbose = 0;

  /*  Start several parameters as negative values or null strings, so
  /*  we can test later to see if they were set in the parameter file  */

  par->ver_samps = -1;
  par->pos_pixels = -1;
  par->pos_width = -9.99;
  par->mask_tol = -9.99;
  par->length_step = par->length_tol = par->length_abstol = -9.99;
  par->ratio_step = par->ratio_tol = par->ratio_abstol = -9.99;
  par->angle_step = par->angle_tol = par->angle_abstol = -9.99;
  par->spin_step = par->spin_tol = par->spin_abstol = -9.99;
  par->spindot_step = par->spindot_tol = par->spindot_abstol = -9.99;
  par->lib_amp_step = par->lib_amp_tol = par->lib_amp_abstol = -9.99;
  par->lib_freq_step = par->lib_freq_tol = par->lib_freq_abstol = -9.99;
  par->lib_phase_step = par->lib_phase_tol = par->lib_phase_abstol = -9.99;
  par->photo_step = par->photo_tol = par->photo_abstol = -9.99;
  par->inertia_step = par->inertia_tol = par->inertia_abstol = -9.99;
  for (i=0; i<MAXDELCORPAR; i++)
    par->delcor_step[i] = par->delcor_abstol[i] = -9.99;
  par->delcor_tol = -9.99;
  par->xyoff_step = par->xyoff_tol = par->xyoff_abstol = -9.99;
  par->term_prec = -9.99;
  strcpy(par->pa_ephfile, "");
  par->nsinc2 = -1;
  par->sinc2width = -1;
  par->dd_clockwiserot = -1;
  par->delcorthresh = -9.99;
  strcpy(par->convex_file, "");
  strcpy(par->delcor_file, "");
  par->slice_viewlong = -HUGENUMBER;
  par->slice_viewlat = -HUGENUMBER;
  for (i=0; i<2; i++) {
    par->semimajor_axis[i] = -9.99;
    par->r_pericenter[i] = -9.99;
    par->eccentricity[i] = -9.99;
    par->t_pericenter[i] = -9.99;
    par->long_asc_node[i] = -HUGENUMBER;
    par->inclination[i] = -HUGENUMBER;
    par->arg_pericenter[i] = -HUGENUMBER;
    par->binary_gravparam[i] = -9.99;
    par->orbit_period[i] = -9.99;
  }
  par->is_triple = -1;
  par->split_const = -HUGENUMBER;
  par->map_set = par->map_frame = -1;
  par->map_dellim[0] = par->map_doplim[0]
                       = par->map_xposlim[0] = par->map_yposlim[0] =  999;
  par->map_dellim[1] = par->map_doplim[1]
                       = par->map_xposlim[1] = par->map_yposlim[1] = -999;
  for (i=0; i<MAXCHOSENFACETS; i++)
    par->map_facet[i] = -1;
  for (i=0; i<MAXCHOSENFACETS; i++)
    par->pa_facet[i] = par->view_facet[i] = par->write_facet[i] = -1;
  par->objfunc_start = -9.99;
}


/******************************************************************/
/*  The following routines read in various types of parameters    */
/******************************************************************/


void read_yesnopar( FILE *fp, unsigned char *parval, char *parname,
                    int par_was_specified[], int n)
{
  char str[MAXLEN];

  check_if_already_read( par_was_specified, NPARAMS, parname, n);
  gettstr( fp, str);
  if (!strcmp( str, "yes")) {
      *parval = 1;
  } else if (!strcmp( str, "no")) {
      *parval = 0;
  } else {
      sprintf(str, "read_par.c: unrecognized value for %s\n", parname);
      bailout(str);
  }
  printf("# %s %s\n", parname, str);
}


int read_yesnovecpar( FILE *fp, unsigned char parval[], char *parname, int n_elements,
                      int par_was_specified[], int n)
{
  const char *yesno[2] = {"no", "yes"};
  char restofline[MAXLEN], message[MAXLEN], *str;
  int n_read, i;

  check_if_already_read( par_was_specified, NPARAMS, parname, n);

  readline( fp, restofline, MAXLEN);
  remove_comments( restofline);
  n_read = 0;
  str = strtok(restofline," \t\n");
  while (str && n_read < n_elements) {
    if (!strcmp( str, "yes")) {
        parval[n_read] = 1;
    } else if (!strcmp( str, "no")) {
        parval[n_read] = 0;
    } else {
        sprintf(message, "read_par.c: unrecognized value for %s component\n", parname);
        bailout(message);
    }
    n_read++;
    str = strtok(NULL, " \t\n");
  }
  if (str) {
    sprintf(message, "read_par.c: %s should have no more than %d components\n",
            parname, n_elements);
    bailout(message);
  }

  printf("# %s", parname);
  for (i=0; i<n_read; i++)
    printf(" %s", yesno[parval[i]]);
  printf("\n");


  return n_read;
}


void read_statepar( FILE *fp, char *parval, char *parname, int equals_allowed,
                    int par_was_specified[], int n)
{
  char str[MAXLEN];

  check_if_already_read( par_was_specified, NPARAMS, parname, n);
  gettstr( fp, str);
  if (!strcmp( str, "c")) {
      *parval = 'c';
  } else if (!strcmp( str, "f")) {
      *parval = 'f';
  } else if (equals_allowed && !strcmp( str, "=")) {
      *parval = '=';
  } else {
      sprintf(str, "ERROR: unrecognized value for %s\n", parname);
      bailout(str);
  }
  printf("# %s %s\n", parname, str);
}


void read_optionspar( FILE *fp, unsigned char *parval, char *parname,
                      const char *options[], unsigned char parvals[], int noptions,
                      int par_was_specified[], int n)
{
  char str[MAXLEN];
  int i, found_option;

  check_if_already_read( par_was_specified, NPARAMS, parname, n);
  gettstr( fp, str);
  i = 0;
  found_option = 0;
  do {
      if (!strcmp( str, options[i])) {
        *parval = parvals[i];
        found_option = 1;
      }
      i++;
  } while (!found_option && i < noptions);
  if (!found_option) {
    sprintf(str, "read_par.c: unrecognized value for %s\n", parname);
    bailout(str);
  }
  printf("# %s %s\n", parname, str);
}


void read_stringpar( FILE *fp, char *parval, char *parname,
                     int par_was_specified[], int n)
{
  check_if_already_read( par_was_specified, NPARAMS, parname, n);
  gettstr( fp, parval);
  printf("# %s %s\n", parname, parval);
}


void read_intpar( FILE *fp, int *parval, char *parname,
                  int par_was_specified[], int n)
{
  check_if_already_read( par_was_specified, NPARAMS, parname, n);
  *parval = getint( fp);
  printf("# %s %d\n", parname, *parval);
}


int read_intvecpar( FILE *fp, int parval[], char *parname, int n_elements,
                    int par_was_specified[], int n)
{
  char restofline[MAXLEN], message[MAXLEN], *str;
  int n_read, i;

  check_if_already_read( par_was_specified, NPARAMS, parname, n);

  readline( fp, restofline, MAXLEN);
  remove_comments( restofline);
  n_read = 0;
  str = strtok(restofline," \t\n");
  while (str && n_read < n_elements) {
    parval[n_read] = string_to_int( str);
    n_read++;
    str = strtok(NULL, " \t\n");
  }
  if (str) {
    sprintf(message, "read_par.c: %s should have no more than %d components\n",
            parname, n_elements);
    bailout(message);
  }

  printf("# %s", parname);
  for (i=0; i<n_read; i++)
    printf(" %d", parval[i]);
  printf("\n");

  return n_read;
}


void read_jdpar( FILE *fp, double *parval, char *parname,
                 int par_was_specified[], int n)
{
  check_if_already_read( par_was_specified, NPARAMS, parname, n);
  *parval = getdouble( fp);
  printf("# %s %lf\n", parname, *parval);
}


int read_jdvecpar( FILE *fp, double parval[], char *parname, int n_elements,
                   int par_was_specified[], int n)
{
  char restofline[MAXLEN], message[MAXLEN], *str;
  int n_read, i;

  check_if_already_read( par_was_specified, NPARAMS, parname, n);

  readline( fp, restofline, MAXLEN);
  remove_comments( restofline);
  n_read = 0;
  str = strtok(restofline," \t\n");
  while (str && n_read < n_elements) {
    parval[n_read] = string_to_double( str);
    n_read++;
    str = strtok(NULL, " \t\n");
  }
  if (str) {
    sprintf(message, "read_par.c: %s should have no more than %d components\n",
            parname, n_elements);
    bailout(message);
  }

  printf("# %s", parname);
  for (i=0; i<n_read; i++)
    printf(" %lf", parval[i]);
  printf("\n");

  return n_read;
}


void read_minmaxpar( FILE *fp, double *parval, char *parname, int minmaxflag,
                     int par_was_specified[], int n)
{
  double doubleval;

  check_if_already_read( par_was_specified, NPARAMS, parname, n);
  doubleval = getdouble( fp);
  if (minmaxflag == 0) {

      /*  minimum  */

      if (doubleval >= *parval) {
          *parval = doubleval;
          printf("# %s %e\n", parname, *parval);
      } else {
          printf("# %s reset to %e\n", parname, *parval);
      }
  } else if (minmaxflag == 1) {

      /*  maximum  */

      if (doubleval <= *parval) {
          *parval = doubleval;
          printf("# %s %e\n", parname, *parval);
      } else {
          printf("# %s reset to %e\n", parname, *parval);
      }
  } else {
      bailout("read_par.c: unrecognized value for minmaxflag\n");
  }
}


void read_doublepar( FILE *fp, double *parval, char *parname,
                     int par_was_specified[], int n)
{
  check_if_already_read( par_was_specified, NPARAMS, parname, n);
  *parval = getdouble( fp);
  printf("# %s %e\n", parname, *parval);
}


int read_doublevecpar( FILE *fp, double parval[], char *parname, int n_elements,
                       int par_was_specified[], int n)
{
  char restofline[MAXLEN], message[MAXLEN], *str;
  int n_read, i;

  check_if_already_read( par_was_specified, NPARAMS, parname, n);

  readline( fp, restofline, MAXLEN);
  remove_comments( restofline);
  n_read = 0;
  str = strtok(restofline," \t\n");
  while (str && n_read < n_elements) {
    parval[n_read] = string_to_double( str);
    n_read++;
    str = strtok(NULL, " \t\n");
  }
  if (str) {
    sprintf(message, "read_par.c: %s should have no more than %d components\n",
            parname, n_elements);
    bailout(message);
  }

  printf("# %s", parname);
  for (i=0; i<n_read; i++)
    printf(" %e", parval[i]);
  printf("\n");

  return n_read;
}


/*  For vector parameters whose components must all be listed on the same line,
    remove comments from that line; each comment must begin and end on that line  */

void remove_comments( char *str)
{
  int i, j, k, found_comment, slen;

  do {
      found_comment = 0;
      slen = strlen(str);
      i = 0;
      while (i < slen && str[i] != '{')
        i++;
      if (i < slen) {
        j = i + 1;
        while (j < slen && str[j] != '}')
          j++;
        if (j < slen) {
          found_comment = 1;
          for (k=j+1; k<=slen; k++)
            str[i+k-j-1] = str[k];
        }
      }
  } while (found_comment);
}


/******************************************************************/
/*  The following routine checks that all essential parameters    */
/*  have been input, and assigns default values to inessential    */
/*  parameters that have not been input                           */
/******************************************************************/


void adjust_par( struct par_t *par)
{
  char dd_maxsides_default[MAXLEN];
  int nsinc2_default, sinc2width_default, dd_clockwiserot_default,
      dd_xflip_default, dd_yflip_default, i, n_sat;
  double delcorthresh_default, xyoff_step_default, xyoff_tol_default,
         xyoff_abstol_default, mask_tol_default, eccentricity_default,
         t_pericenter_default, long_asc_node_default, inclination_default,
         arg_pericenter_default, twopi;

  /*  Set some default parameter values  */

  nsinc2_default = 1;
  sinc2width_default = 6;
  delcorthresh_default = 2.0;
  mask_tol_default = 0.0;
  xyoff_step_default = 1.0;     /* image pixels */
  xyoff_tol_default = 0.0;
  xyoff_abstol_default = 0.01;  /* image pixels */
  eccentricity_default = 0.0;
  t_pericenter_default = 2451545.0;  /* J2000.0 */
  long_asc_node_default = 0.0;
  inclination_default = 0.0;
  arg_pericenter_default = 0.0;
  dd_clockwiserot_default = 0;
  dd_xflip_default = 0;
  dd_yflip_default = 0;
  strcpy(dd_maxsides_default, "right-top");

  /*
      For backwards compatibility, use length_step and length_tol and
      length_abstol for photometric parameters if photo_step and photo_tol
      and photo_abstol weren't given; use spin_step (in deg/day) and
      spin_tol and spin_abstol for the moments of inertia if inertia_step
      and inertia_tol and inertia_abstol weren't given; use spin_step
      (in deg/day) and spin_tol and spin_abstol for the delay correction
      polynomial coefficients if delcor_step and delcor_tol and
      delcor_abstol weren't given; use spin_step/10000 (in deg/day),
      spin_tol, and spin_abstol/10000 if spindot parameters are not given.

      If xyoff_step, xyoff_tol, and xyoff_abstol were not specified,
      use default values rather than halting the program (since these
      three parameters are only needed if the data include optical images).

      For any case where only one of the two tolerances (fractional vs.
      absolute) was specified, zero out the one not specified.
  */

  if (par->action == FIT || par->action == COVAR) {
    if (par->length_step < 0)
      bailout("read_par.c: must specify length_step\n");
    if (par->ratio_step < 0)
      bailout("read_par.c: must specify ratio_step\n");
    if (par->angle_step < 0)
      bailout("read_par.c: must specify angle_step\n");
    if (par->spin_step < 0)
      bailout("read_par.c: must specify spin_step\n");
    if (par->photo_step < 0) {
      par->photo_step = par->length_step;
      printf("# setting photo_step = length_step (%e)\n", par->photo_step);
    }
    if (par->inertia_step < 0) {
      par->inertia_step = par->spin_step / D2R;
      printf("# setting inertia_step = spin_step (%e)\n", par->inertia_step);
    }
    if (par->delcor_step[0] < 0) {
      for (i=0; i<MAXDELCORPAR; i++)
        par->delcor_step[i] = par->spin_step / D2R;
      printf("# setting delcor_step = spin_step (%e)\n", par->delcor_step[0]);
    }
    if (par->xyoff_step < 0) {
      par->xyoff_step = xyoff_step_default;
      printf("# setting xyoff_step   = %e\n", xyoff_step_default);
    }
    if (par->spindot_step < 0) {
      par->spindot_step = par->spin_step / 10000.0;
      printf("# setting spindot_step = spin_step/10000 (%e)\n",
               par->spindot_step * R2D);

     if (par->lib_amp_step < 0) {
      par->lib_amp_step = 0.01 / D2R;
      printf("# setting lib_amp_step = (%e)\n", par->lib_amp_step);
    }     
     if (par->lib_freq_step < 0) {
      par->lib_freq_step = (par->spin_step / D2R);
      printf("# setting lib_freq_step = spin_step (%e)\n", par->lib_freq_step);
    }     
     if (par->lib_phase_step < 0) {
      par->lib_phase_step = 1 / D2R;
      printf("# setting lib_phase_step = (%e)\n", par->lib_phase_step);
    }

    }      
  }

  if (par->action == FIT) {
    if (par->length_tol < 0 && par->length_abstol < 0)
      bailout("read_par.c: must specify length_tol or length_abstol or both\n");
    par->length_tol = MAX( par->length_tol, 0.0);
    par->length_abstol = MAX( par->length_abstol, 0.0);

    if (par->ratio_tol < 0 && par->ratio_abstol < 0)
      bailout("read_par.c: must specify ratio_tol or ratio_abstol or both\n");
    par->ratio_tol = MAX( par->ratio_tol, 0.0);
    par->ratio_abstol = MAX( par->ratio_abstol, 0.0);

    if (par->angle_tol < 0 && par->angle_abstol < 0)
      bailout("read_par.c: must specify angle_tol or angle_abstol or both\n");
    par->angle_tol = MAX( par->angle_tol, 0.0);
    par->angle_abstol = MAX( par->angle_abstol, 0.0);

    if (par->spin_tol < 0 && par->spin_abstol < 0)
      bailout("read_par.c: must specify spin_tol or spin_abstol or both\n");
    par->spin_tol = MAX( par->spin_tol, 0.0);
    par->spin_abstol = MAX( par->spin_abstol, 0.0);

    if (par->photo_tol < 0 && par->photo_abstol < 0) {
        par->photo_tol = par->length_tol;
        par->photo_abstol = par->length_abstol;
        printf("# setting photo_tol    = length_tol    (%e)\n", par->photo_tol);
        printf("# setting photo_abstol = length_abstol (%e)\n", par->photo_abstol);
    } else {
        par->photo_tol = MAX( par->photo_tol, 0.0);
        par->photo_abstol = MAX( par->photo_abstol, 0.0);
    }
    if (par->inertia_tol < 0 && par->inertia_abstol < 0) {
        par->inertia_tol = par->spin_tol;
        par->inertia_abstol = par->spin_abstol;
        printf("# setting inertia_tol    = spin_tol    (%e)\n", par->inertia_tol);
        printf("# setting inertia_abstol = spin_abstol (%e)\n", par->inertia_abstol);
    } else {
        par->inertia_tol = MAX( par->inertia_tol, 0.0);
        par->inertia_abstol = MAX( par->inertia_abstol, 0.0);
    }
    if (par->delcor_tol < 0 && par->delcor_abstol[0] < 0) {
        par->delcor_tol = par->spin_tol;
        for (i=0; i<MAXDELCORPAR; i++)
          par->delcor_abstol[i] = par->spin_abstol;
        printf("# setting delcor_tol    = spin_tol    (%e)\n", par->delcor_tol);
        printf("# setting delcor_abstol = spin_abstol (%e)\n", par->delcor_abstol[0]);
    } else {
        par->delcor_tol = MAX( par->delcor_tol, 0.0);
        for (i=0; i<MAXDELCORPAR; i++)
          par->delcor_abstol[i] = MAX( par->delcor_abstol[i], 0.0);
    }

    if (par->xyoff_tol < 0 && par->xyoff_abstol < 0) {
        par->xyoff_tol = xyoff_tol_default;
        par->xyoff_abstol = xyoff_abstol_default;
        printf("# setting xyoff_tol    = %e\n", xyoff_tol_default);
        printf("# setting xyoff_abstol = %e\n", xyoff_abstol_default);
    } else {
        par->xyoff_tol = MAX( par->xyoff_tol, 0.0);
        par->xyoff_abstol = MAX( par->xyoff_abstol, 0.0);
    }
    if (par->spindot_tol < 0 && par->spindot_abstol < 0) {
        par->spindot_tol = par->spin_tol;
        par->spindot_abstol = (par->spin_abstol / D2R) / 10000.0;
        printf("# setting spindot_tol    = spin_tol    (%e)\n", par->spindot_tol);
        printf("# setting spindot_abstol = spin_abstol/10000 (%e)\n",
                 par->spindot_abstol * R2D);
    } else {
        par->spindot_tol = MAX( par->spindot_tol, 0.0);
        par->spindot_abstol = MAX( par->spindot_abstol, 0.0);
    }

    if (par->lib_amp_tol < 0 && par->lib_amp_abstol < 0) {
        par->lib_amp_tol = 0.1;
        par->lib_amp_abstol = 0.1 / D2R;
        printf("# setting lib_amp_tol    = _tol    (%e)\n", par->lib_amp_tol);
        printf("# setting lib_amp_abstol = spin_abstol/10000 (%e)\n", par->lib_amp_abstol);
    } else {
        par->lib_amp_tol = MAX( par->lib_amp_tol, 0.0);
        par->lib_amp_abstol = MAX( par->lib_amp_abstol, 0.0);
    }
    if (par->lib_freq_tol < 0 && par->lib_freq_abstol < 0) {
        par->lib_freq_tol = 0.1;
        par->lib_freq_abstol = 0.1;
        printf("# setting lib_freq_tol    = spin_tol    (%e)\n", par->lib_freq_tol);
        printf("# setting lib_freq_abstol = spin_abstol/10000 (%e)\n", par->lib_freq_abstol);
    } else {
        par->lib_freq_tol = MAX( par->lib_freq_tol, 0.0);
        par->lib_freq_abstol = MAX( par->lib_freq_abstol, 0.0);
    }
    if (par->lib_phase_tol < 0 && par->lib_phase_abstol < 0) {
        par->lib_phase_tol = 0.1;
        par->lib_phase_abstol = 0.1;
        printf("# setting lib_phase_tol    = 0.1    (%e)\n", par->lib_phase_tol);
        printf("# setting lib_phase_abstol = 0.1 (%e)\n", par->lib_phase_abstol);
    } else {
        par->lib_phase_tol = MAX( par->lib_phase_tol, 0.0);
        par->lib_phase_abstol = MAX( par->lib_phase_abstol, 0.0);
    }

  }

  /*  Check whether or not two crucial parameters,
      pos_pixels and pos_width, were specified: if not, exit.  */

  if (par->action == FIT || par->action == WRITE  || par->action == MOMENTS
                           || par->action == FORMAT || par->action == COVAR
                           || par->action == VIEW   || par->action == AREA
                           || par->action == DELCORINIT
                           || par->action == SLICE  || par->action == ORBIT
                           || par->action == MAP) {
    if (par->pos_pixels < 0)
      bailout("read_par.c: must specify pos_pixels\n");
    if (par->pos_width < 0.0)
      bailout("read_par.c: must specify pos_width\n");
  }

  /*  Exit if any other parameter needed for the desired
      action was not specified in the parameter file.     */

  if (par->action == SAMPLE && par->ver_samps < 0)
    bailout("read_par.c: must specify ver_samps for sample action\n");
  if (par->action == FIT && par->term_prec < 0)
    bailout("read_par.c: must specify term_prec for fit action\n");
  if (par->action == MOMENTS) {
    if (!strcmp( par->pa_ephfile, ""))
      bailout("read_par.c: must specify pa_ephfile for moments action\n");
    else if (par->pa_highlight && par->pa_facet[0] < 0)
      bailout("read_par.c: must specify pa_facet if pa_highlight = yes\n");
  }
  if (par->action == CONVEXHULL && !strcmp( par->convex_file, ""))
    bailout("read_par.c: must specify convex_file for convexhull action\n");
  if (par->action == DELCORINIT && par->delcor_read
                                  && !strcmp( par->delcor_file, ""))
    bailout("read_par.c: must specify delcor_file for delcorinit action if delcor_read = yes\n");
  if (par->action == WRITE && par->mark_unseen && par->pos_scope == GLOBAL)
    bailout("read_par.c: must set pos_scope = local for write action if mark_unseen = yes\n");
  if (par->action == SPLIT && par->split_const == -HUGENUMBER)
    bailout("read_par.c: must specify split_plane for split action\n");
  if (par->action == MAP) {
    if (par->map_mode != MAPMODE_FACETS && (par->map_set < 0 || par->map_frame < 0))
      bailout("read_par.c: must specify map_set and map_frame if map_mode = pos or deldop\n");
    else if (par->map_mode == MAPMODE_POS && (par->map_xposlim[0] > par->map_xposlim[1] ||
                                                par->map_yposlim[0] > par->map_yposlim[1]    ))
      bailout("read_par.c: must specify map_xposlim and map_yposlim if map_mode = pos\n");
    else if (par->map_mode == MAPMODE_DELDOP && (par->map_dellim[0] > par->map_dellim[1] ||
                                                   par->map_doplim[0] > par->map_doplim[1]    ))
      bailout("read_par.c: must specify map_dellim and map_doplim if map_mode = deldop\n");
    else if (par->map_mode == MAPMODE_FACETS && par->map_facet[0] < 0)
      bailout("read_par.c: must specify map_facet if map_mode = facets\n");
  }
  if (par->action == VIEW && par->view_highlight && par->view_facet[0] < 0)
    bailout("read_par.c: must specify view_facet if view_highlight = yes\n");

  /*  Print a warning for a questionable combination of parameter values  */

  if (par->action == FIT && par->pos_scope == LOCAL)
    printf("WARNING: using pos_scope = local can greatly slow down a fit\n");

  /*  Use default values for several nonessential parameters
      if they were not specified in the parameter file        */

  if (par->mask_tol < 0) {
    if (par->pos_pixels > 0 && par->pos_width > 0.0)
      par->mask_tol = par->pos_width / (par->pos_pixels - 1);  /* POS pixel width */
    else
      par->mask_tol = mask_tol_default;
    printf("# setting mask_tol = %e\n", par->mask_tol);
  }
  if (par->nsinc2 < 0) {
    par->nsinc2 = nsinc2_default;
    printf("# setting nsinc2 = %d\n", par->nsinc2);
  }
  if (par->sinc2width < 0) {
    par->sinc2width = sinc2width_default;
    printf("# setting sinc2width = %d\n", par->sinc2width);
  }
  if (par->dd_clockwiserot < 0) {
    par->dd_clockwiserot = dd_clockwiserot_default;
    par->dd_xflip = dd_xflip_default;
    par->dd_yflip = dd_yflip_default;
    printf("# setting dd_maxsides = %s\n", dd_maxsides_default);
  }
  if (par->action == DELCORINIT && par->delcorthresh < 0) {
    par->delcorthresh = delcorthresh_default;
    printf("# setting delcorthresh = %e\n", par->delcorthresh);
  }
  if (par->action == SLICE) {
    if (par->slice_viewlong == -HUGENUMBER) {
      par->slice_viewlong = par->slice_long;
      printf("# setting slice_viewlong = %e\n", par->slice_viewlong * R2D);
    }
    if (par->slice_viewlat == -HUGENUMBER) {
      par->slice_viewlat = par->slice_lat;
      printf("# setting slice_viewlat = %e\n", par->slice_viewlat * R2D);
    }
  }
  if (par->action == ORBIT) {
    n_sat = (par->is_triple == 1) ? 2 : 1;        /* number of satellites */
    if (par->eccentricity[0] < 0.0) {
      for (i=0; i<n_sat; i++)
        par->eccentricity[i] = eccentricity_default;
      printf("# setting eccentricity =\n");
      for (i=0; i<n_sat; i++)
        printf(" %e", par->eccentricity[i]);
      printf("\n");
    }
    if (par->t_pericenter[0] < 0.0) {
      for (i=0; i<n_sat; i++)
        par->t_pericenter[i] = t_pericenter_default;
      printf("# setting t_pericenter =\n");
      for (i=0; i<n_sat; i++)
        printf(" %e", par->t_pericenter[i]);
      printf("\n");
    }
    if (par->long_asc_node[0] == -HUGENUMBER) {
      for (i=0; i<n_sat; i++)
        par->long_asc_node[i] = long_asc_node_default;
      printf("# setting long_asc_node =\n");
      for (i=0; i<n_sat; i++)
        printf(" %e", par->long_asc_node[i] * R2D);
      printf("\n");
    }
    if (par->inclination[0] == -HUGENUMBER) {
      for (i=0; i<n_sat; i++)
        par->inclination[i] = inclination_default;
      printf("# setting inclination =\n");
      for (i=0; i<n_sat; i++)
        printf(" %e", par->inclination[i] * R2D);
      printf("\n");
    }
    if (par->arg_pericenter[0] == -HUGENUMBER) {
      for (i=0; i<n_sat; i++)
        par->arg_pericenter[i] = arg_pericenter_default;
      printf("# setting arg_pericenter =\n");
      for (i=0; i<n_sat; i++)
        printf(" %e", par->arg_pericenter[i] * R2D);
      printf("\n");
    }

    /*  Shift the second Euler angle (inclination) into the range [0, pi] rad
        and the first and third Euler angles (longitude of ascending node,
        argument of pericenter) into the range [0, 2*pi) rad                    */

    twopi = 2*PIE;
    for (i=0; i<n_sat; i++) {
      par->inclination[i] -= twopi*floor(par->inclination[i]/twopi);
      if (par->inclination[i] > PIE) {
        par->inclination[i] = twopi - par->inclination[i];
        par->long_asc_node[i] += PIE;
        par->arg_pericenter[i] += PIE;
      }
      par->long_asc_node[i] -= twopi*floor(par->long_asc_node[i]/twopi);
      par->arg_pericenter[i] -= twopi*floor(par->arg_pericenter[i]/twopi);
    }
  }

  /*  Deal with parameters that can be specified in more than one way  */

  if (par->action == ORBIT) {
    if (par->r_pericenter[0] > 0.0 && par->semimajor_axis[0] > 0.0) {
        for (i=0; i<n_sat; i++)
          par->semimajor_axis[i] = -HUGENUMBER;
        printf("# ignoring semimajor_axis since r_pericenter was specified\n");
    } else if (par->semimajor_axis[0] > 0.0) {
        for (i=0; i<n_sat; i++)
          par->r_pericenter[i] = par->semimajor_axis[i] * (1 - par->eccentricity[i]);
    } else if (par->r_pericenter[0] < 0.0) {
        bailout("read_par.c: must specify r_pericenter or semimajor_axis for orbit action\n");
    }
    if (par->binary_gravparam[0] > 0.0 && par->orbit_period[0] > 0.0) {
        for (i=0; i<n_sat; i++)
          par->orbit_period[i] = -HUGENUMBER;
        printf("# ignoring orbit_period since binary_totmass was specified\n");
    } else if (par->orbit_period[0] > 0.0) {
        if (par->eccentricity[0] < 1.0)
          for (i=0; i<n_sat; i++)
            par->binary_gravparam[i] =
                    pow( 2*PIE/par->orbit_period[i], 2.0)
                    * pow( par->r_pericenter[i] / (1 - par->eccentricity[i]), 3.0);
        else
          bailout("read_par.c: must have eccentricity < 1.0 if orbit_period is specified\n");
    } else if (par->binary_gravparam[0] < 0.0) {
        bailout("read_par.c: must specify binary_totmass or orbit_period for orbit action\n");
    }
  }
}


/******************************************************************/
/*  The following routines read in penalty functions and weights  */
/******************************************************************/


void read_penalties( FILE *fp, struct par_t *par)
{
  int i, pen_was_specified[NPENALTIES];
  char str[80];

  /*  Initialize an array which shows whether or not each
      penalty has been specified so far in the parameter file  */

  for (i=0; i<NPENALTIES; i++)
    pen_was_specified[i] = 0;

  /*  Read the number of penalty functions and allocate
      space to store the information on each penalty     */

  par->pen.n = getint( fp);    /* # of penalty terms */
  printf("# %d penalty terms\n", par->pen.n);
  if (par->pen.n > 0) {
	  if (CUDA) {
		  cudaCalloc((void**)&par->pen.type, sizeof(int), par->pen.n);
		  cudaCalloc((void**)&par->pen.weight, sizeof(double), par->pen.n);
		  cudaCalloc((void**)&par->pen.base, sizeof(double), par->pen.n);
		  /* Fix addressing since the code below inexplicably demands to start
		   * indexing at 1 rather than 0		   */
		  par->pen.type -= 1;
		  par->pen.weight -= 1;
		  par->pen.base -= 1;
	  } else {
		  par->pen.type = ivector( 1, par->pen.n);
		  par->pen.weight = vector( 1, par->pen.n);
		  par->pen.base = vector( 1, par->pen.n);
	  }
  }

  /*  Loop through and read each penalty weight  */

  for (i=1; i<=par->pen.n; i++) {
    gettstr( fp, str);
    par->pen.weight[i] = getdouble( fp);
    if (!strcmp( str, "optalbdel"))
      assign_penalty( OPTALBDEL, &par->pen.type[i], str, par->pen.weight[i],
                      pen_was_specified, 0);
    else if (!strcmp( str, "optalbvar"))
      assign_penalty( OPTALBVAR, &par->pen.type[i], str, par->pen.weight[i],
                      pen_was_specified, 1);
    else if (!strcmp( str, "thetadel"))
      assign_penalty( THETADEL, &par->pen.type[i], str, par->pen.weight[i],
                      pen_was_specified, 2);
    else if (!strcmp( str, "thetavar"))
      assign_penalty( THETAVAR, &par->pen.type[i], str, par->pen.weight[i],
                      pen_was_specified, 3);
    else if (!strcmp( str, "radalbdel"))
      assign_penalty( RADALBDEL, &par->pen.type[i], str, par->pen.weight[i],
                      pen_was_specified, 4);
    else if (!strcmp( str, "radalbvar"))
      assign_penalty( RADALBVAR, &par->pen.type[i], str, par->pen.weight[i],
                      pen_was_specified, 5);
    else if (!strcmp( str, "rad_c_del"))
      assign_penalty( RAD_C_DEL, &par->pen.type[i], str, par->pen.weight[i],
                      pen_was_specified, 6);
    else if (!strcmp( str, "rad_c_var"))
      assign_penalty( RAD_C_VAR, &par->pen.type[i], str, par->pen.weight[i],
                      pen_was_specified, 7);
    else if (!strcmp( str, "nonsmooth"))
      assign_penalty( NONSMOOTH, &par->pen.type[i], str, par->pen.weight[i],
                      pen_was_specified, 8);
    else if (!strcmp( str, "concavity"))
      assign_penalty( CONCAVITY, &par->pen.type[i], str, par->pen.weight[i],
                      pen_was_specified, 9);
    else if (!strcmp( str, "rdev"))
      assign_penalty( RDEV, &par->pen.type[i], str, par->pen.weight[i],
                      pen_was_specified, 10);
    else if (!strcmp( str, "comdev"))
      assign_penalty( COMDEV, &par->pen.type[i], str, par->pen.weight[i],
                      pen_was_specified, 11);
    else if (!strcmp( str, "volume"))
      assign_penalty( VOLUME, &par->pen.type[i], str, par->pen.weight[i],
                      pen_was_specified, 12);
    else if (!strcmp( str, "inertiadev"))
      assign_penalty( INERTIADEV, &par->pen.type[i], str, par->pen.weight[i],
                      pen_was_specified, 13);
    else if (!strcmp( str, "inertiadev_uni"))
      assign_penalty( INERTIADEV_UNI, &par->pen.type[i], str, par->pen.weight[i],
                      pen_was_specified, 14);
    else if (!strcmp( str, "pa3tilt"))
      assign_penalty( PA3TILT, &par->pen.type[i], str, par->pen.weight[i],
                      pen_was_specified, 15);
    else if (!strcmp( str, "nonpa"))
      assign_penalty( NONPA, &par->pen.type[i], str, par->pen.weight[i],
                      pen_was_specified, 16);
    else if (!strcmp( str, "nonpa_uni"))
      assign_penalty( NONPA_UNI, &par->pen.type[i], str, par->pen.weight[i],
                      pen_was_specified, 17);
    else if (!strcmp( str, "euleroffs"))
      assign_penalty( EULEROFFS, &par->pen.type[i], str, par->pen.weight[i],
                      pen_was_specified, 18);
    else if (!strcmp( str, "flattening"))
      assign_penalty( FLATTENING, &par->pen.type[i], str, par->pen.weight[i],
                      pen_was_specified, 19);
    else if (!strcmp( str, "maxrdev"))
      assign_penalty( MAXRDEV, &par->pen.type[i], str, par->pen.weight[i],
                      pen_was_specified, 20);
    else if (!strcmp( str, "maxellipdev"))
      assign_penalty( MAXELLIPDEV, &par->pen.type[i], str, par->pen.weight[i],
                      pen_was_specified, 21);
    else if (!strcmp( str, "noncosine"))
      assign_penalty( NONCOSINE, &par->pen.type[i], str, par->pen.weight[i],
                      pen_was_specified, 22);
    else if (!strcmp( str, "bifurcation"))
      assign_penalty( BIFURCATION, &par->pen.type[i], str, par->pen.weight[i],
                      pen_was_specified, 23);
    else if (!strcmp( str, "impulse"))
      assign_penalty( IMPULSE, &par->pen.type[i], str, par->pen.weight[i],
                      pen_was_specified, 24);
    else {
      printf("????? %s ?????\n", str);
      bailout("read_par.c: don't recognize that penalty\n");
    }
  }
}


void assign_penalty( int pentype_val, int *pentype, char *penname, double penweight,
                     int pen_was_specified[], int n)
{
  check_if_already_read( pen_was_specified, NPENALTIES, penname, n);
  *pentype = pentype_val;
  printf("# penalty %s, weight %e\n", penname, penweight);
}


/******************************************************************/
/*  The following routine checks that no parameter or penalty     */
/*  function is included more than once in the parameter file     */
/******************************************************************/


void check_if_already_read( int was_already_read[], int ntot, char *name, int n)
{
  char str[MAXLEN];

  if (n >= ntot)
    bailout("read_par.c: Going past end of array in routine check_if_already_read\n");

  if (was_already_read[n]) {
      sprintf(str, "read_par.c: '%s' is specified more than once in the parameter file\n",
              name);
      bailout(str);
  } else {
      was_already_read[n] = 1;
  }
}


/******************************************************************/
/*  The following routine checks that parameters for the "orbit"  */
/*  action are entered appropriately for either a binary system   */
/*  or a triple system, not some one way and some the other way   */
/******************************************************************/

void check_triple( int n_read, int n_max, int *is_triple)
{

  if (*is_triple == -1)
    *is_triple = (n_read == n_max) ? 1 : 0;
  else if ((n_read == n_max && *is_triple == 0) ||
           (n_read != n_max && *is_triple == 1)    )
    bailout("read_par.c: 'orbit' action parameters are mixed on binary vs. triple system\n");

}

#undef NPARAMS
#undef NPENALTIES
#undef NACTIONS
