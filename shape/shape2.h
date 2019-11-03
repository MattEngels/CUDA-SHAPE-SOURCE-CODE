/*****************************************************************************************
                                                                                 shape2.h
Modified 2016 July 7 by Matt Engels:
	Adapted for use in shape-cuda - all function prototypes not related to the 'fit' action
		have been commented out.  All structs and unions have been left in place.

Modified 2015 June 10 by CM:
    Add "nviews" "view_interval" "smearing_mode" and "v0" to deldop_t, doppler_t, poset_t,
        and lghtcrv_t
    Add new "deldopview_t" structure to deldopfrm_t and move/copy several variables from
        deldopfrm_t to deldopview_t
    Add new "dopview_t" structure to dopfrm_t and move/copy several variables from
        dopfrm_t to dopview_t
    Add new "posetview_t" structure to posetfrm_t and move/copy several variables from
        posetfrm_t to posetview_t
    Change "t" from a vector to a matrix in lghtcrv_t

Modified 2014 August 25 by SN:
    Add "lib_amp_step", "lib_amp_tol", and "lib_amp_abstol" parameters.
    Add "lib_freq_step", "lib_freq_tol", and "lib_freq_abstol" parameters.
    Add "lib_phase_step", "lib_phase_tol", and "lib_phase_abstol" parameters.
    Add "lib_amp", "lib_freq" and "lib_phase" to spin_t struct

Modified 2014 February 25 by CM:
    Add "delcor_verbose" parameter

Modified 2014 February 12 by CM:
    Implement multiple radar and optical scattering laws

Modified 2013 July 28 by CM:
    Add "write_highlight" "write_comp" and "write_facet" parameters

Modified 2013 July 14 by CM:
    Add "term_maxiter" parameter

Modified 2013 June 2 by CM:
    Fix a comment

Modified 2013 May 19 by CM:
    Add "ovoid_t" structure
    Add "ovoid_t" to comp_t and add OVOID defined constant

Modified 2013 April 14 by CM:
    Add "listpos_deldop" "listpos_opt" and "listpos_path" parameters

Modified 2012 December 5 by CM:
    Add "chi2fit0_thresh" parameter

Modified 2012 June 13 by CM:
    Add "objfunc_start" parameter

Modified 2012 March 24 by CM:
    Add "dopscale" and "dopscale_save" to deldop_t and doppler_t
    Add "sum_cos_subradarlat_weights" to deldop_t, doppler_t, and dat_t
    Add "vary_dopscale" "dopscale_min" and "dopscale_max" parameters
    Add "baddopscale" and "baddopscale_logfactor" to par_t

Modified 2012 March 5 by CM:
    Add "pa_highlight" "pa_comp" "pa_facet" "view_highlight" "view_comp" and "view_facet"
        parameters
    Rename MAXMAPFACETS defined constant as MAXCHOSENFACETS
    Add "pa_bodycoords" parameter

Modified 2011 September 2 by CM:
    Add "area" and "x" to facet_t
    Add HARMLAMBERT and INHOLAMBERT optical scattering laws, and rename the harmlommel_t
        and inholommel_t structures as harmR_t and inhoR_t

Modified 2011 August 19 by CM:
    Add "int_abstol" parameter
    Add "n_impulse" and "impulse" to spin_t
    Add "n_integrate" "t_integrate" and "impulse" to deldopfrm_t, dopfrm_t,
        posetfrm_t, and crvrend_t
    Add MAXIMP defined constant
    Remove "onepos" from dat_t

Modified 2010 August 25 by CM:
    Change "map_forward" parameter to "map_mode"
    Add "map_comp" "map_facet" and "map_verbose" parameters
    Add "map_facet_power" to deldopfrm_t and dopfrm_t

Modified 2010 August 10 by CM:
    Added "radfitmin" and "radobsmin" parameters

Modified 2010 June 15 by CM:
    Remove "scalefactor" parameter in par_t
    Add "map_set" "map_frame" "map_forward" "map_dellim" "map_doplim"
        "map_xposlim" "map_yposlim" "map_posfactor" "map_fitcutoff"
        "map_poscutoff" and "map_overflow" parameters
    Add "map_fit" and "map_pos" to deldopfrm_t and dopfrm_t

Modified 2010 June 1 by CM:
    Changed scalefactor from a scalar to a 3-component vector in harmonic_t
        and vertices_t

Modified 2010 May 20 by CM:
    Add "split_norm" and "split_const" parameters

Modified 2010 April 27 by CM:
    Add tabular_t structure for the "tabular" radar scattering law

Modified 2010 March 19 by CM:
    Add "v_mirror" to vertex_t

Modified 2009 August 2 by CM:
    Add "act" to side_t

Modified 2009 July 5 by CM:
    Add "npar_update" parameter

Modified 2009 April 10 by CM:
    Add "slice_posfactor" and "view_posfactor" parameters

Modified 2009 April 3 by CM:
    Add "warn_badradar" "plot_angmom" "pierce_spinvec" and "pierce_angmom"
        parameters
    Add "ncalc_obsfile" "jdstart" "jdstop" "jdinterval" to lghtcrv_t
    Add "badposet" "badradar" "baddiam_logfactor" "badphoto_logfactor"
        "posbnd_logfactor" "badposet_logfactor" and "badradar_logfactor"
        parameters to par_t
    Add "posbnd_logfactor" parameter to pos_t
    Add "badradar_logfactor" parameter to deldopfrm_t and dopfrm_t

Modified 2008 August 19 by CM:
    Add "idelvig" and "idopvig" to deldopfrm_t
    Add "idopvig" to dopfrm_t

Modified 2008 August 10 by CM:
    Change "delcor_step" and "delcor_abstol" parameters to be vectors
        rather than scalars

Modified 2008 July 11 by CM:
    Turned "orbit" action parameters into vectors so that triple systems
       can be handled

Modified 2007 October 22 by CM:
    Added "dopDC_vig" to deldopfrm_t

Modified 2007 September 24 by CM:
    Change "pa_ephformat" from char to unsigned char

Modified 2007 September 14 by CM:
    Change some parameters from int to unsigned char
    Added "write_chi2fit0" parameter

Modified 2007 August 31 by CM:
    Added "avoid_badpos" "bad_chi2factor" and "maskdir" parameters

Modified 2007 August 15 by CM:
    Added "term_badmodel" parameter

Modified 2007 August 11 by CM:
    Added "radfitmax" and "radobsmax" and "write_posfactor" parameters

Modified 2007 August 4 by CM:
    Added "semimajor_axis" "r_pericenter" "eccentricity" "t_pericenter"
        "long_asc_node" "inclination" "arg_pericenter" "binary_gravparam"
        "orbit_period" "orbit_reflex" and "orbit_posfactor" parameters
    Added body, bodyill, comp, and compill matrices to pos_t

Modified 2007 February 21 by CM:
    Added "slice_long" "slice_lat" "slice_offset" "slice_viewlong"
        "slice_viewlat" "slice_sunlong" "slice_sunlat" "slice_scatlaw"
        "slice_shadows" "slice_planefrac" "slice_skyfactor"
        "slice_dimfactor" "slice_read" and "slice_dointerp" parameters

Modified 2007 January 6 by CM:
    Added rotphase_calc and rotphase_obs to lghtcrv_t

Modified 2006 October 1 by CM:
    Added scalefactor to harmonic_t and vertices_t
    In ellipse_t, replaced diameters D with two_a, a_over_b, b_over_c
    Added "ratio_step" "ratio_tol" and "ratio_abstol" parameters
    Added "vary_delcor0" "vary_radalb" and "vary_optalb" parameters,
        and added three related sums (sum_deldop_zmax_weights,
        sum_rad_xsec_weight, and sum_opt_brightness_weights) to dat_t
    Changed parampoly_t to delcor_t and added delcor0_save to it
    Added R_save to RC_t, quasispec_t, R_t, and kaas_t
    Added w_save to hapke_t
    Added a_save and b_save to harmonic_t

Modified 2006 June 20 by CM:
    In deldop_t, changed delres to del_per_pixel and dopres to
        dop_per_pixel, and improved comments
    In doppler_t, changed dopres to dop_per_bin and improved comment
    In poset_t, changed angres to angle_per_pixel and improved comment
    In pos_t, changed res to km_per_pixel and improved comment

Modified 2006 June 18 by CM:
    Add 24 parameters which are user-specified upper and lower limits to
        photometric parameters (rad_R_min, opt_wt_max, etc.)
    Add "ndel" "ndop" "delcom_vig" "dopcom_vig" and "pixels_weighted" to
        deldopfrm_t
    Add "ndop" "dopcom_vig" and "pixels_weighted" to dopfrm_t
    Add "nrow" "ncol" "rowcom_vig" "colcom_vig" "pixels_weighted" and
        "oneovervar" to posetfrm_t
    Add "dof" (degrees of freedom) to deldopfrm_t, deldop_t, dopfrm_t,
        doppler_t, posetfrm_t, poset_t, and lghtcrv_t
    Eliminate range datasets

Modified 2006 April 6 by PT:
    Add "spindot_step", "spindot_tol", and "spindot_abstol" parameters
    Add "omegadot[3]" to spin_t struct

Modified 2006 March 10 by CM:
    Add "overflow_xsec" and "overflow_dopmean" to deldopfrm_t and dopfrm_t,
         and add "overflow_delmean" to deldopfrm_t, for use with the
        "delcorinit" action 

Modified 2005 November 18 by MCN:
    Change "nlooks" (number of looks) from int to double in deldopfrm_t
        and dopfrm_t, to handle zero-filled (delay-)Doppler data

Modified 2005 October 6 by CM:
    Add "sun_appmag" parameter

Modified 2005 September 18 by CM:
    Add "view_shadows" "view_sunlat" and "view_sunlong" parameters

Modified 2005 September 11 by CM:
    Add parameters for the "photoharm" action: radharm, optharm,
        rad_R_nharm, rad_C_nharm, opt_R_nharm, etc.

Modified 2005 September 8 by CM:
    Add harmlommel_t, harmhapke_t, and harmkaas_t structures for the
        "harmlommel" "harmhapke" and "harmkaas" optical scattering laws
    Add harmcosine_t structure for the "harmcosine" radar scattering law
    Add parameters for the "photofacets" action: radfacets, optfacets,
        rad_global_R_state, rad_local_C_state, opt_local_R_state, etc.

Modified 2005 August 17 by CM:
    Add afactor and bfactor (matrices of spherical harmonic functions) to
        vertex_t, so that they can be computed by the read_mod routine
        just once per fit rather than by the realize_mod routine every time
        a harmonic model is realized

Modified 2005 August 8 by CM:
    Add inhokaas_t structure for the "inhokaas" optical scattering law

Modified 2005 August 1 by CM:
    Add "pa_scatlaw" "view_scatlaw" "sky_radlaw" and "sky_optlaw"
        parameters

Modified 2005 July 21 by CM:
    Add "exclude_seen" parameter
    Create inhocosine_t structure so that the "inhocosine" radar scattering
        law can be set up just like the inhomogeneous optical scattering
        laws
    Add the "hagfors" and "cosine_qs" quasispecular radar scattering laws,
        and the quasispec_t structure for those and the "gaussian" law
    Add the "gauss+cosine" and "hagfors+cosine" and "cosine+cosine"
        hybrid radar scattering laws, and the hybridradar_t structure which
        describes them
    Eliminate the "flat" radar scattering law
    Rename and renumber the defined constants which refer to radar
        scattering laws

Modified 2005 July 4 by CM:
    Create inholommel_t structure so that the "inholommel" optical
        scattering law can be set up just like the "inhohapke" law

Modified 2005 July 3 by CM:
    Add "lcrv_writeall" parameter

Modified 2005 June 25 by CM:
    In deldopfrm_t and dopfrm_t, rename old "dellim" and "doplim" to
        "idellim" and "idoplim"; these are integer (delay-)Doppler limits
        in image rows and columns (deldopfrm_t) or spectral bins (dopfrm_t)
    In deldopfrm_t and dopfrm_t, add new "dellim" and "doplim"; these are
        floating point (delay-)Doppler limits in usec and Hz, respectively,
        determined PRIOR to convolution with the delay and Doppler response
        functions

Modified 2005 June 24 by CM:
    Add "mark_unseen" and "mincosine_seen" parameters; add "seen" to
        facet_t

Modified 2005 June 20 by CM:
    Add "endian" parameter

Modified 2005 April 25 by CM:
    Add "chi2_variance" to dat_t

Modified 2005 March 10 by CM:
    Change "dof" (degrees of freedom) from int to double for dat_t
    Change "weight" from int to double for deldopfrm_t, dopfrm_t,
        posetfrm_t, lghtcrv_t, and rangefrm_t
    Add "dof_deldop" "dof_doppler" "dof_poset" "dof_lghtcrv" and
        "dof_range" (floating-point degrees of freedom) to dat_t
    Add "dof" (floating-point degrees of freedom) to set_t

Modified 2005 March 6 by CM:
    Add "list_posetcal" parameter

Modified 2005 March 1 by CM:
    Rename a few "var" and "sdev" values to "overovervar"
        (1/variance)
    Add "northangle" (direction of north) to posetfrm_t
    Add "xyoff_step" "xyoff_tol" and "xyoff_abstol" parameters

Modified 2005 February 21 by CM:
    Add "poset_resample" and "image_rebin" parameters

Modified 2005 February 6 by CM:
    Add "delta_rotphase" parameter

Modified 2005 January 19 by CM:
    Add POS frame, ae and oe transformation matrices, and spin properties
        to posetfrm_t and rangefrm_t
    Add solar phase angle and solar azimuth angle to posetfrm_t
    Add "poset_scaling" and "poset_resid" parameters

Modified 2005 January 13 by CM:
    Add "inputnode" parameter to set_t to hold the mpi node number which
        was listed in the input obs file

Modified 2004 December 19 by CM:
    Add "area_latincr" and "area_longincr" parameters

Modified 2004 November 30 by CM:
    Add "listfit" and "listfit_path" parameters

Modified 2004 November 23 by CM:
    Add "view_long" and "view_lat" parameters

Modified 2004 October 29 by CM:
    Add "first_fitpar" parameter

Modified 2004 August 13 by CM:
    Add six new parameters, absolute tolerances for fitting:
        "length_abstol" "angle_abstol"   "spin_abstol"
        "photo_abstol"  "inertia_abstol" "delcor_abstol"
    Add "*fparabstol" as pointers to these absolute tolerances

Modified 2004 July 31 by CM:
    Add "delcor_read" and "delcor_file" parameters

Modified 2004 May 8 by CM:
    Add "scalefactor" parameter

Modified 2004 April 29 by CM:
    Change kaas_t structure to use "wt" weighting factor rather
        than the original "c" weighting factor for Lambert vs.
        Lommel-Seeliger optical scattering

Modified 2004 April 9 by CM:
    Add vector of solar azimuth angles to lightcurve datasets

Modified 2004 March 27 by CM:
    Add spin properties to crvrend_t so we can include the spin vector
        in pgm output
    Add "plot_spinvec" parameter to par_t to include the spin vector
        in pgm output
    Add "plot_subradar" parameter to par_t to mark the subradar point
        in pgm output for (delay-)Doppler plane-of-sky views
    Add "plot_pa" parameter vector to par_t to include the positive
        ends of one or more principal axes in in pgm output for
        (delay-)Doppler plane-of-sky views

Modified 2004 March 6 by CM:
    Add "convex_file" parameter to par_t for convex hull output
    Move prototypes for functions setupsides and setupvertices from
        read_mod.c to here, so that routine convex_hull can access them

Modified 2004 February 29 by CM:
    Add "delcorthresh" parameter to par_t
    Add "jdoffset" and "perform_ltc" parameters to par_t as replacements
        for the JD244 and LTC variables in switch.h
    Define kaas_t structure to hold parameters for the Kaasalainen
        "Lambert + Lommel-Seeliger" optical scattering law, and add
        this structure to photo_t
    Add vector of solar phase angles to lightcurve datasets

Modified 2003 July 30 by CM:
    Add "dd_clockwiserot" and "dd_xflip" and "dd_yflip" parameters to par_t

Modified 2003 May 16 by CM:
    Add "listres" parameter to par_t

Modified 2003 May 10 by CM:
    Add "overflow_o2" and "overflow_m2" to deldopfrm_t and dopfrm_t

Modified 2003 May 7 by CM:
    Add "sinc2width" parameter to par_t

Modified 2003 April 29 by CM:
    Add "nsinc2" parameter to par_t

Modified 2003 April 23 by CM:
    Move "delcom" from delay-Doppler datasets to individual frames
    Add "weight" to delay-Doppler, Doppler, POS, and range frames
        and to lightcurve datasets

Modified 2003 April 21 by CM:
    Add "pa_*" parameters to par_t

Modified 2003 April 17 by CM:
    Add "baddiam" parameter to par_t and add *par argument to realize_mod
    Add "area" field to comp_t and shape_t

Modified 2003 April 15 by CM:
    Add delay-Doppler parameters for code method, spb, stride,
        Doppler fft length, and Doppler DC bin

Modified 2003 April 3 by CM:
    Add "photo_step" and "photo_tol" fields for fitting photometric parameters,
        "inertia_step" and "inertia_tol" for fitting moments of inertia,
        "delcor_step' and "delcor_tol" for fitting delay correction polynomial
*****************************************************************************************/

#include <stdio.h>
#include "const.h"
#include <cuda_runtime_api.h>
/*****************************************************************************************
                                DEFINITION OF struct par_t
*****************************************************************************************/

struct pen_t {
  int n;                        /* # of penalties */
  int *type;                
  double *weight;
  double *base;                 /* total pen is weight*base */
  double *x_noncosine;          /* storage vector for the noncosine penalty */
  double *y_noncosine;          /* storage vector for the noncosine penalty */
  double xmean_noncosine;       /* mean of x_noncosine values */
};

#define MAXDELCORPAR 10
#define MAXSUNMAGS 10
#define MAXCHOSENFACETS 20
struct par_t {
  unsigned char action;
  unsigned char speckle;
  unsigned char pos_smooth;
  unsigned char pos_scope;
  unsigned char dd_scaling;
  unsigned char int_method;     /* method for integrating Euler's equations */
  unsigned char lcrv_pos;
  unsigned char posbnd;         /* =1 if POS rendering exceeds POS frame size */
  unsigned char badphoto;       /* =1 if photometric function invalid */
  unsigned char baddiam;        /* =1 if ellipsoid diams tiny or negative */
  unsigned char badposet;       /* =1 if plane-of-sky fit image is too small to "contain" target */
  unsigned char badradar;       /* =1 if model is too wide in (delay-)Doppler space */
  unsigned char baddopscale;    /* =1 if model has Doppler scaling factors out of allowed range */
  double bad_objfactor;         /* factor to increase obj func for illegal properties (default = 2.0) */
  double posbnd_logfactor;      /*                                                                    */
  double baddiam_logfactor;     /* if posbnd = 1, increase the objective function by a factor of      */
  double badphoto_logfactor;    /*              bad_objfactor * exp(posbnd_logfactor)                 */
  double badposet_logfactor;    /* -- and similarly for badphoto, baddiam, badposet, badradar, and    */
  double badradar_logfactor;    /*                      baddopscale                                   */
  double baddopscale_logfactor; /*                                                                    */
  unsigned char term_badmodel;  /* =1 stop fit if model has illegal properties at end of iteration */
  unsigned char avoid_badpos;   /* =1 try to shrink model if it extends beyond POS frame */
  unsigned char warn_badradar;  /* =1 warn if model too wide in delay-Doppler space (default = 0) */
  unsigned char showstate;
  unsigned char read_node;      /* =1 then obs file assigns to nodes */
  int theta_steps;
  int ver_samps;
  int pos_pixels;
  double pos_width;
  double dd_gamma;
  double mask_tol;
  double length_step;           /* initial step size for bracketing minima (km) */
  double length_tol;            /* fractional tolerance for length parameters */
  double length_abstol;         /* absolute tolerance for length parameters (km) */
  double ratio_step;
  double ratio_tol;
  double ratio_abstol;
  double angle_step;            /* input in degrees, convert internally to radians */
  double angle_tol;
  double angle_abstol;          /* input in degrees, convert internally to radians */
  double spin_step;             /* input in deg/day, convert internally to rad/day */
  double spin_tol;
  double spin_abstol;           /* input in deg/day, convert internally to rad/day */
  double spindot_step;          /* input in deg/day/day, convert internally to rad/day/day */
  double spindot_tol;
  double spindot_abstol;        /* input in deg/day/day, convert internally to rad/day/day */
  double lib_amp_step;          /* input in degrees, convert internally to radians */
  double lib_amp_tol;           
  double lib_amp_abstol;        /* input in degrees, convert internally to radians */
  double lib_freq_step;         /* input in deg/day, convert internally to rad/day */
  double lib_freq_tol;
  double lib_freq_abstol;       /* input in deg/day, convert internally to rad/day */
  double lib_phase_step;        /* input in degrees, convert internally to radians */
  double lib_phase_tol;         
  double lib_phase_abstol;      /* input in degrees, convert internally to radians */
  double photo_step;
  double photo_tol;
  double photo_abstol;
  double inertia_step;
  double inertia_tol;
  double inertia_abstol;
  double delcor_step[MAXDELCORPAR];
  double delcor_tol;
  double delcor_abstol[MAXDELCORPAR];
  double xyoff_step;            /* unit = image pixels */
  double xyoff_tol;
  double xyoff_abstol;          /* unit = image pixels */
  double term_prec;             /* minimum fractional decrease in obj. function to continue a fit */
  int term_maxiter;             /* maximum number of fit iterations (default = 0: no maximum) */
  double dd_resid;              /* residual image |pixel value| that maps to bright white */
  double optposmax;             /* optical POS image pixel value that maps to bright white */
  double radposmax;             /* (delay-)Doppler POS image pixel value --> bright white */
  double radfitmin;             /* d-D fit pixel value (incl. calfact) --> black (default = 0.0) */
  double radfitmax;             /* delay-Doppler fit pixel value (incl. calfact) --> bright white */
  double radobsmin;             /* delay-Doppler obs pixel value that maps to black (default = 0.0) */
  double radobsmax;             /* delay-Doppler obs pixel value that maps to bright white */
  char pa_ephfile[255];         /* name of standalone ephemeris file for "moments" action */
  unsigned char pa_ephformat;   /* format of pa_ephfile (default = horizons) */
  double pa_startepoch;         /* to start search for when long principal axis is in POS */
  double pa_stopepoch;          /* to stop  search for when long principal axis is in POS */
  double pa_angleoff[3];        /* if nonzero angle offset is wanted for PA epoch search  */
  double pa_omegaoff[3];        /* if nonzero spin  offset is wanted for PA epoch search  */
  int nsinc2;                   /* # of points per POS pixel width to evaluate sinc^2 */
  int sinc2width;               /* # of Doppler bins over which to evaluate sinc^2 */
  unsigned char scalefitobs;    /* fit & obs pgm files scaled same vs. each separately */
  unsigned char listres;        /* =1 outputs files with residual matrices */
  int dd_clockwiserot;          /* =1 rotates fit,obs,res pgm images 90 deg clockwise */
  int dd_xflip;                 /* =1 flips x-axis of fit,obs,res pgm images (after rotation) */
  int dd_yflip;                 /* =1 flips y-axis of fit,obs,res pgm images (after rotation) */
  double delcorthresh;          /* frame SNR must be this high to use for delcorinit */
  double jdoffset;              /* subtract this from Julian dates for lightcurve output */
  unsigned char perform_ltc;    /* =1 correct input epochs for 1-way light travel time */
  char convex_file[255];        /* output mod file for convex hulls */
  struct pen_t pen;             /* penalties */
  unsigned char plot_spinvec;   /* =1 shows spin vector in plane-of-sky pgm files */
  unsigned char plot_subradar;  /* =1 marks subradar point in (delay-)Doppler POS pgm files */
  unsigned char plot_com;       /* =1 marks projected COM in (delay-)Doppler POS pgm files */
  unsigned char plot_pa[3];     /* =1 shows principal axis i in (delay-)Doppler POS pgm files */
  unsigned char plot_angmom;    /* =1 shows angular momentum vector in plane-of-sky pgm files */
  unsigned char pierce_spinvec; /* =1 spin vector pierces model (default = 1) */
  unsigned char pierce_angmom;  /* =1 angular momentum vector pierces model (default = 1) */
  unsigned char write_obs;      /* =1 overwrites obs file for "write" action */
  unsigned char delcor_read;    /* =1 reads data centroids from file delcor_file */
  char delcor_file[255];        /* input file for delcorinit action (data centroids) */
  unsigned char delcor_verbose; /* =1 print data centroids to the screen (default = 0) */
  int first_fitpar;             /* first fit parameter (0-based) for first fit iteration */
  double view_long;             /* body-fixed viewing longitude for view action (deg -> rad) */
  double view_lat;              /* body-fixed viewing latitude  for view action (deg -> rad) */
  unsigned char view_shadows;   /* =1 includes shadowing in view action */
  double view_sunlong;          /* body-fixed solar longitude for view action (deg -> rad) */
  double view_sunlat;           /* body-fixed solar latitude  for view action (deg -> rad) */
  unsigned char listfit;        /* =1 outputs model "data" files for delay-Doppler frames */
  char listfit_path[255];       /* path for listfit output files (default = null string) */
  double area_latincr;          /* approximate subobs. lat. increment for "area" action (deg -> rad) */
  double area_longincr;         /* approximate subobs. long. increment for "area" action (deg -> rad) */
  unsigned char poset_resample; /* for resampling model POS frame to get plane-of-sky fit frame */
  unsigned char image_rebin;    /* =1 turns on rebinning for undersampled pgm images */
  unsigned char poset_scaling;
  double poset_resid;
  unsigned char list_posetcal;  /* =1 writes plane-of-sky calibration factors to disk */
  double delta_rotphase;        /* angle added to all rotation phases for "write" action (deg -> rad) */
  unsigned char endian;         /* "big" or "little" for binary I/O (default = big) */
  unsigned char mark_unseen;    /* =1 marks unseen/shadowed regions in color */
  double mincosine_seen;        /* minimum cose or cosi to have "seen" a facet (default = 0.0) */
  unsigned char lcrv_writeall;  /* =1 writes info on all model lightcurve points (default = 0) */
  int exclude_seen;             /* dataset to exclude in determining "unseen" model facets */
  unsigned char pa_scatlaw;     /* scattering law for moments action (default = lambertian) */
  unsigned char view_scatlaw;   /* scattering law for view action (default = lambertian) */
  unsigned char sky_radlaw;     /* scattering law for radar POS images (default = lambertian) */
  unsigned char sky_optlaw;     /* scattering law for optical POS images (default = optical) */
  unsigned char radfacets;      /* =1 (default) converts radar scat. law for photofacets action */
  char rad_global_R_state;      /* for photofacets action (default = 'c') */
  char rad_global_C_state;
  char rad_local_R_state;       /* for photofacets action (default = 'f') */
  char rad_local_C_state;
  unsigned char optfacets;      /* =1 (default) converts opt. scat. law for photofacets action */
  char opt_global_R_state;      /* for photofacets action (default = 'c') */
  char opt_global_wt_state;
  char opt_global_A0_state;
  char opt_global_D_state;
  char opt_global_k_state;
  char opt_global_w_state;
  char opt_global_h_state;
  char opt_global_B0_state;
  char opt_global_g_state;
  char opt_global_theta_state;
  char opt_local_R_state;       /* for photofacets action (default = 'f') */
  char opt_local_wt_state;
  char opt_local_A0_state;
  char opt_local_D_state;
  char opt_local_k_state;
  char opt_local_w_state;
  char opt_local_h_state;
  char opt_local_B0_state;
  char opt_local_g_state;
  char opt_local_theta_state;
  unsigned char radharm;        /* =1 (default) converts radar scat. law for photoharm action */
  int rad_R_nharm;              /* max harmonic degree for photoharm action (default = 0) */
  int rad_C_nharm;
  unsigned char optharm;        /* =1 (default) converts opt. scat. law for photoharm action */
  int opt_R_nharm;              /* max harmonic degree for photoharm action (default = 0) */
  int opt_wt_nharm;
  int opt_A0_nharm;
  int opt_D_nharm;
  int opt_k_nharm;
  int opt_w_nharm;
  int opt_h_nharm;
  int opt_B0_nharm;
  int opt_g_nharm;
  int opt_theta_nharm;
  double rad_R_min;             /* min, max allowed values (default = 0 - inf) */
  double rad_R_max;
  double rad_C_min;             /* (default = 0 - inf) */
  double rad_C_max;
  double rad_rho_min;           /* (default = 0 - inf) */
  double rad_rho_max;
  double opt_R_min;             /* (default = 0 - inf) */
  double opt_R_max;
  double opt_wt_min;            /* (default = 0 - 1) */
  double opt_wt_max;
  double opt_A0_min;            /* (default = 0 - inf) */
  double opt_A0_max;
  double opt_D_min;             /* (default = 0 - inf) */
  double opt_D_max;
  double opt_k_min;             /* (default = -inf - 0) */
  double opt_k_max;
  double opt_w_min;             /* (default = 0 - 1) */
  double opt_w_max;
  double opt_h_min;             /* (default = 0 - inf) */
  double opt_h_max;
  double opt_B0_min;            /* (default = 0 - inf) */
  double opt_B0_max;
  double opt_g_min;             /* (default = -1 - 1) */
  double opt_g_max;
  double opt_theta_min;         /* (default = 0 - pi/2) */
  double opt_theta_max;
  double sun_appmag[MAXSUNMAGS];            /* Sun's apparent magnitude (default = -26.75 [V band]) */
  unsigned char vary_delcor0;   /* varies delcor0 jointly with size/shape/spin params        */
                                /*     if set to VARY_ALL or VARY_SIZE (default = VARY_NONE) */
  unsigned char vary_radalb;    /* varies radar R jointly with size/shape/spin params        */
                                /*     if set to VARY_ALL or VARY_SIZE (default = VARY_NONE) */
  unsigned char vary_optalb;    /* varies optical R or w jointly with size/shape/spin params */
                                /*     if set to VARY_ALL or VARY_SIZE (default = VARY_NONE) */
  unsigned char vary_dopscale;  /* varies Doppler scale factor jointly with size/shape/spin params */
                                /*     if set to VARY_ALL or VARY_SPIN (default = VARY_NONE) */
  double slice_long;            /* body-fixed longitude of slice normal for slice action (deg -> rad) */
  double slice_lat;             /* body-fixed latitude  of slice normal for slice action (deg -> rad) */
  double slice_offset;          /* perpendicular origin-slice distance  for slice action (km)  */
  double slice_viewlong;        /* body-fixed viewing longitude for slice action (deg -> rad) */
  double slice_viewlat;         /* body-fixed viewing latitude  for slice action (deg -> rad) */
  double slice_sunlong;         /* body-fixed solar longitude for slice action (deg -> rad) */
  double slice_sunlat;          /* body-fixed solar latitude  for slice action (deg -> rad) */
  unsigned char slice_scatlaw;  /* scattering law for slice action (default = lambertian) */
  unsigned char slice_shadows;  /* =1 includes shadowing in slice action */
  double slice_planefrac;       /* slice plane width / POS frame width (default = 0.85) */
  double slice_skyfactor;       /* slice bright / max bright for blank sky (default = 0.2) */
  double slice_dimfactor;       /* fractional surface dimming if behind slice (default = 0.2) */
  unsigned char slice_read;     /* =1 read rather than write intersection coords (default = 0) */
  unsigned char slice_dointerp; /* =1 use interpolation when displaying contour (default = 1) */
  double semimajor_axis[2];     /* orbit semimajor axis for binary system (km) */
  double r_pericenter[2];       /* pericenter distance for binary system (km) */
  double eccentricity[2];       /* eccentricity for binary system */
  double t_pericenter[2];       /* pericenter epoch for binary system (JD) */
  double long_asc_node[2];      /* equatorial longitude of ascending node for binary (deg -> rad) */
  double inclination[2];        /* equatorial inclination for binary system (deg -> rad) */
  double arg_pericenter[2];     /* equatorial argument of pericenter for binary system (deg -> rad) */
  double binary_gravparam[2];   /* gravitational parameter (GM) for binary system (km^3 day^-2) */
  double orbit_period[2];       /* orbit period for binary system (days) */
  unsigned char orbit_reflex;   /* =1 don't fix primary of binary system motionless */
  int is_triple;                /* =1 triple asteroid system */
  double orbit_posfactor[3];    /* length factors for POS image annotations (default = 1.0 1.0) */
  double write_posfactor;       /* length factor  for POS image annotations (default = 1.0) */
  double slice_posfactor;       /* length factor  for POS image annotations (default = 1.0) */
  double view_posfactor;        /* length factor  for POS image annotations (default = 1.0) */
  double map_posfactor;         /* length factor  for POS image annotations (default = 1.0) */
  char maskdir[80];             /* directory for pixel masks (default = data directory) */
  unsigned char write_chi2fit0; /* =1 write chi2 just for pixels with zero model power (default = 0) */
  double chi2fit0_thresh;       /* Threshold # sigmas for the write_chi2fit0 parameter (default = 0.0) */
  int npar_update;              /* update mod+obs files after fitting this many params (default = 20) */
  double split_norm[3];         /* unit normal n to dividing plane for split action */
  double split_const;           /* constant d defining plane for split action: n dot r + d = 0 */
  int map_set;                  /* dataset for map action */
  int map_frame;                /* frame   for map action */
  unsigned char map_mode;       /* determines mode of operation for map action */
  int map_dellim[2];            /* min, max pixel # in delay   for inverse map action (0-based) */
  int map_doplim[2];            /* min, max pixel # in Doppler for inverse map action (0-based) */
  int map_xposlim[2];           /* min, max POS column # for forward map action (0-based) */
  int map_yposlim[2];           /* min, max POS row    # for forward map action (0-based) */
  int map_comp[MAXCHOSENFACETS];  /* list of components for map action with map_mode = facets */
  int map_facet[MAXCHOSENFACETS]; /* list of facets     for map action with map_mode = facets */
  double map_fitcutoff;         /* threshold for turning map color on for fit (default = -9.99) */
  double map_poscutoff;         /* threshold for turning map color on for POS (default = -9.99) */
  unsigned char map_verbose;    /* =1 extra screen display for map action */
  unsigned char map_overflow;   /* =1 forward mapping extends beyond fit frame */
  double int_abstol;            /* abs. tolerance for integrating Euler's equations (default = 1e-7) */
  unsigned char pa_bodycoords;  /* =1 moments action images oriented along body coords (default = 0) */
  unsigned char pa_highlight;    /* =1 highlight selected facets for moments action (default = 0) */
  int pa_comp[MAXCHOSENFACETS];  /* list of components to highlight for moments action */
  int pa_facet[MAXCHOSENFACETS]; /* list of facets     to highlight for moments action */
  unsigned char view_highlight;    /* =1 highlight selected facets for view action (default = 0) */
  int view_comp[MAXCHOSENFACETS];  /* list of components to highlight for view action */
  int view_facet[MAXCHOSENFACETS]; /* list of facets     to highlight for view action */
  unsigned char write_highlight;    /* =1 highlight selected facets for write action (default = 0) */
  int write_comp[MAXCHOSENFACETS];  /* list of components to highlight for write action */
  int write_facet[MAXCHOSENFACETS]; /* list of facets     to highlight for write action */
  double dopscale_min;          /* min, max allowed values (default = 0 - inf) */
  double dopscale_max;
  double objfunc_start;         /* Initial objective function to use in deciding whether to stop a fit */
  unsigned char listpos_deldop; /* =1 writes (delay-)Doppler values in POS for (delay-)Doppler frames */
  unsigned char listpos_opt;    /* =1 writes optical brightness in POS for lightcurves, plane-of-sky */
  char listpos_path[255];       /* path for listpos_deldop, listpos_opt files (default = null string) */
  int nfpar;
  double **fpntr;               /* pointers to free parameters */
  double *fparstep;
  double *fpartol;
  double *fparabstol;
  int *fpartype;                /* type of parameters: shape, spin, etc. */
};


/*****************************************************************************************
                                DEFINITION OF struct mod_t
*****************************************************************************************/

/* Structure param_t defines a parameter state and value */
struct param_t {
  char state;                   /* free "f", constrained "c", or "=" */
  double val;                   /* value */
};

/* Structure vertex_t defines a single vertex. */
struct vertex_t {
  double a[3];                  /* "base point" of vertex */
  double u[3];                  /* direction of variation */
  struct param_t r;             /* distance along u */
  double x[3];                  /* coordinates */
  double n[3];                  /* normal at vertex */
  int naf;                      /* # of attached facets */
  int *af;                      /* list of attached facets */
  int nas;                      /* # of attached sides */
  int *as;                      /* list of attached sides */
  unsigned char act;            /* is vertex active or not? */
  double **afactor;             /* multiples "a" coefficients for harmonic realizations */
  double **bfactor;             /* multiples "b" coefficients for harmonic realizations */
  int v_mirror;                 /* vertex number of northern "mirror" vertex ('=' state) */
};

/* Structure facet_t defines a single surface facet. */
struct facet_t {
  int v[3];                     /* vertex indices (right-hand sense) */
  int s[3];                     /* indices of sides */
  double n[3];                  /* surface normal */
  unsigned char act;            /* facet active or not? */
  unsigned char seen;           /* facet ever seen from Earth and unshadowed? */
  double theta;                 /* colatitude (radians) */
  double phi;                   /* azimuth (radians) */
  double area;                  /* surface area */
  double x[3];                  /* mean coordinates of corner vertices */
};

/* Structure side_t defines a single facet side. */
struct side_t {
  int v[2];                     /* vertices which form endpoints */
  int f[2];                     /* attached facets */
  unsigned char act;            /* side active or not? */
};

/* The structure ellipse_t defines a triaxial ellipsoid. */
struct ellipse_t {
  int ntheta;                   /* # of theta steps */
  struct param_t two_a;         /* long diameter 2a (km) */
  struct param_t a_over_b;      /* long/intermediate axis ratio */
  struct param_t b_over_c;      /* intermediate/short axis ratio */
};

/* The structure ovoid_t defines a triaxial ovoid. */
struct ovoid_t {
  int ntheta;                   /* # of theta steps */
  struct param_t two_a;         /* long diameter 2a (km) */
  struct param_t a_over_b;      /* long/intermediate axis ratio */
  struct param_t b_over_c;      /* intermediate/short axis ratio */
  struct param_t k;             /* asymmetry parameter (-1 <= k <= 1) */
};

/* The structure harmonic_t defines a spherical harmonic series component. */
struct harmonic_t {
  int ntheta;                   /* # of theta steps */
  int nhar;                     /* # of harmonics */
  struct param_t scalefactor[3];
  struct param_t **a;           /* a coefficients */
  struct param_t **b;           /* b coefficients */
  double **a_save;
  double **b_save;
};

/* The structure vertices_t defines a vertex model. */
struct vertices_t {
  struct param_t scalefactor[3];
  int nv;                       /* # of vertices */
  struct vertex_t *v;           /* vertices */
  int nf;                       /* # of facets */
  struct facet_t *f;            /* facets */
  int ns;                       /* # of sides */
  struct side_t *s;             /* sides */
};

/* Structure comp_t describes a single component of a 3D model. */
#define ELLIPSE 1
#define OVOID 2
#define HARMONIC 3
#define VERTEX 4
struct comp_t {
  unsigned char type;           /* type of component description */
  struct param_t off[3];        /* linear translation */
  struct param_t rot[3];        /* Euler angles */
  double m[3][3];               /* matrix corresponding to Euler angles */
  union desc_t {                /* shape description */
    struct ellipse_t ell;
    struct ovoid_t ovoid;
    struct harmonic_t har;
    struct vertices_t ver;
  } desc;
  struct vertices_t real;       /* realization of shape description */
  double inertia[3][3];         /* inertia tensor */
  double com[3];                /* center of mass */
  double volume;
  double area;
};

/* Struture shape_t describes a 3D shape as composed of one or more */
/* components. */
struct shape_t {
  int ncomp;                    /* # of components */
  struct comp_t *comp;          /* pointers to component descriptions */
  double inertia[3][3];         /* inertia tensor */
  double com[3];                /* center of mass */
  double volume;
  double area;
};

/* Radar scattering law types. */
#define NOLAW 0

#define COSINELAW_DIFF 1
struct RC_t {
  struct param_t R;
  struct param_t C;
  double R_save;
};

#define GAUSSIANLAW 2
#define HAGFORSLAW 3
#define COSINELAW_QS 4
struct quasispec_t {
  struct param_t R;
  struct param_t C;
  double cos_cutoff;
  double R_save;
};

#define GAUSSIAN_COSINE 5
#define HAGFORS_COSINE 6
#define COSINE_COSINE 7
struct hybridradar_t {
  struct quasispec_t qs;
  struct RC_t diff;
};

#define HARMCOSINE_DIFF 8
struct harmcosine_t {
  struct harmonic_t R;
  struct harmonic_t C;
  struct RC_t **local;
};

#define INHOCOSINE_DIFF 9
struct inhocosine_t {
  struct RC_t global;
  struct RC_t **local;
};

#define TABULARLAW 10
struct tabular_t {
  int n;
  struct param_t *rho;
  double *rho_save;
};

union radscat_t {
  struct RC_t RC;
  struct tabular_t tabular;
  struct quasispec_t quasispec;
  struct hybridradar_t hybrid;
  struct harmcosine_t harmcosine;
  struct inhocosine_t inhocosine;
};

/* Optical scattering law types. */
#define GEOMETRICAL 1
#define LAMBERTLAW 2
#define LOMMEL 3
struct R_t {
  struct param_t R;
  double R_save;
};

#define HARMLAMBERT 4
#define HARMLOMMEL 5
struct harmR_t {
  struct harmonic_t R;
  struct R_t **local;
};

#define INHOLAMBERT 6
#define INHOLOMMEL 7
struct inhoR_t {
  struct R_t global;
  struct R_t **local;
};

#define HAPKE 8
struct hapke_t {
  struct param_t w;             /* single scattering albedo */
  struct param_t h;             /* large h -> wide opposition surge (dimensionless) */
  struct param_t B0;            /* opposition surge amplitude */
  struct param_t g;             /* asymmetry factor for Henyey-Greenstein phase function */
  struct param_t theta;         /* average topographic slope (deg) */
  double w_save;
};

#define HARMHAPKE 9
struct harmhapke_t {
  struct harmonic_t w;          /* see hapke_t for parameter descriptions */
  struct harmonic_t h;
  struct harmonic_t B0;
  struct harmonic_t g;
  struct harmonic_t theta;
  struct hapke_t **local;
};

#define INHOHAPKE 10
struct inhohapke_t {
  struct hapke_t global;
  struct hapke_t **local;
};

#define KAASALAINEN 11
struct kaas_t {
  struct param_t R;             /* overall brightness scale factor */
  struct param_t wt;            /* Lambert weight w.r.t. Lommel-Seeliger (0 <= wt <= 1) */
  struct param_t A0;            /* opposition surge amplitude */
  struct param_t D;             /* opposition surge scale length (deg) */
  struct param_t k;             /* overall slope of phase function (1/deg) */
  double R_save;
};

#define HARMKAAS 12
struct harmkaas_t {
  struct harmonic_t R;          /* see kaas_t for parameter descriptions */
  struct harmonic_t wt;
  struct harmonic_t A0;
  struct harmonic_t D;
  struct harmonic_t k;
  struct kaas_t **local;
};

#define INHOKAAS 13
struct inhokaas_t {
  struct kaas_t global;
  struct kaas_t **local;
};

union optscat_t {
  struct R_t R;
  struct harmR_t harmR;
  struct inhoR_t inhoR;
  struct hapke_t hapke;
  struct harmhapke_t harmhapke;
  struct inhohapke_t inhohapke;
  struct kaas_t kaas;
  struct harmkaas_t harmkaas;
  struct inhokaas_t inhokaas;
};

/* Structure photo_t describes the radar and optical photometric properties */
/* of the surface. */
struct photo_t {
  int nradlaws;                 /* number of radar scattering laws */
  unsigned char *radtype;       /* type(s) of radar scattering law */
  union radscat_t *radar;
  int noptlaws;                 /* number of optical scattering laws */
  unsigned char *opttype;       /* type of optical scattering law */
  union optscat_t *optical;
};

#define MAXIMP 10

/* Structure spin_t describes a general spin state via initial conditions */
/* at time t0. */
struct spin_t {
  unsigned char pa;             /* =0 if npa, =1 if pa */
  double t0;                    /* time of initial conditions */
  struct param_t angle[3];      /* Euler angles of principal axes */
  struct param_t omega[3];      /* initial spin vector */
  struct param_t omegadot[3];   /* change in spin rate vector */
  int n_impulse;                /* number of spin impulses */
  double t_impulse[MAXIMP];     /* times of spin impulses */
  struct param_t impulse[MAXIMP][3]; /* spin impulses */
  struct param_t inertia[3];    /* moments of inertia */
  struct param_t lib_amp;       /* amplitude of libration*/
  struct param_t lib_freq;      /* frequency of libration*/
  struct param_t lib_phase;     /* initial phase of libration*/
};

/* Structure mod_t gives a complete description of a physical asteroid */
/* model as three parts: shape, photometric, spin state. */
struct mod_t {
  int nfpar;                    /* # of free parameters */
  struct shape_t shape;         /* shape description */
  struct photo_t photo;         /* photometric description */
  struct spin_t spin;           /* spin state */
  char name[80];                /* mod file name */
};


/*****************************************************************************************
                                DEFINITION OF struct dat_t
*****************************************************************************************/

/* A delay correction polynomial */
struct delcor_t {
  double t0;
  int n;
  struct param_t *a;
  double delcor0_save;
  int *equals_set;
};

/* An ephemeris point */
struct ephpnt_t {
  double t;                     /* Julian date */
  double ra;                    /* radians */
  double dec;                   /* radians */
  double dist;                  /* AU */
};

/* An ephemeris */
struct ephem_t {
  int n;
  struct ephpnt_t *pnt;
};

/* Structure pos_t describes a plane-of-sky image */
struct pos_t {
  int n;                        /* number of >0 pixels (2n+1 in all) */
  double km_per_pixel;          /* linear pixel size (km) */
  double **b;                   /* brightness */
  double **cose;                /* cos(emission) */
  double **cosi;                /* cos(incidence) */
  double **z;                   /* distance toward observer */
  int **body;                   /* index of corresponding model body */
  int **comp;                   /* index of corresponding model component */
  int **f;                      /* index of corresponding surface facet */
  int xlim[2];                  /* x range of nonzero region */  
  int ylim[2];                  /* y range of nonzero region */
  int bistatic;                 /* =1 if need following matrices */
  double **cosill;              /* cos(incidence) in source point of view*/
  double **zill;                /* distance toward src */
  int **bodyill;                /* index of corresponding model body */
  int **compill;                /* index of corresponding model component */
  int **fill;                   /* index of corresponding surface facet */
  int xlim2[2];                 /* x range of nonzero region */  
  int ylim2[2];                 /* y range of nonzero region */
  double ae[3][3];              /* ecliptic to asteroid transformation */
  double oe[3][3];              /* ecliptic to observer transformation */
  double se[3][3];              /* ecliptic to source transformation */
  double posbnd_logfactor;      /* multiples obj. fcn. if POS image exceeds window */
  int nf;                       /* # of facets */
};

/* Structure deldopview_t describes a single view that contributes to a smeared delay-Doppler frame. */
struct deldopview_t {
  double t;                     /* 1-way light-time corrected time */
  double ae[3][3];              /* trans. from eclip. to ast. coords */
  double oe[3][3];              /* trans. form eclip. to obs. coords */
  double intspin[3];            /* intrinsic spin vector (ecl. co.) */
  double orbspin[3];            /* orbital contribution to spin (ecl. co.) */
  double spin[3];               /* total spin (ecl. co.) */
  int idellim[2];               /* delay limits of region of interest (rows) */
  int idoplim[2];               /* doppler limits of region of interest (cols) */
  double dellim[2];             /* delay limits of region of interest (usec) */
  double doplim[2];             /* doppler limits of region of interest (Hz) */
  double deloff;                /* delay offset (bins) */
  double dopoff;                /* doppler offset (bins) */
  double km2Hz;                 /* km to Hz conversion factor */
};

/* Structure deldopfrm_t describes a single delay-Doppler frame. */
struct deldopfrm_t {
  double t0;                    /* time */
  int n_integrate;              /* # of times from spin epoch t0 to t (inclusive), given impulses */
  double t_integrate[MAXIMP+2]; /* times from initial spin epoch t0 to t (inclusive), given impulses */
  double impulse[MAXIMP+2][3];  /* spin impulse at each time in t_integrate */
  double nlooks;                /* # of incoherent looks */
  double weight;                /* relative weight given to this frame */
  int pixels_weighted;          /* =1 read pixel-weighting mask for this frame */
  double sdev;                  /* sdev of receiver noise */
  double delcom;                /* ephemeris delay bin of COM (1-based) */
  struct param_t cal;           /* sdev calibration */
  int ndel;                     /* # of delay bins after vignetting */
  int ndop;                     /* # of doppler bins after vignetting */
  double delcom_vig;            /* ephemeris delay bin of COM (1-based) after vignetting */
  double dopcom_vig;            /* ephemeris doppler bin of COM (1-based) after vignetting */
  double dopDC_vig;             /* Doppler bin of DC (1-based) after vignetting */
  double **obs;                 /* observed image */
  double **fit;					/* fit image */
  double **oneovervar;          /* 1/variance of pixels */
  struct pos_t pos;             /* POS image */
  struct deldopview_t *view;    /* individual views contributing to smeared frame */
  char name[40];                /* name of data file */
  int idellim[2];               /* delay limits of region of interest (rows) */
  int idoplim[2];               /* doppler limits of region of interest (cols) */
  double dellim[2];             /* delay limits of region of interest (usec) */
  double doplim[2];             /* doppler limits of region of interest (Hz) */
  double idelvig[2];            /* delay limits, within unvig. image, of vig. image (rows) */
  double idopvig[2];            /* Doppler limits, within unvig. image, of vig. image (cols) */
  double chi2;                  /* chi2 error of this fit */
  double overflow_o2;           /* overflow obs^2 contribution to chi squared */
  double overflow_m2;           /* overflow model^2 contribution to chi squared */
  double overflow_xsec;         /* overflow summed cross section */
  double overflow_delmean;      /* overflow mean delay bin */
  double overflow_dopmean;      /* overflow mean Doppler bin */
  double dof;                   /* degrees of freedom */
  double badradar_logfactor;    /* multiples obj. fcn. if model too wide in delay-Doppler space */
  double **map_fit;             /* fit image needed for the map action */
  double **map_pos;             /* POS image needed for the map action */
  double **map_facet_power;     /* array of facet contributions needed for the map action */
  /* The following are used for models where we want to combine multiple sets with single frames
   * into fewer sets with multiple frames    */
  struct param_t angleoff[3];   /* Euler angles offsets */
  struct param_t omegaoff[3];   /* spin vector offsets */
  struct delcor_t delcor;		/* Delay correction polynomial */
  double Ftx;					/* Transmitter frequency */
  double del_per_pixel;			/* Delay resolution */
  double dop_per_pixel;			/* Doppler resolution */
  double dopDC;					/* Doppler bin of COM (1-based) */
  double dopfftlen;				/* Doppler fft length */
  double dopcom;                /* ephemeris doppler bin of COM (1-based) */
  struct param_t dopscale;      /* Doppler scaling factor */
  double dopscale_save;
  int nviews;                   /* # of views per frame */
  double view_interval;         /* time interval between views (secs, convert internally to days) */
  unsigned char smearing_mode;  /* =0 frame epoch is center view, =1 frame epoch is first view */
  int v0;                       /* # of the view corresponding to the frame epoch (0-based) */
  int codemethod;               /* code+reduction combination used */
  int spb;                      /* # samples per baud in original data */
  int stride;                   /* # spb between adjacent image rows */
};

/* Structure deldop_t describes a delay-Doppler data set. */
struct deldop_t {
  int iradlaw;                  /* # of the radar scattering law used with this set (0-based) */
  struct ephem_t astephem;      /* ephemeris of asteroid */
  double Ftx;                   /* transmitter frequency (MHz) */
  struct delcor_t delcor;       /* polynomial for delay correction (usec, usec/day, ...) */
  int ndel;                     /* # of delay bins */
  int ndop;                     /* # of doppler bins */
  double del_per_pixel;         /* pixel height = baud/(spb/stride) (usec) */
  double dop_per_pixel;         /* pixel width (Hz) */
  double dopcom;                /* ephemeris doppler bin of COM (1-based) */
  double dopDC;                 /* Doppler bin of DC (1-based) */
  int codemethod;               /* code+reduction combination used */
  int spb;                      /* # samples per baud in original data */
  int stride;                   /* # spb between adjacent image rows */
  int dopfftlen;                /* Doppler fft length */
  struct param_t dopscale;      /* Doppler scaling factor */
  double dopscale_save;
  int nviews;                   /* # of views per frame */
  double view_interval;         /* time interval between views (secs, convert internally to days) */
  unsigned char smearing_mode;  /* =0 frame epoch is center view, =1 frame epoch is first view */
  int v0;                       /* # of the view corresponding to the frame epoch (0-based) */
  int nframes;                  /* # of frames */
  struct deldopfrm_t *frame;    /* pointers to frames of data */
  char dir[80];                 /* directory of data files */
  double dof;                   /* degrees of freedom */
  double sum_deldop_zmax_weights;     /* weight sum for the vary_delcor0 parameter */
  double sum_rad_xsec_weights;        /* weight sum for the vary_radalb parameter */
  double sum_cos_subradarlat_weights; /* weight sum for the vary_dopscale parameter */
  double fit_overflow[MAXOVERFLOW][MAXOVERFLOW]; /* Overflow buffer for host multi-threaded mode.  Each set gets its own */
};

/* Structure dopview_t describes a single view that contributes to a smeared Doppler frame. */
struct dopview_t {
  double t;                     /* 1-way light-time corrected time */
  double ae[3][3];              /* trans. from eclip. to ast. coords */
  double oe[3][3];              /* trans. form eclip. to obs. coords */
  double intspin[3];            /* intrinsic spin vector (ecl. co.) */
  double orbspin[3];            /* orbital contribution to spin (ecl. co.) */
  double spin[3];               /* total spin (ecl. co.) */
  int idoplim[2];               /* doppler limits of region of interest (bins) */
  double doplim[2];             /* doppler limits of region of interest (Hz) */
  double dopoff;                /* doppler offset (bins) */
  double km2Hz;                 /* km to Hz conversion factor */
};

/* Structure dopfrm_t describes a single Doppler "frame," i.e., spectrum */
struct dopfrm_t {
  double t0;                    /* time */
  int n_integrate;              /* # of times from spin epoch t0 to t (inclusive), given impulses */
  double t_integrate[MAXIMP+2]; /* times from initial spin epoch t0 to t (inclusive), given impulses */
  double impulse[MAXIMP+2][3];  /* spin impulse at each time in t_integrate */
  double nlooks;                /* # of incoherent looks */
  double weight;                /* relative weight given to this frame */
  int pixels_weighted;          /* =1 read pixel-weighting mask for this frame */
  double sdev;                  /* sdev of receiver noise */
  struct param_t cal;           /* sdev calibration */
  int ndop;                     /* # of doppler bins after vignetting */
  double dopcom_vig;            /* ephemeris doppler bin of COM (1-based) after vignetting */
  double *obs;                  /* observed image */
  double *fit;                  /* fit image */
  double *oneovervar; 			/* 1/variance of pixels */
  float *fit_s;					/* but single precision for CUDA atomics  		*/
  struct pos_t pos;             /* POS image */
  struct dopview_t *view;       /* individual views contributing to smeared frame */
  char name[40];                /* name of data file */
  int idoplim[2];               /* doppler limits of region of interest (bins) */
  double doplim[2];             /* doppler limits of region of interest (Hz) */
  double idopvig[2];            /* Doppler limits, within unvig. spec., of vig. spec. (bins) */
  double chi2;                  /* chi2 error of this fit */
  double overflow_o2;           /* overflow obs^2 contribution to chi squared */
  double overflow_m2;           /* overflow model^2 contribution to chi squared */
  double overflow_xsec;         /* overflow summed cross section */
  double overflow_dopmean;      /* overflow mean Doppler bin */
  double dof;                   /* degrees of freedom */
  double badradar_logfactor;    /* multiples obj. fcn. if model too wide in Doppler space */
  double *map_fit;              /* fit spectrum needed for the map action */
  double **map_pos;             /* POS image needed for the map action */
  double **map_facet_power;     /* array of facet contributions needed for the map action */
 };

/* Structure doppler_t describes a Doppler data set. */
struct doppler_t {
  int iradlaw;                  /* # of the radar scattering law used with this set (0-based) */
  struct ephem_t astephem;      /* ephemeris of asteroid */
  double Ftx;                   /* transmitter frequency (MHz) */
  struct delcor_t delcor;       /* polynomial for delay correction (usec, usec/day, ...) */
  int ndop;                     /* # of doppler bins */
  double dop_per_bin;           /* bin width (Hz) */
  double dopcom;                /* ephemeris doppler bin of COM (1-based) */
  struct param_t dopscale;      /* Doppler scaling factor */
  double dopscale_save;
  int nviews;                   /* # of views per frame */
  double view_interval;         /* time interval between views (secs, convert internally to days) */
  unsigned char smearing_mode;  /* =0 frame epoch is center view, =1 frame epoch is first view */
  int v0;                       /* # of the view corresponding to the frame epoch (0-based) */
  int nframes;                  /* # of frames */
  struct dopfrm_t *frame;       /* pointers to frames of data */
  char dir[80];                 /* directory of data files */
  double dof;                   /* degrees of freedom */
  double sum_rad_xsec_weights;        /* weight sum for the vary_radalb parameter */
  double sum_cos_subradarlat_weights; /* weight sum for the vary_dopscale parameter */
  double fit_overflow[MAXOVERFLOW];	/* This is intended for host-multithreaded application.  Each set gets its own overflow buffer */
};

/* Structure posetview_t describes a single view that contributes to a smeared POS frame. */
struct posetview_t {
  double t;                     /* 1-way light-time corrected time */
  double ae[3][3];              /* trans. from eclip. to ast. coords */
  double oe[3][3];              /* trans. form eclip. to obs. coords */
  double se[3][3];              /* trans. from eclip. to src. coords */
  double intspin[3];            /* intrinsic spin vector (ecl. co.) */
  double orbspin[3];            /* orbital contribution to spin (ecl. co.) */
  double spin[3];               /* total spin (ecl. co.) */
  double solar_phase;           /* solar phase angle */
  double solar_azimuth;         /* ast-to-sun direc (N->E in POS) */
};

/* Structure posetfrm_t describes a single POS frame. */
struct posetfrm_t {
  double t0;                    /* time */
  int n_integrate;              /* # of times from spin epoch t0 to t (inclusive), given impulses */
  double t_integrate[MAXIMP+2]; /* times from initial spin epoch t0 to t (inclusive), given impulses */
  double impulse[MAXIMP+2][3];  /* spin impulse at each time in t_integrate */
  double weight;                /* relative weight given to this frame */
  int pixels_weighted;          /* =1 read pixel-weighting mask for this frame */
  double northangle;            /* direction of north: 0 = up, pi/2 = right */
  struct param_t cal;           /* calibration factor */
  int nrow;                     /* # of image rows after vignetting */
  int ncol;                     /* # of image columns after vignetting */
  double rowcom_vig;            /* row # (1-based) of COM in vignetted image if off[1]=0 */
  double colcom_vig;            /* column # (1-based) of COM in vignetted image if off[0]=0 */
  struct pos_t obs;             /* observed image */
  struct pos_t fit;             /* fit image */
  struct pos_t pos;             /* POS image */
  double **oneovervar;          /* 1/variance of pixels */
  struct posetview_t *view;     /* individual views contributing to smeared frame */
  char name[40];                /* name of data file */
  struct param_t off[2];        /* x,y offset of COM (bins) */
  double chi2;                  /* chi2 error of this fit */
  double dof;                   /* degrees of freedom */
};

/* Structure poset_t describes a plane-of-sky data set. */
struct poset_t {
  int ioptlaw;                  /* # of the optical scattering law used with this set (0-based) */
  struct ephem_t astephem;      /* ephemeris of asteroid */
  struct ephem_t solephem;      /* ephemeris of sun */
  double angle_per_pixel;       /* angular pixel size (arcsec) */
  int nviews;                   /* # of views per frame */
  double view_interval;         /* time interval between views (secs, convert internally to days) */
  unsigned char smearing_mode;  /* =0 frame epoch is center view, =1 frame epoch is first view */
  int v0;                       /* # of the view corresponding to the frame epoch (0-based) */
  int nframes;                  /* # of frames */
  struct posetfrm_t *frame;     /* pointers to frames of data */
  char dir[80];                 /* directory of data files */
  double dof;                   /* degrees of freedom */
};

/* everything you need for a single point of a lightcurve */
struct crvrend_t {
  struct pos_t pos;             /* plane of sky rendering */
  double ae[3][3];              /* ecliptic to asteroid transformation */
  double oe[3][3];              /* ecliptic to observer transformation */
  double se[3][3];              /* ecliptic to source transformation */
  double intspin[3];            /* intrinsic spin vector (ecl. co.) */
  double orbspin[3];            /* orbital contribution to spin (ecl. co.) */
  double spin[3];               /* total spin (ecl. co.) */
  int n_integrate;              /* # of times from spin epoch t0 to t (inclusive), given impulses */
  double t_integrate[MAXIMP+2]; /* times from initial spin epoch t0 to t (inclusive), given impulses */
  double impulse[MAXIMP+2][3];  /* spin impulse at each time in t_integrate */
};

/* lghtcrv_t describes a lightcurve data set */
struct lghtcrv_t {
  int ioptlaw;                  /* # of the optical scattering law used with this set (0-based) */
  struct ephem_t astephem;      /* ephemeris of asteroid */
  struct ephem_t solephem;      /* ephemeris of sun */
  struct ephem_t astephem_sun;	/* ephemeris of asteroid as viewed from sun */
  struct ephem_t obsephem_sun;	/* ephemeris of observer as viewed from sun */
  int nviews;                   /* # of views per point */
  double view_interval;         /* time interval between views (secs, convert internally to days) */
  unsigned char smearing_mode;  /* =0 point epoch is center view, =1 point epoch is first view */
  int v0;                       /* # of the view corresponding to the point epoch (0-based) */
  int n;                        /* # of observed data */
  double weight;                /* relative weight given to this lightcurve */
  double *t0;                   /* times of observed points */
  double **t;                   /* time of each view for observed points, 1-way light-time corrected */
  double *obs;                  /* observed magnitudes */
  double *oneovervar;           /* 1/variance of noise*/
  double *fit;                  /* fit */
  double *rotphase_calc;        /* rotation phase at calculated epochs (deg) */
  double *rotphase_obs;         /* rotation phase at observation epochs (deg) */
  struct param_t cal;           /* calibration offset */
  char name[80];                /* full name of file */
  /* The following are for use with NR in C routines spline and splint */
  int ncalc;                    /* # of calculated points */
  double *x0;                   /* times of calculated points */
  double *x;                    /* times, 1-way light-time corrected */
  double *y;                    /* calculated points for interpolation */
  double *y2;                   /* 2nd derivates of calc. pnts. */
  double *solar_phase;          /* solar phase angles of calculated points */
  double *solar_azimuth;        /* ast-to-sun direc (N->E in POS) for calc points */
  struct crvrend_t *rend;       /* rendering */
  double dof;                   /* degrees of freedom */
  double sum_opt_brightness_weights;  /* weight sum for the vary_optalb parameter */
  int ncalc_obsfile;            /* # of calculated points as listed in obs file */
  double jdstart;               /* start JD for calculated lightcurve points */
  double jdstop;                /* stop  JD for calculated lightcurve points */
  double jdinterval;            /* JD interval between calculated lightcurve points */
};

/* Structure set_t describes a general set. */
#define DELAY 0
#define DOPPLER 1
#define POS 2
#define LGHTCRV 3
struct set_t {
  unsigned char type;           /* type of set */
  union setdesc_t {
    struct deldop_t deldop;
    struct doppler_t doppler;
    struct poset_t poset;
    struct lghtcrv_t lghtcrv;
  } desc;
  double chi2;                  /* chi2 error of fit for this set */
  double dof;                   /* degrees of freedom */
  int inputnode;                /* mpi node listed in the input obs file */
  struct param_t angleoff[3];   /* Euler angles offsets */
  struct param_t omegaoff[3];   /* spin vector offsets */
};

/* Structure dat_t.h describes a complete radar/optical data set. */
struct dat_t {
  struct pos_t pos;             /* POS image */
  int nsets;                    /* # of sets */
  struct set_t *set;            /* pointers to sets */
  double chi2;                  /* chi2 error of fit */
  double chi2_variance;         /* variance of chi2 estimate */
  double dof;                   /* degrees of freedom */
  double dof_deldop;
  double dof_doppler;
  double dof_poset;
  double dof_lghtcrv;
  double sum_deldop_zmax_weights;      /* weight sum for the vary_delcor0 parameter */
  double sum_rad_xsec_weights;         /* weight sum for the vary_radalb parameter */
  double sum_opt_brightness_weights;   /* weight sum for the vary_optalb parameter */
  double sum_cos_subradarlat_weights;  /* weight sum for the vary_dopscale parameter */
  char name[40];                /* dat file name */
};


/*****************************************************************************************
                                SHAPE 2.0 LIBRARY ROUTINES
*****************************************************************************************/

void annotate_plot( struct par_t *par, struct mod_t *mod, double spin_ecl[3],
                    double maxbrightness, double posmax, struct pos_t *pos,
                    int **color, double **brightness, double **z);
double apply_photo( struct mod_t *mod, int ilaw, double phase, double intensityfactor,
                    struct pos_t *pos, int body, int s, int frm);
void bailout(const char *message);
double bestfit(struct par_t *par, struct mod_t *mod, struct dat_t *dat);
void branch( struct par_t *par, struct mod_t *mod, struct dat_t *dat);
void calc_fits( struct par_t *par, struct mod_t *mod, struct dat_t *dat);
void calc_orbit( struct par_t *par,
                 struct mod_t *mod1, struct mod_t *mod2, struct mod_t *mod3,
                 struct dat_t *dat1, struct dat_t *dat2, struct dat_t *dat3);
double chi2( struct par_t *par, struct dat_t *dat, int list_breakdown);
void convex_hull( struct par_t *par, struct mod_t *mod);
void covar( struct par_t *par, struct mod_t *mod, struct dat_t *dat);
void delcorinit( struct par_t *par, struct mod_t *mod, struct dat_t *dat);
void deldopoffs( struct deldop_t *deldop);
void diag_inertia( double inertia[3][3], double pmoment[3], double ap[3][3]);
void dopoffs( struct doppler_t *doppler);
double ephem2mat( struct ephem_t ast, struct ephem_t sunn, double t,
                  double oe[3][3], double se[3][3], double s[3],
                  double *solar_phase, double *solar_azimuth,
                  int bistatic);
void facmom( double fv0[3], double fv1[3], double fv2[3], double fn[3],
             double *dv, double dvr[3], double dI[3][3]);
double facnrm( struct vertices_t verts, int fi);
int gamma_trans( double *datum, double gamma);
void get_calfact( struct dat_t *dat);
void init( int argc, char *argv[], char *progname);
void inteuler( struct spin_t spin, double t[], double impulse[][3], int n,
               double w[3], double m[3][3], unsigned char pa, unsigned char method,
               double int_abstol);
void map_radar( struct par_t *par, struct mod_t *mod, struct dat_t *dat);
void merge_comps( struct par_t *par, struct mod_t *mod);
void mirror_dat( struct dat_t *dat);
void mirror_mod( struct mod_t *mod);
void mkparlist( struct par_t *par, struct mod_t *mod, struct dat_t *dat);
double penalties( struct par_t *par, struct mod_t *mod, struct dat_t *dat);
void photofacets( struct par_t *par, struct mod_t *mod);
void photoharm( struct par_t *par, struct mod_t *mod);
void plot_arrow( int colortype, double com_obs[3], double arrowvec[3], int has_head,
                 int pierce_model, double sizefactor, double maxbrightness, double posmax,
                 struct pos_t *pos, int **color, double **brightness, double **zval);
void plot_com( int colortype, int body, double com_obs[3], double sizefactor,
               double maxbrightness, struct pos_t *pos, int **color,
               double **brightness, double **z);
void plot_subradar( int colortype, int body, double sizefactor, double maxbrightness,
                    double posmax, struct pos_t *pos, int **color, double **brightness,
                    double **z);
void plot_surface( struct par_t *par, struct mod_t *mod, unsigned char scatlaw,
                   int iradlaw, char *name, double *maxbrightness, double *posmax,
                   struct pos_t *pos, int **color, double **brightness, double **z);
int pos2deldop( struct par_t *par, struct photo_t *photo,
                double orbit_xoff, double orbit_yoff, double orbit_dopoff,
                struct deldop_t *deldop, int body, int set, int frm, int v);
int pos2doppler( struct par_t *par, struct photo_t *photo,
                 double orbit_xoff, double orbit_yoff, double orbit_dopoff,
                 struct doppler_t *doppler, int body, int set, int frm, int v);
void posclr( struct pos_t *pos);
void posmask( struct pos_t *pos, double tol);
int posvis( struct vertices_t *verts, double orbit_offset[3], struct pos_t *pos,
            int smooth, int src, int body, int comp);
void proj_area( struct par_t *par, struct mod_t *mod);
double radlaw( struct photo_t *photo, int ilaw, double cosinc, int c, int f);
int rayfacint( double *r, double *s, double *t, double u[3], double a[3],
               double v0[3], double v1[3], double v2[3], double facetnorm[3],
               double tol);
int read_dat( struct par_t *par, struct mod_t *mod, struct dat_t *dat);
void read_ephem( struct par_t *par, struct ephem_t *ephem);
int read_mod( struct par_t *par, struct mod_t *mod);
void read_par( char *name, struct par_t *par);
int readparam( FILE *fp, struct param_t *p);
void realize_angleoff( struct dat_t *dat);
void realize_delcor( struct dat_t *dat, double delta_delcor0, int delcor0_mode);
void realize_dopscale( struct par_t *par, struct dat_t *dat, double dopscale_factor,
                       int dopscale_mode);
void realize_mod( struct par_t *par, struct mod_t *mod);
void realize_omegaoff( struct dat_t *dat);
void realize_photo( struct par_t *par, struct mod_t *mod,
                    double radalb_factor, double optalb_factor, int albedo_mode);
void realize_spin( struct par_t *par, struct mod_t *mod, struct dat_t *dat);
void realize_xyoff( struct dat_t *dat);
void ref_mod( struct mod_t *mod);
void rescale( struct par_t *par, struct mod_t *mod);
void sample_mod( struct par_t *par, struct mod_t *mod);
void setupsides( struct vertices_t *vt);
void setupvertices( struct vertices_t *vt);
void slice( struct par_t *par, struct mod_t *mod);
void show_deldoplim( struct dat_t *dat);
void show_moments( struct par_t *par, struct mod_t *mod);
void split_mod( struct par_t *par, struct mod_t *mod);
void vary_params( struct par_t *par, struct mod_t *mod, struct dat_t *dat,
                  int action, double *deldop_zmax, double *rad_xsec,
                  double *opt_brightness, double *cos_subradarlat);
void view_mod( struct par_t *par, struct mod_t *mod);
void write_dat( struct par_t *par, struct dat_t *dat);
void write_mod( struct par_t *par, struct mod_t *mod);
void write_pnm( double posmax, int n, int **color, double **brightness,
                int color_output, char *name);
void write_pos( struct par_t *par, struct mod_t *mod, struct pos_t *pos,
                double spin_ecl[3], int iradlaw, int color_output, char *name);
void write_wf( struct mod_t *mod);
void initialize_delcor( struct dat_t *dat);
void deldopoffs_MFS_initial( struct deldop_t *deldop);
void set_up_pos( struct par_t *par, struct dat_t *dat);
void read_deldop_ascii(FILE *fin, struct deldop_t *deldop, int iframe,
		int idel_use[2], int idop_use[2]);
void read_deldop_binary(FILE *fin, struct deldop_t *deldop, int iframe,
		int idel_use[2], int idop_use[2],  int swap_bytes);
void read_deldop_rdf(FILE *fin, struct deldop_t *deldop, int iframe,
		int idel_use[2], int idop_use[2],  int swap_bytes);
void read_deldop_fits(char *filename, struct deldop_t *deldop, int iframe,
		int idel_use[2], int idop_use[2]);
void read_doppler_ascii(FILE *fin, struct doppler_t *doppler, int iframe,
		int idop_use[2]);
void read_doppler_binary(FILE *fin, struct doppler_t *doppler, int iframe,
		int idop_use[2], int swap_bytes);
void read_doppler_rdf(FILE *fin, struct doppler_t *doppler, int iframe,
		int idop_use[2], int swap_bytes);
void read_doppler_fits(char *filename, struct doppler_t *doppler, int iframe,
		int idop_use[2]);
void read_poset_fits(char *filename, struct poset_t *poset, int iframe,
		int irow_use[2], int icol_use[2], int read_data);

void calc_deldop( struct par_t *par, struct mod_t *mod, struct deldop_t *deldop,
		int s);
void calc_doppler( struct par_t *par, struct mod_t *mod, struct doppler_t *doppler,
		int s);
void calc_poset( struct par_t *par, struct mod_t *mod, struct poset_t *poset,
		int s);
void calc_lghtcrv( struct par_t *par, struct mod_t *mod, struct lghtcrv_t *lghtcrv,
		int s);
void write_pos_deldop( struct par_t *par, struct mod_t *mod,
		struct deldop_t *deldop, int s, int f);
void write_pos_doppler( struct par_t *par, struct mod_t *mod,
		struct doppler_t *doppler, int s, int f);
void write_pos_poset( struct par_t *par, struct mod_t *mod,
		struct poset_t *poset, int s, int f);
void write_pos_lghtcrv( struct par_t *par, struct mod_t *mod,
		struct lghtcrv_t *lghtcrv, int s, int i);

void realize_impulse( struct spin_t spin, double t, double t_integrate[],
		double impulse[][3], int *n_integrate);
void realize_angleoff( struct dat_t *dat);
void realize_omegaoff( struct dat_t *dat);
void checkphotopar( double parval, double parmin, double parmax, int mode,
                    unsigned char *badphoto, double *badphoto_logfactor);
