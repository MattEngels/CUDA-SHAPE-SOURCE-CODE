/* Modified 2015 June 10 by CM to add SMEAR_CENTER and SMEAR_FIRST */
/* Modified 2013 July 5 by CM to add PA3TILT */
/* Modified 2012 March 23 by CM to add VARY_SPIN and DOPSCALEPAR */
/* Modified 2011 August 22 by CM to add IMPULSE */
/* Modified 2010 August 26 by CM to add MAPMODE_DELDOP, MAPMODE_POS, MAPMODE_FACETS,
       and TINYCALFACT */
/* Modified 2010 June 15 by CM to replace RESCALE by MAP */
/* Modified 2010 May 20 by CM to add SPLIT */
/* Modified 2010 April 27 by CM to add NONCOSINE and BIFURCATION */
/* Modified 2009 April 27 by CM to add SOLAR_VMAG */
/* Modified 2008 December 15 by CM to add PIOVER2 and TWOPI */
/* Modified 2007 August 4 by CM to add ORBIT and GRAVCONST and delete MAXFACETS */
/* Modified 2007 February 21 by CM to add MAXRDEV and MAXELLIPDEV,
       to remove MONTE, add SLICE, and renumber other actions */
/* Modified 2006 October 1 by CM to add SIZEPAR and renumber other parameters, */
/*     to add VARY_NONE, VARY_SIZE, VARY_ALL, and to add LN10  */
/* Modified 2005 October 6 by CM to delete SUNMAG (in favor of sun_appmag parameter) */
/* Modified 2005 September 11 by CM to add PHOTOHARM */
/* Modified 2005 September 8 by CM to add PHOTOFACETS */
/* Modified 2005 August 1 by CM to add OPTICALVIEW, RADARVIEW, and LAMBERTVIEW */
/* Modified 2005 July 20 by CM to add THETADEL, RAD_C_VAR, and RAD_C_DEL */
/*     (and to renumber the penalty function constants) */
/* Modified 2005 June 27 by CM to rename INFINITY to HUGENUMBER */
/* Modified 2005 June 20 by CM to add BIG_ENDIAN_DATA and LITTLE_ENDIAN_DATA */
/* Modified 2005 March 10 by CM to add SMALLVAL */
/* Modified 2005 March 8 by CM to delete MPI_MESLEN */
/* Modified 2005 February 24 by CM to move SHAPEPAR, SPINPAR, and PHOTOPAR here  */
/*     from mpi_vars.h, to change DATAPAR to DELCORPAR, and to add XYOFFPAR      */
/* Modified 2005 February 18 by CM to add CUBICCONV */
/* Modified 2005 January 20 by CM to add BILINEAR, to change BICUBIC from 2 to 3, */
/*     and to move INFINITY here from posclr.c                                    */
/* Modified 2004 December 19 by CM to add AREA */
/* Modified 2004 November 23 by CM to add VIEW */
/* Modified 2004 May 8 by CM to add FLATTENING and RESCALE */
/* Modified 2004 April 25 by CM to add INERTIADEV_UNI and NONPA_UNI */
/* Modified 2004 March 3 by CM to add CONVEXHULL */
/* Modified 2003 May 15 by CM to add KM2US, KM2HZFACT, MAXBINS */
/* Modified 2003 May 14 by CM to add DELCORINIT */
/* Modified 2003 May 10 by CM to add SCALE_* (fit vs. obs pgm) and MAXOVERFLOW */
/* Modified 2003 April 23 by CM to add HORIZONS and DATATAKING (ephemeris formats) */
/* Modified 2003 April 19 by CM to add MOMENTS */
/* Modified 2003 April 18 by CM to add MAXLEN (max string length) */
/* Modified 2003 April 15 by CM to add "code method" parameters */

#define WRITE 1
#define COVAR 2
#define FORMAT 3
#define FACETS 4
#define SAMPLE 5
#define MIRROR 6
#define FIT 7
#define WAVEFRONT 8
#define REFSHAPE 9
#define MOMENTS 10
#define DELCORINIT 11
#define CONVEXHULL 12
#define MAP 13
#define VIEW 14
#define AREA 15
#define PHOTOFACETS 16
#define PHOTOHARM 17
#define SLICE 18
#define ORBIT 19
#define SPLIT 20

#define SIZEPAR 1
#define SHAPEPAR 2
#define PHOTOPAR 3
#define SPINPAR 4
#define DELCORPAR 5
#define DOPSCALEPAR 6
#define XYOFFPAR 7

#define NONE 0
#define BLOCK 1
#define BILINEAR 2
#define BICUBIC 3
#define CUBICCONV 4

#define GLOBAL 1
#define LOCAL 2

#define BRUTE 1
#define PERIODIC 2

#define BIG_ENDIAN_DATA 1
#define LITTLE_ENDIAN_DATA 2

#define OPTICALVIEW 1
#define RADARVIEW 2
#define LAMBERTVIEW 3

#define VARY_NONE 0
#define VARY_SIZE 1
#define VARY_SPIN 2
#define VARY_ALL 3

#define MAPMODE_DELDOP 0
#define MAPMODE_POS 1
#define MAPMODE_FACETS 2

#define OPTALBDEL 1
#define OPTALBVAR 2
#define THETAVAR 3
#define THETADEL 4
#define RADALBDEL 5
#define RADALBVAR 6
#define RAD_C_DEL 7
#define RAD_C_VAR 8
#define NONSMOOTH 9
#define CONCAVITY 10
#define RDEV 11
#define VOLUME 12
#define COMDEV 13
#define INERTIADEV 14
#define INERTIADEV_UNI 15
#define PA3TILT 16
#define NONPA 17
#define NONPA_UNI 18
#define EULEROFFS 19
#define FLATTENING 20
#define MAXRDEV 21
#define MAXELLIPDEV 22
#define NONCOSINE 23
#define BIFURCATION 24
#define IMPULSE 25

#define D2R 0.01745329251994330
#define R2D 57.29577951308232
#define PIE 3.141592653589793
#define PIOVER2 1.570796326794897
#define TWOPI 6.283185307179586
#define LN10 2.302585092994046
#define HUGENUMBER 1.0e20
#define AU 1.4959787061e8

#define SMALLVAL 1.0e-10
#define TINYCALFACT 1.0e-40

/* a half-second epoch tolerance, expressed in days */
#define HALFSECOND 5.8e-6

/* 499.0047835 seconds per AU (1-way) */
#define DAYSPERAU 5.775518328e-3

/* gravitational constant -- 2006 CODATA value,
   (6.67428 +/- 0.00067) x 10^-11 m^3 kg^-1 s^-2
   -- expressed in km^3 kg^-1 day^-2             */

#define GRAVCONST 4.98232e-10

/* Sun's apparent V magnitude, -26.762 +/- 0.017, from Campins et al. 1985
   (AJ, 90, 896-899), cited by Pravec and Harris 2007 (Icarus, 190, 250-259)
   and implicitly assumed in the commonly used diameter-magnitude expression
   D = [ (1329 +/- 10 km) / sqrt(visual albedo) ] * 10^(-H/5)                */

#define SOLAR_VMAG -26.762

#define SHORT 0
#define LONG_ORIG 1
#define LONG_MOD 2

#define SMEAR_CENTER 0
#define SMEAR_FIRST 1

#define MAXLEN 255

#define HORIZONS 0
#define DATATAKING 1

#define SCALE_SEPARATELY 0
#define SCALE_MAXFITOBS 1
#define SCALE_FIT 2
#define SCALE_OBS 3

/* If model power extends beyond the (delay-)Doppler data limits,
   we need a large vector or matrix to hold the "overflow" pixel
   values (in order to compute chi squared later): each dimension
   of this vector/matrix has MAXOVERFLOW bins.                     */

#define MAXOVERFLOW 2000

/* KM2US is 2/c where c is in km per microsecond */
/* KM2HZFACT is 2e6/c where c is in km per day   */

#define KM2US 6.67128190396304
#define KM2HZFACT 7.72139109254982e-5

/* We might distribute power to more than sinc2width Doppler bins per
   POS pixel if nsinc2 > 1 and each pixel spans multiple bins;
   choose MAXBINS to be large enough to cover every realistic case.    */

#define MAXBINS 100
