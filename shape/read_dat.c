/* $Id: read_dat.c,v 2.28 2017/06/02 01:29:30 magri Exp $ */
/* $Log: read_dat.c,v $
/* Revision 2.28  2017/06/02 01:29:30  magri
/* Make the "sun_appmag" parameter a vector rather than a scalar
/*
/* Revision 2.27  2017/03/16 16:01:26  magri
/* Allow (delay-)Doppler datasets to have some delay correction polynomial
/*     coefficients with the '=' state and others with the 'f' or 'c' state
/*
/* Revision 2.26  2016/11/07 01:14:41  magri
/* Add "heliocurve" datasets
/*
/* Revision 2.25  2016/02/24 22:48:22  magri
/* Bug fix reading pixel masks for plane-of-sky frames
/*
/* Revision 2.24  2015/06/11 13:07:04  magri
/* For the "fit" and "write" actions, implement smearing
/*
/* Revision 2.23  2014/03/12 14:25:59  magri
/* In the "read_dat" routine, check that the number of the scattering law
/*     specified for each dataset is within legal bounds
/*
/* Revision 2.22  2014/02/21 18:14:40  magri
/* Implement multiple radar and optical scattering laws
/*
/* Revision 2.21  2013/06/17 18:21:30  magri
/* Check that pixel-weighting masks contain the expected number of entries
/*
/* Revision 2.20  2013/04/25 16:35:31  magri
/* Adjust names of output files and input pixel-weighting mask files so they
/*     are in alphanumeric order if > 100 frames (or lightcurve points) per
/*     dataset
/*
/* Revision 2.19  2012/04/04 15:51:04  magri
/* Add Doppler scaling factors and the "dopscale_min" "dopscale_max" and
/*     "vary_dopscale" parameters
/*
/* Revision 2.18  2011/08/24 03:36:08  magri
/* Implement spin impulses
/*
/* Revision 2.17  2010/04/27 23:53:37  magri
/* In read_deldop, check return value of "system" commands in order to avoid
/*     compilation warnings
/*
/* Revision 2.16  2009/04/04 02:48:29  magri
/* For lightcurve datasets, don't automatically turn the obs file
/*     specification of calculated points into an explicit list of
/*     Julian dates
/*
/* Revision 2.15  2008/08/20 06:00:56  magri
/* For the "delcorinit" action, hand-specified row/column/bin numbers should
/* refer to the unvignetted data if pixel-weighting masks are being used to
/* vignette the data
/*
/* Revision 2.14  2007/10/22 13:29:48  magri
/* Fix small bug: for delay-Doppler frames that have been vignetted via a
/* pixel mask, adjust the DC Doppler column accordingly
/*
/* Revision 2.13  2007/09/09 20:00:29  magri
/* In read_deldop, use tempname function to get unique temporary filenames
/*     for zipped and unzipped delay-Doppler datafiles rather than using
/*     the fixed names /tmp/deldop.dat.gz and /tmp/deldop.dat.
/* Add "maskdir" parameter.
/*
/* Revision 2.12  2007/08/12 01:20:53  magri
/* Eliminate unused variables, initialize potentially uninitialized variables
/*
/* Revision 2.11  2007/08/06 21:58:44  magri
/* Check that the number of observed points in each lightcurve datafile is
/* equal to the number claimed in the obs file.
/*
/* Revision 2.10  2007/01/07 23:45:48  magri
/* For the "write" action for lightcurve datasets, output rotation phases
/* (deg) as a new column in output files fit_MM.dat and calc_MM.dat
/* (MM = dataset number)
/*
/* Revision 2.9  2006/10/02 20:16:11  magri
/* lightcurve data now treated internally as intensities rather than magnitudes
/* two new arguments to realize_delcor
/* implement "vary_delcor0" "vary_radalb" and "vary_optalb" parameters
/*
/* Revision 2.8  2006/06/21 20:43:57  magri
/* Revert to v2.5; then improve external documentation, internal comments,
/* and internal variable names to do away with the nebulous concept of
/* "resolution" and make it clear that obs file entries for delay-Doppler,
/* Doppler, and plane-of-sky datasets involve the dimensions of single
/* pixels/bins, regardless of the true effective resolution of the data.
/* Similarly, the resolution ("res") of POS renderings has been renamed
/* "km_per_pixel."
/*
/* Revision 2.7  2006/06/19 00:10:04  magri
/* Implement new pixel-weighting "mask" flag for delay-Doppler, Doppler, and
/* plane-of-sky frames.  Delay-Doppler, Doppler, and plane-of-sky frames
/* within a given dataset can have different dimensions after being
/* effectively vignetted (via pixel-weighting masks).  Plane-of-sky data
/* frames no longer must be square or must have an odd number of pixels on a
/* side.  Plane-of-sky data frames now are read from FITS images rather than
/* from pgm images.  Range datasets have been eliminated.
/*
/* Revision 2.6  2005/11/22 20:55:34  magri
/* Fixed comment about the "local" value for the "pos_scope" parameter
/*
/* Revision 2.5  2005/11/19 17:30:55  magri
/* Change "nlooks" (number of looks) from integer to floating-point for
/* (delay-)Doppler frames, to handle zero-filled data for which the
/* effective number of looks is less than one
/*
/* Revision 2.4  2005/07/22 06:24:06  magri
/* Fully implement the "local" value for the "pos_scope" parameter
/*
/* Revision 2.3  2005/07/01 19:24:55  magri
/* Fixed bug in read_lghtcrv for evenly spaced calculated lightcurve points:
/* one extra point was being defined a bit later than the specified stop epoch
/*
/* Revision 2.2  2005/06/21 17:38:51  magri
/* CM: Slight changes to reading of delay-Doppler datafiles in read_deldop
/* and four subroutines (e.g., read_deldop_rdf); code copied to read_doppler
/* and to four new subroutines (e.g., read_doppler_rdf) to handle reading of
/* Doppler datafiles.
/*
/* Revision 2.1  2005/06/20 19:42:42  jlm
/* Changed read_fits() function name to read_deldop_fits() at CM's suggestion
/* to avoid confusion with other data types.  Same for RDF, binary, and ASCII.
/* Improved RDF type detection.
/* Note that the changes in revision 2.x eliminated the need to define
/* the additional data type suggested in revision 1.2.
/* */

/*****************************************************************************************
                                                                               read_dat.c

Reads file describing observed data.

Modified 2017 June 1 by CM:
    Change "sun_appmag" parameter to be a vector rather than a scalar, on the assumption
        that a mod file with more than one optical scattering law will represent more
        than one broadband filter, thus requiring more than one solar apparent magnitude

Modified 2017 March 15 by CM:
    Allow a (delay-)Doppler dataset to have some delay correction polynomial coefficients
        with the '=' state and others with the 'f' or 'c' state

Modified 2016 November 6 by CM:
    Add "heliocurve" datasets

Modified 2016 February 24 by CM:
    Bug fix in the "read_poset" routine: pixel mask arrays were being read in with rows
        and column switched, causing the program to crash

Modified 2015 June 10 by CM:
    Implement smearing for the "fit" and "write" actions by reading number of views per
        frame (or lightcurve point), view interval, and smearing mode for all datasets

Modified 2014 March 12 by CM:
    Pass the "mod" argument to read_dat so we can check that the scattering law specified
        for each dataset is within legal bounds

Modified 2014 February 12 by CM:
    Implement multiple radar and optical scattering laws

Modified 2013 June 17 by CM:
    Check that pixel-weighting masks contain the expected number of entries and quit with
        an informative error message if they don't; this way the program doesn't choke if
        you add or remove datasets and forget to rename the mask files accordingly

Modified 2013 April 24 by CM:
    Adjust input filenames for pixel-weighting masks so they are in alphanumeric order if
        > 100 per dataset

Modified 2012 March 24 by CM:
    In read_deldop and read_doppler, read "dopscale" parameter and initialize
        "dopscale_save" for each (delay-)Doppler dataset
    Call realize_dopscale routine to initialize Doppler scaling factors (including those
        with the '=' state)

Modified 2011 August 11 by CM:
    Initialize quantities related to spin impulses

Modified 2010 April 27 by CM:
    In read_deldop, check return value of "system" commands in order to
        avoid compilation warnings

Modified 2009 March 24 by CM:
    In read_lghtcrv, store the values that are (potentially) listed in the
        obs file for calculated lightcurve points: number of points,
        JD start time, JD stop time, and JD interval

Modified 2008 August 19 by CM:
    Save the row and column limits within the unvignetted image of
        vignetted images, and similarly for vignetted spectra

Modified 2007 October 22 by CM:
    Fix bug in read_deldop: DC bin must be adjusted for vignetted images

Modified 2007 August 31 by CM:
    Implement "maskdir" parameter
    In read_deldop, use tempname function to get unique temporary filenames
        for zipped and unzipped delay-Doppler datafiles rather than using
        the fixed names /tmp/deldop.dat.gz and /tmp/deldop.dat

Modified 2007 August 10 by CM:
    Eliminate unused variable

Modified 2007 August 4 by CM:
    In read_lghtcrv, check that the number of observed lightcurve points is
        equal to what the obs file claims it to be

Modified 2007 July 23 by CM:
    Add body, bodyill, comp, and compill matrices to POS frames

Modified 2007 January 6 by CM:
    In read_lghtcrv, allocate memory for rotation phases at calculated
        epochs (rotphase_calc) and at observation epochs (rotphase_obs)

Modified 2006 October 1 by CM:
    Add two new arguments to realize_delcor
    In read_deldop and read_doppler, initialize "delcor0_save" for each
        delay correction polynomial
    Compute three sums of weights for use by the vary_params routine
    In read_lghtcrv, convert input magnitudes and magnitude errors to
        intensities (relative to solar intensity) and intensity errors

Modified 2006 June 21 by CM:
    In read_deldop, changed delres to del_per_pixel and dopres to
        dop_per_pixel, and improved comments
    In read_doppler, changed dopres to dop_per_bin and improved comments
    In read_poset, changed angres to angle_per_pixel and and res to
        km_per_pixel, and improved comments
    For POS renderings, changed res to km_per_pixel

Modified 2006 June 18 by CM:
    Implement "mask" pixel-weighting flag for delay-Doppler, Doppler, and
        plane-of-sky frames
    Allow each delay-Doppler frame within a dataset to have different
        dimensions after vignetting
    Allow each Doppler frame within a dataset to have different
        dimensions after vignetting
    Allow plane-of-sky frames to be rectangular and to have any number
        of pixels (not just an odd number) per side
    Read plane-of-sky frames from FITS files rather than from pgm files
    Eliminate range datasets

Modified 2005 November 18 by MCN:
    Change nlooks (# of looks) from integer to floating-point, to handle
        zero-filled data

Modified 2005 July 21 by CM:
    Fully implement "local" value for "pos_scope" parameter:
        Each delay-Doppler frame, Doppler frame, range frame, plane-of-sky
        data frame, and calculated lightcurve point has separate memory
        allocated for its own model POS frame

Modified 2005 July 1 by CM:
    Fix bug in read_lghtcrv: When assigning evenly spaced calculation
        epochs, the last epoch was later than the specified stop epoch

Modified 2005 June 20 by JLM & CM:
    Create new routines to read delay-Doppler and Doppler datafiles in
        ASCII, rdf, FITS, or binary format.  Byte-swapping is done
        automatically if needed.

Modified 2005 April 25 by CM:
    Compute variance of chi2 estimate

Modified 2005 March 10 by CM:
    Compute degrees of freedom here rather than in routine chi2, so that
        it only needs to be done once for a fit rather than repeatedl;
    Add "dof" argument to read_deldop, read_doppler, read_poset,
        read_lghtcrv, and read_range to enable this change
    Allow weights (and hence degrees of freedom) to be floating-point
        rather than integer

Modified 2005 March 6 by CM:
    For plane-of-sky datasets, eliminate "calfact" from the obs file;
        instead, for each frame, force the calibration factor to float
        and assign it an initial dummy value

Modified 2005 March 1 by CM:
    Rename some "sdev" and "var" values to "oneovervar" (1/variance)
    For plane-of-sky datasets, eliminate "sdev" and add "northangle"
        for each frame
    Fix bug: the number of floating parameters (npar) was too large
        because floating plane-of-sky calibration factors were being
        included
    Add call to new realize_xyoff routine

Modified 2005 January 25 by CM:
    Take care of unused and uninitialized variables

Modified 2005 January 21 by CM:
    Allocate memory for POS images (sky renderings) for POS and range
        datasets
    Permit one-way light-time corrections for POS and range datasets
    For POS and range datasets, pass "frame[i].oe" rather than
        "frame[i].fit.oe" to ephem2mat; for POS datasets, pass
        "frame[i].se" rather than "frame[i].fit.se" to ephem2mat
    Eliminate null processing block for POS and range datasets when the
        "pos_scope" parameter is set to "local"
    Assign solar phase angles and solar azimuth angles for POS datasets
    Convert POS angular resolution from arcseconds to radians

Modified 2005 January 13 by CM:
    Always read the "mpi node" value for each dataset in the obs file, but
        only pay attention to it if the "read_node" parameter is set to 1.
        Before this it was only read if read_node = 1, meaning that lines
        had to be added to or deleted from the obs file depending on the
        value of this parameter.
    If the mpi node for a given dataset is input as a negative number,
        assign that dataset to the next processing node, even if read_node
        is set to 1.  This reproduces the behavior that one would obtain
        by setting read_node = 0, except that one can choose it for only a
        subset of the datasets if desired.
    For parallel processing, data should only be read by the appropriate
        node.  (Until now this was being done only for delay-Doppler and
        lightcurve datasets.)
    For parallel processing, suppress most screen output for nodes
        other than root
    For parallel processing, show which node is assigned to which dataset
    Added "s" (set number) argument to read_poset and read_range routines

Modified 2004 August 6 by CM:
    Shift all Euler angle offsets to the range (-180, +180] degrees

Modified 2004 July 26 by CM:
    For Doppler and delay-Doppler frames, don't allow constant
        calibration factors which are <= 0.0
    Added "s" (set number) argument to read_doppler routine

Modified 2004 April 9 by CM:
    Compute solar azimuth angles (N -> E in POS) for lightcurve datasets

Modified 2004 March 22 by CM:
    For lightcurve points, save the orbital (plane-of-sky) contribution
        to the apparent spin vector

Modified 2004 March 13 by CM:
    For lightcurve datasets, add option to calculate model magnitudes
        at the same epochs for which observations are available;
        denote this in the obs file by specifying "-1" as the number of
        points at which to calculate, and don't actually list any
        calculation epochs below that line.  Repeated observation epochs
        (presumably due to observations at multiple observatories on the
        same night) are weeded out.
    For lightcurve datasets with calculation epochs explicitly listed
        or else specified via start/stop/interval, validate that the
        epochs are in increasing order with no repeated values.

Modified 2004 February 13 by CM:
    Removed "sdev" argument to routine gamma_trans
    Replaced LTC variable by perform_ltc parameter for computing
        1-way light-time correction
    Compute solar phase angles for lightcurve datasets

Modified 2003 December 12 by CM:
    Stop program if a Doppler dataset uses the 'f' state for its
        zeroth-order delay polynomial coefficient

Modified 2003 April 24 by CM:
    Move "delcom" from delay-Doppler datasets to individual frames
    Add "weight" to delay-Doppler, Doppler, POS, and range frames
        and to lightcurve datasets
    Integrate the extra delay-Doppler parameters added April 15 with
        two existing lines in the datafile

Modified 2003 April 23 by CM:
    Implement '=' state for delay correction polynomial coefficients:
        now call "realize_delcor" rather than directly calling
        "deldopoffs" or "dopoffs"

Modified 2003 April 15 by CM:
    Read an extra line of delay-Doppler information
        (e.g., # samples per baud)
*****************************************************************************************/

#include "head.h"

int read_deldop( FILE *fp, struct par_t *par, struct deldop_t *deldop,
                           int nradlaws, int s, double *chi2_variance);
int read_doppler( FILE *fp, struct par_t *par, struct doppler_t *doppler,
                            int nradlaws, int s, double *chi2_variance);
void set_up_pos( struct par_t *par, struct dat_t *dat);
int read_poset( FILE *fp, struct par_t *par, struct poset_t *poset,
                          int noptlaws, int s, double *chi2_variance);
int read_lghtcrv( FILE *fp, struct par_t *par, struct lghtcrv_t *lghtcrv,
                            int noptlaws, int s, unsigned char type,
                            double *chi2_variance);

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

//void initialize_delcor( struct dat_t *dat);

int read_dat( struct par_t *par, struct mod_t *mod, struct dat_t *dat)
{
  FILE *fp;
  int s, npar=0, i, cntr=0;
  double chi2_variance;
  char str[80];

  /*  Initialize degrees of freedom, variance of chi2 estimate, and sums of weights */

  (*dat).dof = (*dat).dof_doppler = (*dat).dof_deldop = (*dat).dof_poset
             = (*dat).dof_lghtcrv = 0.0;
  (*dat).chi2_variance = 0.0;
  (*dat).sum_deldop_zmax_weights = 0.0;
  (*dat).sum_rad_xsec_weights = 0.0;
  (*dat).sum_opt_brightness_weights = 0.0;
  (*dat).sum_cos_subradarlat_weights = 0.0;

  /*  Open obs file and read how many datasets there are  */

  FOPEN( fp, (*dat).name, "r");

    if ((*par).action == FIT) {
      printf("#\n# fitting with  1 processing node\n");
    }
    printf("#\n# reading data through file: %s ...\n", (*dat).name);
    fflush(stdout);

  (*dat).nsets = getint( fp);
  (*dat).set = (struct set_t *) calloc( (*dat).nsets, sizeof( struct set_t));

  /*  Read each dataset in turn  */

  for (s=0; s<(*dat).nsets; s++) {

    /*  Assign a processor node to this dataset  */

    (*dat).set[s].inputnode = getint( fp);

    /*  Read the angle and spin offsets for this dataset  */

    for (i=0; i<=2; i++) {
      npar += readparam( fp, &(*dat).set[s].angleoff[i]);
      (*dat).set[s].angleoff[i].val += 360.0*floor((180.0 - (*dat).set[s].angleoff[i].val)/360.0);
      (*dat).set[s].angleoff[i].val *= D2R;
    }
    for (i=0; i<=2; i++) {
      npar += readparam( fp, &(*dat).set[s].omegaoff[i]);
      (*dat).set[s].omegaoff[i].val *= D2R;
    }

    /*  Figure out what type of dataset this is (delay-Doppler, Doppler, etc.)
        and call the appropriate routine to read in the rest of the information  */

    gettstr( fp, str);
    if (!strcmp( str, "doppler")) {
      (*dat).set[s].type = DOPPLER;
      npar += read_doppler( fp, par, &(*dat).set[s].desc.doppler, (*mod).photo.nradlaws,
                            s, &chi2_variance);
      (*dat).dof_doppler += (*dat).set[s].desc.doppler.dof;
      (*dat).sum_rad_xsec_weights += (*dat).set[s].desc.doppler.sum_rad_xsec_weights;
      (*dat).sum_cos_subradarlat_weights += (*dat).set[s].desc.doppler.sum_cos_subradarlat_weights;
    }
    else if (!strcmp( str, "delay-doppler")) {
      (*dat).set[s].type = DELAY;
      npar += read_deldop( fp, par, &(*dat).set[s].desc.deldop, (*mod).photo.nradlaws,
                           s, &chi2_variance);
      (*dat).dof_deldop += (*dat).set[s].desc.deldop.dof;
      (*dat).sum_deldop_zmax_weights +=
                  (*dat).set[s].desc.deldop.sum_deldop_zmax_weights;
      (*dat).sum_rad_xsec_weights += (*dat).set[s].desc.deldop.sum_rad_xsec_weights;
      (*dat).sum_cos_subradarlat_weights += (*dat).set[s].desc.deldop.sum_cos_subradarlat_weights;
    }
    else if (!strcmp( str, "plane-of-sky")) {
      (*dat).set[s].type = POS;
      npar += read_poset( fp, par, &(*dat).set[s].desc.poset, (*mod).photo.noptlaws,
                          s, &chi2_variance);
      (*dat).dof_poset += (*dat).set[s].desc.poset.dof;
    }
    else if (!strcmp( str, "lightcurve")) {
      (*dat).set[s].type = LGHTCRV;
      npar += read_lghtcrv( fp, par, &(*dat).set[s].desc.lghtcrv, (*mod).photo.noptlaws,
                            s, (*dat).set[s].type, &chi2_variance);
      (*dat).dof_lghtcrv += (*dat).set[s].desc.lghtcrv.dof;
      (*dat).sum_opt_brightness_weights +=
                  (*dat).set[s].desc.lghtcrv.sum_opt_brightness_weights;
    }
    else if (!strcmp( str, "heliocurve")) {
      (*dat).set[s].type = LGHTCRV;
      npar += read_lghtcrv( fp, par, &(*dat).set[s].desc.lghtcrv, (*mod).photo.noptlaws,
                            s, (*dat).set[s].type, &chi2_variance);
      (*dat).dof_lghtcrv += (*dat).set[s].desc.lghtcrv.dof;
      (*dat).sum_opt_brightness_weights +=
                  (*dat).set[s].desc.lghtcrv.sum_opt_brightness_weights;
    }
    else {
      printf("ERROR: Can't handle type '%s' for dataset %d\n", str, s);
      bailout("read_dat.c\n");
    }
    (*dat).chi2_variance += chi2_variance;
  }
  (*dat).dof = (*dat).dof_doppler + (*dat).dof_deldop + (*dat).dof_poset
                                                      + (*dat).dof_lghtcrv;
  fclose( fp);

  /*  Initialize the delay correction polynomials, the horizontal and vertical offsets
      for plane-of-sky datasets, and the plane-of-sky view(s)                           */

  initialize_delcor( dat);
  realize_delcor( dat, 0.0, 0);
  realize_dopscale( par, dat, 1.0, 0);
  realize_xyoff( dat);
  set_up_pos( par, dat);

  printf("# finished reading obs file\n");
  fflush(stdout);

  return npar;
}

int read_deldop( FILE *fp, struct par_t *par, struct deldop_t *deldop,
                           int nradlaws, int s, double *chi2_variance)
{
  int i, j, npar=0, k, swap_bytes, is_binary, is_fits, is_rdf,
      bufferlength=128, idel_use[2], idop_use[2], jskip, kskip,
      datafile_is_gzipped, n, nmaskvals;
  char fullname[160], cmd[255], codestring[10], smearingstring[7], buffer[200],
       pixweightfile[MAXLEN], datafile_temp[MAXLEN], datafile_gzipped_temp[MAXLEN];
  FILE *fin, *wp=0;
  double lookfact, se[3][3], dist, solar_phase, solar_azimuth, dof,
         **pixweight=NULL;

  /* Note:  Matt Engels put this following buffer initialization loop to
   * eliminate valgrind errors about unitialized values being used for
   * conditional jumps.  The strstr() function expects a null-terminated
   * string.   */
  for (int kx=0; kx<200; kx++)
	  buffer[kx] = '0';
  buffer[199] = '\0';

  /*  Initialize degrees of freedom, variance of chi2 estimate, and weight sums  */

  (*deldop).dof = 0.0;
  *chi2_variance = 0.0;
  (*deldop).sum_deldop_zmax_weights = 0.0;
  (*deldop).sum_rad_xsec_weights = 0.0;
  (*deldop).sum_cos_subradarlat_weights = 0.0;

  /*  Read which radar scattering law to use for this dataset  */

  (*deldop).iradlaw = getint( fp);
  if ((*deldop).iradlaw < 0 || (*deldop).iradlaw >= nradlaws) {
    printf("ERROR in set %d: must have 0 <= radar scattering law <= %d\n",
           s, nradlaws-1);
    bailout("read_deldop in read_dat.c\n");
  }

  /*  Read the asteroid ephemeris  */

  (*deldop).astephem.n = getint( fp); /* # of points in ephemeris */
  (*deldop).astephem.pnt = (struct ephpnt_t *) calloc( (*deldop).astephem.n,
                                                       sizeof( struct ephpnt_t));
  for (i=0; i<(*deldop).astephem.n; i++) {
    rdcal2jd( fp, &(*deldop).astephem.pnt[i].t);
    (*deldop).astephem.pnt[i].ra = getdouble( fp)*D2R;
    (*deldop).astephem.pnt[i].dec = getdouble( fp)*D2R;
    (*deldop).astephem.pnt[i].dist = getdouble( fp);
  }

  /*  Read the transmitter frequency (MHz)  */

  (*deldop).Ftx = getdouble( fp);

  /*  Read delay information for the unvignetted images.

      Note that del_per_pixel is NOT necessarily the baud length
      for multiple-spb data: it is equal to the delay height of
      each pixel (or image row), which is baud / (spb/stride).    */

  (*deldop).ndel = getint( fp); /* # delay bins */
  (*deldop).del_per_pixel = getdouble( fp); /* pixel height (usec) */
  (*deldop).spb = getint( fp); /* # samples per baud in original data */
  (*deldop).stride = getint( fp); /* image rows are stride spb's apart */
  if ((*deldop).spb % (*deldop).stride != 0)
    bailout("read_deldop in read_dat.c: stride must divide evenly into spb\n");
  gettstr( fp, codestring);   /* type of code+reduction used */
  if (!strcmp(codestring, "short"))
    (*deldop).codemethod = SHORT;
  else if (!strcmp(codestring, "long1") || !strcmp(codestring, "long_orig"))
    (*deldop).codemethod = LONG_ORIG;
  else if (!strcmp(codestring, "long2") || !strcmp(codestring, "long_mod"))
    (*deldop).codemethod = LONG_MOD;
  else
    bailout("read_deldop in read_dat.c: can't do that code method yet\n");

  /*  Read Doppler information for the unvignetted images  */

  (*deldop).ndop = getint( fp); /* # doppler bins */
  (*deldop).dop_per_pixel = getdouble( fp); /* pixel width (Hz) */
  (*deldop).dopcom = getdouble( fp); /* ephemeris COM doppler bin (1-based) */
  (*deldop).dopDC = getdouble( fp); /* Doppler DC bin (1-based) */
  (*deldop).dopfftlen = getint( fp); /* Doppler fft length */

  /*  Compute the reference epoch (JD) for the delay correction polynomial  */

  rdcal2jd( fp, &(*deldop).delcor.t0);

  /*  Read the delay correction polynomial itself:
      coefficients have units of usec, usec/day, usec/day^2, etc.  */

  (*deldop).delcor.n = getint( fp);
  (*deldop).delcor.a = (struct param_t *) calloc( (*deldop).delcor.n+1,
                                                  sizeof(struct param_t));
  (*deldop).delcor.equals_set = ivector( 0, (*deldop).delcor.n);
  for (i=0; i<=(*deldop).delcor.n; i++) {
    npar += readparam( fp, &(*deldop).delcor.a[i]);
    (*deldop).delcor.equals_set[i] = -1;
  }
  (*deldop).delcor.delcor0_save = (*deldop).delcor.a[0].val;

  /*  Read the Doppler scaling factor  */

  npar += readparam( fp, &(*deldop).dopscale);
  (*deldop).dopscale_save = (*deldop).dopscale.val;

  /*  Read smearing information  */

  (*deldop).nviews = getint( fp); /* # views per frame */
  (*deldop).view_interval = getdouble( fp); /* view interval (s) */
  (*deldop).view_interval /= 86400; /* convert to days */
  gettstr( fp, smearingstring);   /* smearing mode */
  if (!strcmp(smearingstring, "center")) {
      (*deldop).smearing_mode = SMEAR_CENTER;
      (*deldop).v0 = (*deldop).nviews / 2;
  } else if (!strcmp(smearingstring, "first")) {
      (*deldop).smearing_mode = SMEAR_FIRST;
      (*deldop).v0 = 0;
  } else {
      bailout("read_deldop in read_dat.c: can't do that smearing mode yet\n");
  }

  /*  Get the data directory and the number of frames in the dataset  */

  gettstr( fp, (*deldop).dir);
  (*deldop).nframes = getint( fp);
  (*deldop).frame = (struct deldopfrm_t *)
                    calloc( (*deldop).nframes, sizeof(struct deldopfrm_t));
  for (i=0; i<(*deldop).nframes; i++)
    (*deldop).frame[i].view = (struct deldopview_t *)
                              calloc( (*deldop).nviews, sizeof(struct deldopview_t));

  /*  Loop through the frames  */

  for (i=0; i<(*deldop).nframes; i++) {
    gettstr( fp, (*deldop).frame[i].name); /* name of data file */
    sprintf( fullname, "%s/%s", (*deldop).dir, (*deldop).frame[i].name);

    /*  Read this frame's mid-receive epoch and convert to a Julian date.  */

    rdcal2jd( fp, &(*deldop).frame[i].t0);

    /*  Read sdev, calibration factor, number of looks, and
        ephemeris COM delay bin (1-based) for this frame     */

    (*deldop).frame[i].sdev = getdouble( fp);
    readparam( fp, &(*deldop).frame[i].cal);
    if ((*deldop).frame[i].cal.val <= 0.0 && (*deldop).frame[i].cal.state == 'c') {
      printf("ERROR in set %d: frame %d has constant calfact <= 0.0\n", s, i);
      bailout("read_deldop in read_dat.c\n");
    }
    if (((*deldop).frame[i].nlooks = getdouble( fp)) > 0)
      lookfact = 1/sqrt((*deldop).frame[i].nlooks);
    else
      lookfact = 0.0;
    (*deldop).frame[i].delcom = getdouble( fp);

    /*  Read this frame's relative weight and add to three weight sums  */

    (*deldop).frame[i].weight = getdouble( fp);
    if ((*deldop).delcor.a[0].state != 'c')
      (*deldop).sum_deldop_zmax_weights += (*deldop).frame[i].weight;
    if ((*deldop).frame[i].cal.state == 'c')
      (*deldop).sum_rad_xsec_weights += (*deldop).frame[i].weight;
    if ((*deldop).dopscale.state != 'c')
      (*deldop).sum_cos_subradarlat_weights += (*deldop).frame[i].weight;

    /*  Read this frame's pixel-weighting "mask" flag  */

    (*deldop).frame[i].pixels_weighted = getint( fp);

    /*  If a pixel-weighting mask is being used, check that it has the
        expected number of values and read them in, and then get the
        frame's vignetted dimensions and number of degrees of freedom   */

    dof = 0.0;
    if ((*deldop).frame[i].pixels_weighted) {

        /*  Open the mask file, count the entries, and make sure that it's the
            expected number of entries (in case someone has changed the numbering
            of datasets without changing mask filenames accordingly) 

            Note that the countdata routine resets the file position indicator to
            its initial value after it finishes reading data entries               */

        if (strcmp( (*par).maskdir, "")) {
            if ((*deldop).nframes > 100)
              sprintf( pixweightfile, "%s/mask_%02d_%03d.txt", (*par).maskdir, s, i);
            else
              sprintf( pixweightfile, "%s/mask_%02d_%02d.txt", (*par).maskdir, s, i);
        } else {
            if ((*deldop).nframes > 100)
              sprintf( pixweightfile, "%s/mask_%02d_%03d.txt", (*deldop).dir, s, i);
            else
              sprintf( pixweightfile, "%s/mask_%02d_%02d.txt", (*deldop).dir, s, i);
        }
        FOPEN( wp, pixweightfile, "r");
        nmaskvals = countdata( wp);
        if (nmaskvals != (*deldop).ndel * (*deldop).ndop) {
          fprintf(stderr,"ERROR: expected %d x %d = %d mask weights for set %2d frame %2d\n",
                  (*deldop).ndel, (*deldop).ndop, (*deldop).ndel * (*deldop).ndop, s, i);
          fprintf(stderr,"       -- instead there are %d weights in %s\n",
                  nmaskvals, pixweightfile);
          bailout("read_deldop in read_dat.c\n");
        }

        /*  Allocate memory for the mask values and read them in  */

        pixweight = matrix( 1, (*deldop).ndel, 1, (*deldop).ndop);
        idel_use[0] = (*deldop).ndel + 1;
        idel_use[1] = 0;
        idop_use[0] = (*deldop).ndop + 1;
        idop_use[1] = 0;
        for (j=1; j<=(*deldop).ndel; j++)
          for (k=1; k<=(*deldop).ndop; k++) {
            pixweight[j][k] = getdouble( wp);
            if (pixweight[j][k] > 0.0) {
              dof += (*deldop).frame[i].weight;
              *chi2_variance += 2 * (*deldop).frame[i].weight * (*deldop).frame[i].weight;
              idel_use[0] = MIN( idel_use[0], j);
              idel_use[1] = MAX( idel_use[1], j);
              idop_use[0] = MIN( idop_use[0], k);
              idop_use[1] = MAX( idop_use[1], k);
            }
          }
        fclose( wp);
    } else {
        dof = (*deldop).frame[i].weight * (*deldop).ndel * (*deldop).ndop;
        *chi2_variance += 2 * (*deldop).frame[i].weight * (*deldop).frame[i].weight
                            * (*deldop).ndel * (*deldop).ndop;
        idel_use[0] = 1;
        idel_use[1] = (*deldop).ndel;
        idop_use[0] = 1;
        idop_use[1] = (*deldop).ndop;
    }
    (*deldop).frame[i].dof = dof;
    (*deldop).dof += dof;
    (*deldop).frame[i].ndel = idel_use[1] - idel_use[0] + 1;
    (*deldop).frame[i].ndop = idop_use[1] - idop_use[0] + 1;
    (*deldop).frame[i].idelvig[0] = idel_use[0];
    (*deldop).frame[i].idelvig[1] = idel_use[1];
    (*deldop).frame[i].idopvig[0] = idop_use[0];
    (*deldop).frame[i].idopvig[1] = idop_use[1];
    (*deldop).frame[i].delcom_vig = (*deldop).frame[i].delcom - idel_use[0] + 1;
    (*deldop).frame[i].dopcom_vig = (*deldop).dopcom - idop_use[0] + 1;
    (*deldop).frame[i].dopDC_vig = (*deldop).dopDC - idop_use[0] + 1;

    /*  If this node handles this dataset,
        allocate memory for observed data and fits  */


    (*deldop).frame[i].obs = matrix( 1, (*deldop).frame[i].ndel,
    		1, (*deldop).frame[i].ndop);
    (*deldop).frame[i].fit = matrix( 1, (*deldop).frame[i].ndel,
    		1, (*deldop).frame[i].ndop);
    (*deldop).frame[i].oneovervar = matrix( 1, (*deldop).frame[i].ndel,
    		1, (*deldop).frame[i].ndop);

    /*  If file is gzipped, copy to a temporary file, unzip, then read  */

    if (!strcmp( ".gz", &fullname[strlen(fullname)-3])) {
    	datafile_is_gzipped = 1;
    	tempname(datafile_temp, MAXLEN, "datafile.temp", "dat");
    	strcpy(datafile_gzipped_temp, datafile_temp);
    	strcat(datafile_gzipped_temp, ".gz");
    	sprintf( cmd, "cp %s %s", fullname, datafile_gzipped_temp);
    	if (!system( cmd))
    		bailout("read_deldop in read_dat.c: can't make copy of gzipped data file\n");
    	sprintf( cmd, "gunzip -f %s", datafile_gzipped_temp);
    	if (!system( cmd))
    		bailout("read_deldop in read_dat.c: can't unzip gzipped data file\n");
    	FOPEN( fin, datafile_temp, "r");
    } else {
    	datafile_is_gzipped = 0;
    	FOPEN( fin, fullname, "r");
    }

    /*  Read first line of file so that we can identify file type  */

    if (!fread(buffer, bufferlength, 1, fin))
    	bailout("read_deldop in read_dat.c: can't read first line of data file\n");
    rewind(fin);

    /*  Set binary flag if first line is not ASCII  */

    is_binary = 0;
    for (j=0; j<bufferlength; j++)
    	if (!isascii(buffer[j]))
    		is_binary = 1;

    /*  Read data according to file type  */

    swap_bytes = ( is_little_endian() && (*par).endian == BIG_ENDIAN_DATA) ||
    		(!is_little_endian() && (*par).endian == LITTLE_ENDIAN_DATA);
    is_fits = !is_binary && (strstr(buffer, "SIMPLE") != NULL);
    lowcase(buffer);
    is_rdf = !is_binary && !is_fits && (strstr(buffer, "type") != NULL);

    if (is_fits)
    	read_deldop_fits(fullname, deldop, i, idel_use, idop_use);
    else if (is_rdf)
    	read_deldop_rdf(fin, deldop, i, idel_use, idop_use, swap_bytes);
    else if (is_binary)
    	read_deldop_binary(fin, deldop, i, idel_use, idop_use, swap_bytes);
    else
    	read_deldop_ascii(fin, deldop, i, idel_use, idop_use);

    /*  Get variance for each pixel;
          apply gamma, speckle, and pixel weighting if desired  */

    jskip = idel_use[0] - 1;
    kskip = idop_use[0] - 1;

    for (j=1; j<=(*deldop).frame[i].ndel; j++)
    	for (k=1; k<=(*deldop).frame[i].ndop; k++) {
    		if ((*par).dd_gamma != 1.0)
    			gamma_trans( &(*deldop).frame[i].obs[j][k], (*par).dd_gamma);
    		if ((*par).speckle)
    			(*deldop).frame[i].oneovervar[j][k] =
    					1/( pow((*deldop).frame[i].sdev,2.0) +
    							pow(lookfact*(*deldop).frame[i].obs[j][k],2.0) );
    		else
    			(*deldop).frame[i].oneovervar[j][k] = 1/pow((*deldop).frame[i].sdev,2.0);
    		if ((*deldop).frame[i].pixels_weighted)
    			(*deldop).frame[i].oneovervar[j][k] *= pixweight[j+jskip][k+kskip];
    	}

    fclose( fin);
    if (datafile_is_gzipped) {
    	sprintf( cmd, "\\rm -f %s %s", datafile_temp, datafile_gzipped_temp);
    	if (!system( cmd))
    		bailout("read_deldop in read_dat.c: can't delete gzipped and unzipped data files\n");
    }

    /*  Free memory for the pixel-weighting mask  */

    if ((*deldop).frame[i].pixels_weighted)
    	free_matrix( pixweight, 1, (*deldop).ndel, 1, (*deldop).ndop);

    /*  Loop through all views contributing to this (smeared) frame  */

    for (k=0; k<(*deldop).nviews; k++) {

    	/*  Compute the epoch of this view, uncorrected for light-travel time  */

    	(*deldop).frame[i].view[k].t = (*deldop).frame[i].t0
    			+ (k - (*deldop).v0)*(*deldop).view_interval;

    	/*  Use this dataset's ephemeris (and linear interpolation) to get the target's
          distance (AU) at this view's epoch.  Also compute frame[i].view[k].oe, the
          transformation matrix from ecliptic to observer coordinates at that epoch,
          and frame[i].view[k].orbspin, the plane-of-sky-motion contribution to the
          apparent spin vector at that epoch.  (orbspin is in ecliptic coordinates.)

          Note that the final "0" in the call to ephem2mat signifies monostatic
          observations; this means that arguments 2 and 5, which would pertain to
          the Sun's ephemeris, are unused here, and arguments 7 and 8 (solar phase
          angle and azimuth angle) are returned as 0.                                  */

      dist = ephem2mat( (*deldop).astephem, (*deldop).astephem,
                        (*deldop).frame[i].view[k].t, (*deldop).frame[i].view[k].oe, se,
                        (*deldop).frame[i].view[k].orbspin, &solar_phase, &solar_azimuth, 0);

      /*  If the perform_ltc parameter is turned on, use the distance just
          obtained to subtract the one-way light-travel time from this view's
          epoch; then go back to the ephemeris and recompute frame[i].view[k].oe
          and frame[i].view[k].orbspin for the corrected epoch.                   */

      if ((*par).perform_ltc) {
        (*deldop).frame[i].view[k].t -= DAYSPERAU*dist;
        ephem2mat( (*deldop).astephem, (*deldop).astephem,
                   (*deldop).frame[i].view[k].t, (*deldop).frame[i].view[k].oe, se,
                   (*deldop).frame[i].view[k].orbspin, &solar_phase, &solar_azimuth, 0);
      }

    }

    /*  Initialize quantities related to spin impulses  */

    (*deldop).frame[i].n_integrate = -999;
    for (n=0; n<MAXIMP+2; n++) {
      (*deldop).frame[i].t_integrate[n] = -HUGENUMBER;
      for (j=0; j<=2; j++)
        (*deldop).frame[i].impulse[n][j] = 0.0;
    }

    printf("#     %s\n", fullname);
    fflush(stdout);

  }

  return npar;
}

int read_doppler( FILE *fp, struct par_t *par, struct doppler_t *doppler,
                            int nradlaws, int s, double *chi2_variance)
{
  int i, j, npar=0, k, swap_bytes, is_binary, is_fits, is_rdf,
      bufferlength=128, idop_use[2], kskip, n, nmaskvals;
  char fullname[160], smearingstring[7], buffer[200], binweightfile[MAXLEN];
  FILE *fin, *wp=0;
  double lookfact, se[3][3], dist, solar_phase, solar_azimuth, dof,
         *binweight=NULL;

  /* Note:  Matt Engels put this following buffer initialization loop to
   * eliminate valgrind errors about unitialized values being used for
   * conditional jumps   */
  for (int kx=0; kx<200; kx++)
	  buffer[kx] = '0';
  buffer[199] = '\0';

  /*  Initialize degrees of freedom, variance of chi2 estimate, and weight sum  */

  (*doppler).dof = 0.0;
  *chi2_variance = 0.0;
  (*doppler).sum_rad_xsec_weights = 0.0;
  (*doppler).sum_cos_subradarlat_weights = 0.0;

  /*  Read which radar scattering law to use for this dataset  */

  (*doppler).iradlaw = getint( fp);
  if ((*doppler).iradlaw < 0 || (*doppler).iradlaw >= nradlaws) {
    printf("ERROR in set %d: must have 0 <= radar scattering law <= %d\n",
           s, nradlaws-1);
    bailout("read_doppler in read_dat.c\n");
  }

  /*  Read the asteroid ephemeris  */

  (*doppler).astephem.n = getint( fp); /* # of points in ephemeris */
  (*doppler).astephem.pnt = (struct ephpnt_t *) calloc( (*doppler).astephem.n,
                                                        sizeof( struct ephpnt_t));
  for (i=0; i<(*doppler).astephem.n; i++) {
    rdcal2jd( fp, &(*doppler).astephem.pnt[i].t);
    (*doppler).astephem.pnt[i].ra = getdouble( fp)*D2R;
    (*doppler).astephem.pnt[i].dec = getdouble( fp)*D2R;
    (*doppler).astephem.pnt[i].dist = getdouble( fp);
  }

  /*  Read the transmitter frequency (MHz)  */

  (*doppler).Ftx = getdouble( fp);

  /*  Read Doppler information for the unvignetted spectra  */

  (*doppler).ndop = getint( fp); /* # doppler bins */
  (*doppler).dop_per_bin = getdouble( fp); /* bin width (Hz) */
  (*doppler).dopcom = getdouble( fp); /* ephemeris COM doppler bin */

  /*  Compute the reference epoch (JD) for the delay correction polynomial  */

  rdcal2jd( fp, &(*doppler).delcor.t0);

  /*  Read the delay correction polynomial itself:
      coefficients have units of usec, usec/day, usec/day^2, etc.  */

  (*doppler).delcor.n = getint( fp);
  (*doppler).delcor.a = (struct param_t *) calloc( (*doppler).delcor.n+1,
                                                   sizeof(struct param_t));
  (*doppler).delcor.equals_set = ivector( 0, (*doppler).delcor.n);
  for (i=0; i<=(*doppler).delcor.n; i++) {
    npar += readparam( fp, &(*doppler).delcor.a[i]);
    (*doppler).delcor.equals_set[i] = -1;
  }
  if ((*doppler).delcor.a[0].state == 'f')
    bailout("can't use \"f\" state for a Doppler dataset's 0th-order delay polynomial coefficient\n");
  (*doppler).delcor.delcor0_save = (*doppler).delcor.a[0].val;

  /*  Read the Doppler scaling factor  */

  npar += readparam( fp, &(*doppler).dopscale);
  (*doppler).dopscale_save = (*doppler).dopscale.val;

  /*  Read smearing information  */

  (*doppler).nviews = getint( fp); /* # views per frame */
  (*doppler).view_interval = getdouble( fp); /* view interval (s) */
  (*doppler).view_interval /= 86400; /* convert to days */
  gettstr( fp, smearingstring);   /* smearing mode */
  if (!strcmp(smearingstring, "center")) {
      (*doppler).smearing_mode = SMEAR_CENTER;
      (*doppler).v0 = (*doppler).nviews / 2;
  } else if (!strcmp(smearingstring, "first")) {
      (*doppler).smearing_mode = SMEAR_FIRST;
      (*doppler).v0 = 0;
  } else {
      bailout("read_doppler in read_dat.c: can't do that smearing mode yet\n");
  }

  /*  Get the data directory and the number of frames in the dataset  */

  gettstr( fp, (*doppler).dir);
  (*doppler).nframes = getint( fp);
  (*doppler).frame = (struct dopfrm_t *)
                     calloc( (*doppler).nframes, sizeof(struct dopfrm_t));
  for (i=0; i<(*doppler).nframes; i++)
    (*doppler).frame[i].view = (struct dopview_t *)
                               calloc( (*doppler).nviews, sizeof(struct dopview_t));

  /*  Loop through the frames  */

  for (i=0; i<(*doppler).nframes; i++) {
    gettstr( fp, (*doppler).frame[i].name); /* name of data file */
    sprintf( fullname, "%s/%s", (*doppler).dir, (*doppler).frame[i].name);

    /*  Read this frame's mid-receive epoch and convert to a Julian date.  */

    rdcal2jd( fp, &(*doppler).frame[i].t0);

    /*  Read sdev, calibration factor, and number of looks for this frame  */

    (*doppler).frame[i].sdev = getdouble( fp);
    readparam( fp, &(*doppler).frame[i].cal);
    if ((*doppler).frame[i].cal.val <= 0.0 && (*doppler).frame[i].cal.state == 'c') {
      printf("ERROR in set %d: frame %d has constant calfact <= 0.0\n", s, i);
      bailout("read_doppler in read_dat.c\n");
    }
    if (((*doppler).frame[i].nlooks = getdouble( fp)) > 0)
      lookfact = 1/sqrt((*doppler).frame[i].nlooks);
    else
      lookfact = 0.0;

    /*  Read this frame's relative weight and add to two weight sums  */

    (*doppler).frame[i].weight = getdouble( fp);
    if ((*doppler).frame[i].cal.state == 'c')
      (*doppler).sum_rad_xsec_weights += (*doppler).frame[i].weight;
    if ((*doppler).dopscale.state != 'c')
      (*doppler).sum_cos_subradarlat_weights += (*doppler).frame[i].weight;

    /*  Read this frame's pixel-weighting "mask" flag  */

    (*doppler).frame[i].pixels_weighted = getint( fp);

    /*  If a bin-weighting mask is being used, check that it has the
        expected number of values and read them in, and then get the
        frame's vignetted dimensions and number of degrees of freedom   */

    dof = 0.0;
    if ((*doppler).frame[i].pixels_weighted) {

        /*  Open the mask file, count the entries, and make sure that it's the
            expected number of entries (in case someone has changed the numbering
            of datasets without changing mask filenames accordingly) 

            Note that the countdata routine resets the file position indicator to
            its initial value after it finishes reading data entries               */

        if (strcmp( (*par).maskdir, "")) {
            if ((*doppler).nframes > 100)
              sprintf( binweightfile, "%s/mask_%02d_%03d.txt", (*par).maskdir, s, i);
            else
              sprintf( binweightfile, "%s/mask_%02d_%02d.txt", (*par).maskdir, s, i);
        } else {
            if ((*doppler).nframes > 100)
              sprintf( binweightfile, "%s/mask_%02d_%03d.txt", (*doppler).dir, s, i);
            else
              sprintf( binweightfile, "%s/mask_%02d_%02d.txt", (*doppler).dir, s, i);
        }
        FOPEN( wp, binweightfile, "r");
        nmaskvals = countdata( wp);
        if (nmaskvals != (*doppler).ndop) {
          fprintf(stderr,"ERROR: expected 1 x %d = %d mask weights for set %2d frame %2d\n",
                  (*doppler).ndop, (*doppler).ndop, s, i);
          fprintf(stderr,"       -- instead there are %d weights in %s\n",
                  nmaskvals, binweightfile);
          bailout("read_doppler in read_dat.c\n");
        }

        /*  Allocate memory for the mask values and read them in  */

        binweight = vector( 1, (*doppler).ndop);
        idop_use[0] = (*doppler).ndop + 1;
        idop_use[1] = 0;
        for (k=1; k<=(*doppler).ndop; k++) {
          binweight[k] = getdouble( wp);
          if (binweight[k] > 0.0) {
            dof += (*doppler).frame[i].weight;
            *chi2_variance += 2 * (*doppler).frame[i].weight * (*doppler).frame[i].weight;
            idop_use[0] = MIN( idop_use[0], k);
            idop_use[1] = MAX( idop_use[1], k);
          }
        }
        fclose( wp);
    } else {
        dof = (*doppler).frame[i].weight * (*doppler).ndop;
        *chi2_variance += 2 * (*doppler).frame[i].weight * (*doppler).frame[i].weight
                            * (*doppler).ndop;
        idop_use[0] = 1;
        idop_use[1] = (*doppler).ndop;
    }
    (*doppler).frame[i].dof = dof;
    (*doppler).dof += dof;
    (*doppler).frame[i].ndop = idop_use[1] - idop_use[0] + 1;
    (*doppler).frame[i].idopvig[0] = idop_use[0];
    (*doppler).frame[i].idopvig[1] = idop_use[1];
    (*doppler).frame[i].dopcom_vig = (*doppler).dopcom - idop_use[0] + 1;

    /*  If this node handles this dataset,
        allocate memory for observed data and fits  */

    (*doppler).frame[i].obs = vector( 1, (*doppler).frame[i].ndop);
    (*doppler).frame[i].fit = vector( 1, (*doppler).frame[i].ndop);
    (*doppler).frame[i].oneovervar = vector( 1, (*doppler).frame[i].ndop);
    FOPEN( fin, fullname, "r");

    /*  Read first line of file so that we can identify file type  */

    if (!fread(buffer, bufferlength, 1, fin))
    	bailout("read_doppler in read_dat.c: can't read first line of data file\n");
    rewind(fin);

    /*  Set binary flag if first line is not ASCII  */

    is_binary = 0;
    for (j=0; j<bufferlength; j++)
    	if (!isascii(buffer[j]))
    		is_binary = 1;

    /*  Read data according to file type  */

    swap_bytes = ( is_little_endian() && (*par).endian == BIG_ENDIAN_DATA) ||
    		(!is_little_endian() && (*par).endian == LITTLE_ENDIAN_DATA);
    is_fits = !is_binary && (strstr(buffer, "SIMPLE") != NULL);
    lowcase(buffer);
    is_rdf = !is_binary && !is_fits && (strstr(buffer, "type") != NULL);

    if (is_fits)
    	read_doppler_fits(fullname, doppler, i, idop_use);
    else if (is_rdf)
    	read_doppler_rdf(fin, doppler, i, idop_use, swap_bytes);
    else if (is_binary)
    	read_doppler_binary(fin, doppler, i, idop_use, swap_bytes);
    else
    	read_doppler_ascii(fin, doppler, i, idop_use);

    /*  Get variance for each bin; apply and bin weighting if desired  */

    kskip = idop_use[0] - 1;

    for (k=1; k<=(*doppler).frame[i].ndop; k++) {
    	if ((*par).speckle)
    		(*doppler).frame[i].oneovervar[k] =
    				1/( pow((*doppler).frame[i].sdev,2.0) +
    						pow(lookfact*(*doppler).frame[i].obs[k],2.0) );
    	else
    		(*doppler).frame[i].oneovervar[k] = 1/pow((*doppler).frame[i].sdev,2.0);
    	if ((*doppler).frame[i].pixels_weighted)
    		(*doppler).frame[i].oneovervar[k] *= binweight[k+kskip];
    }
    fclose( fin);


    /*  Free memory for the bin-weighting mask  */

    if ((*doppler).frame[i].pixels_weighted)
      free_vector( binweight, 1, (*doppler).ndop);

    /*  Loop through all views contributing to this (smeared) frame  */

    for (k=0; k<(*doppler).nviews; k++) {

      /*  Compute the epoch of this view, uncorrected for light-travel time  */

      (*doppler).frame[i].view[k].t = (*doppler).frame[i].t0
                                      + (k - (*doppler).v0)*(*doppler).view_interval;

      /*  Use this dataset's ephemeris (and linear interpolation) to get the target's
          distance (AU) at this view's epoch.  Also compute frame[i].view[k].oe, the
          transformation matrix from ecliptic to observer coordinates at that epoch,
          and frame[i].view[k].orbspin, the plane-of-sky-motion contribution to the
          apparent spin vector at that epoch.  (orbspin is in ecliptic coordinates.)

          Note that the final "0" in the call to ephem2mat signifies monostatic
          observations; this means that arguments 2 and 5, which would pertain to
          the Sun's ephemeris, are unused here, and arguments 7 and 8 (solar phase
          angle and azimuth angle) are returned as 0.                                  */

      dist = ephem2mat( (*doppler).astephem, (*doppler).astephem,
                        (*doppler).frame[i].view[k].t, (*doppler).frame[i].view[k].oe, se,
                        (*doppler).frame[i].view[k].orbspin, &solar_phase, &solar_azimuth, 0);

      /*  If the perform_ltc parameter is turned on, use the distance just
          obtained to subtract the one-way light-travel time from this view's
          epoch; then go back to the ephemeris and recompute frame[i].view[k].oe
          and frame[i].view[k].orbspin for the corrected epoch.                   */

      if ((*par).perform_ltc) {
        (*doppler).frame[i].view[k].t -= DAYSPERAU*dist;
        ephem2mat( (*doppler).astephem, (*doppler).astephem,
                   (*doppler).frame[i].view[k].t, (*doppler).frame[i].view[k].oe, se,
                   (*doppler).frame[i].view[k].orbspin, &solar_phase, &solar_azimuth, 0);
      }

    }

    /*  Initialize quantities related to spin impulses  */

    (*doppler).frame[i].n_integrate = -999;
    for (n=0; n<MAXIMP+2; n++) {
      (*doppler).frame[i].t_integrate[n] = -HUGENUMBER;
      for (j=0; j<=2; j++)
        (*doppler).frame[i].impulse[n][j] = 0.0;
    }

   printf("#     %s\n", fullname);
   fflush(stdout);
  }

  return npar;
}

void set_up_pos( struct par_t *par, struct dat_t *dat)
{
  int s, f, i, n;

  /*  Allocate memory for a plane-of-sky rendering for the data as a whole;
      as discussed below, if pos_scope = "global" then this will be the
      ONLY memory allocated for POS renderings.                              */

  n = ((*par).pos_pixels - 1)/2;
  (*dat).pos.n = n;
  (*dat).pos.b = matrix( -n, n, -n, n);
  (*dat).pos.cosi = matrix( -n, n, -n, n);
  (*dat).pos.cose = matrix( -n, n, -n, n);
  (*dat).pos.z = matrix( -n, n, -n, n);
  (*dat).pos.body = imatrix( -n, n, -n, n);
  (*dat).pos.comp = imatrix( -n, n, -n, n);
  (*dat).pos.f = imatrix( -n, n, -n, n);
  (*dat).pos.km_per_pixel = (*par).pos_width/(2.0*n);
  (*dat).pos.cosill = matrix( -n, n, -n, n);
  (*dat).pos.zill = matrix( -n, n, -n, n);
  (*dat).pos.bodyill = imatrix( -n, n, -n, n);
  (*dat).pos.compill = imatrix( -n, n, -n, n);
  (*dat).pos.fill = imatrix( -n, n, -n, n);
  (*dat).pos.bistatic = 1;

  /*  If the pos_scope parameter is "global" then there is just ONE chunk of memory
      allocated for a POS image and associated variables (a pos_t structure), and
      every delay-Doppler frame, Doppler frame, POS frame, and lightcurve point
      shares this one memory slot for its POS rendering.  Hence these POS images
      and associated variables -- e.g., cos(scattering angle) for each POS pixel --
      amount to (very) temporary workspace, soon to be overwritten.

      If pos_scope = "local" then each data frame or lightcurve point gets its own
      pos_t structure in memory, so the POS images and associated variables persist
      rather than being overwritten by other frames/points.                            */
  if (HMT)	{
	  par->pos_scope = LOCAL;
	  printf("\n###     Switching pos_scope to LOCAL for Host-Multi-Threaded Mode...     ###\n");
  }
  switch ((*par).pos_scope) {
  case GLOBAL:
      for (s=0; s<(*dat).nsets; s++) {
        switch ((*dat).set[s].type) {
        case DELAY:
            for (f=0; f<(*dat).set[s].desc.deldop.nframes; f++)
              (*dat).set[s].desc.deldop.frame[f].pos = (*dat).pos;
            break;
        case DOPPLER:
            for (f=0; f<(*dat).set[s].desc.doppler.nframes; f++)
              (*dat).set[s].desc.doppler.frame[f].pos = (*dat).pos;
            break;
        case POS:
            for (f=0; f<(*dat).set[s].desc.poset.nframes; f++)
              (*dat).set[s].desc.poset.frame[f].pos = (*dat).pos;
            break;
        case LGHTCRV:
              for (i=1; i<=(*dat).set[s].desc.lghtcrv.ncalc; i++)
              (*dat).set[s].desc.lghtcrv.rend[i].pos = (*dat).pos;
            break;
        default:
            bailout("set_up_pos in read_dat.c: can't do that type yet\n");
        }
      }
      break;
  case LOCAL:
      for (s=0; s<(*dat).nsets; s++) {
        switch ((*dat).set[s].type) {
        case DELAY:
            for (f=0; f<(*dat).set[s].desc.deldop.nframes; f++) {
              (*dat).set[s].desc.deldop.frame[f].pos.n = n;
              (*dat).set[s].desc.deldop.frame[f].pos.b = matrix( -n, n, -n, n);
              (*dat).set[s].desc.deldop.frame[f].pos.cosi = matrix( -n, n, -n, n);
              (*dat).set[s].desc.deldop.frame[f].pos.cose = matrix( -n, n, -n, n);
              (*dat).set[s].desc.deldop.frame[f].pos.z = matrix( -n, n, -n, n);
              (*dat).set[s].desc.deldop.frame[f].pos.body = imatrix( -n, n, -n, n);
              (*dat).set[s].desc.deldop.frame[f].pos.comp = imatrix( -n, n, -n, n);
              (*dat).set[s].desc.deldop.frame[f].pos.f = imatrix( -n, n, -n, n);
              (*dat).set[s].desc.deldop.frame[f].pos.km_per_pixel = (*par).pos_width/(2.0*n);
              (*dat).set[s].desc.deldop.frame[f].pos.cosill = matrix( -n, n, -n, n);
              (*dat).set[s].desc.deldop.frame[f].pos.zill = matrix( -n, n, -n, n);
              (*dat).set[s].desc.deldop.frame[f].pos.bodyill = imatrix( -n, n, -n, n);
              (*dat).set[s].desc.deldop.frame[f].pos.compill = imatrix( -n, n, -n, n);
              (*dat).set[s].desc.deldop.frame[f].pos.fill = imatrix( -n, n, -n, n);
              (*dat).set[s].desc.deldop.frame[f].pos.bistatic = 0;
            }
            break;
        case DOPPLER:
            for (f=0; f<(*dat).set[s].desc.doppler.nframes; f++) {
              (*dat).set[s].desc.doppler.frame[f].pos.n = n;
              (*dat).set[s].desc.doppler.frame[f].pos.b = matrix( -n, n, -n, n);
              (*dat).set[s].desc.doppler.frame[f].pos.cosi = matrix( -n, n, -n, n);
              (*dat).set[s].desc.doppler.frame[f].pos.cose = matrix( -n, n, -n, n);
              (*dat).set[s].desc.doppler.frame[f].pos.z = matrix( -n, n, -n, n);
              (*dat).set[s].desc.doppler.frame[f].pos.body = imatrix( -n, n, -n, n);
              (*dat).set[s].desc.doppler.frame[f].pos.comp = imatrix( -n, n, -n, n);
              (*dat).set[s].desc.doppler.frame[f].pos.f = imatrix( -n, n, -n, n);
              (*dat).set[s].desc.doppler.frame[f].pos.km_per_pixel = (*par).pos_width/(2.0*n);
              (*dat).set[s].desc.doppler.frame[f].pos.cosill = matrix( -n, n, -n, n);
              (*dat).set[s].desc.doppler.frame[f].pos.zill = matrix( -n, n, -n, n);
              (*dat).set[s].desc.doppler.frame[f].pos.bodyill = imatrix( -n, n, -n, n);
              (*dat).set[s].desc.doppler.frame[f].pos.compill = imatrix( -n, n, -n, n);
              (*dat).set[s].desc.doppler.frame[f].pos.fill = imatrix( -n, n, -n, n);
              (*dat).set[s].desc.doppler.frame[f].pos.bistatic = 0;
            }
            break;
        case POS:
            for (f=0; f<(*dat).set[s].desc.poset.nframes; f++) {
              (*dat).set[s].desc.poset.frame[f].pos.n = n;
              (*dat).set[s].desc.poset.frame[f].pos.b = matrix( -n, n, -n, n);
              (*dat).set[s].desc.poset.frame[f].pos.cosi = matrix( -n, n, -n, n);
              (*dat).set[s].desc.poset.frame[f].pos.cose = matrix( -n, n, -n, n);
              (*dat).set[s].desc.poset.frame[f].pos.z = matrix( -n, n, -n, n);
              (*dat).set[s].desc.poset.frame[f].pos.body = imatrix( -n, n, -n, n);
              (*dat).set[s].desc.poset.frame[f].pos.comp = imatrix( -n, n, -n, n);
              (*dat).set[s].desc.poset.frame[f].pos.f = imatrix( -n, n, -n, n);
              (*dat).set[s].desc.poset.frame[f].pos.km_per_pixel = (*par).pos_width/(2.0*n);
              (*dat).set[s].desc.poset.frame[f].pos.cosill = matrix( -n, n, -n, n);
              (*dat).set[s].desc.poset.frame[f].pos.zill = matrix( -n, n, -n, n);
              (*dat).set[s].desc.poset.frame[f].pos.bodyill = imatrix( -n, n, -n, n);
              (*dat).set[s].desc.poset.frame[f].pos.compill = imatrix( -n, n, -n, n);
              (*dat).set[s].desc.poset.frame[f].pos.fill = imatrix( -n, n, -n, n);
              (*dat).set[s].desc.poset.frame[f].pos.bistatic = 1;
            }
            break;
        case LGHTCRV:
            for (i=1; i<=(*dat).set[s].desc.lghtcrv.ncalc; i++) {
              (*dat).set[s].desc.lghtcrv.rend[i].pos.n = n;
              (*dat).set[s].desc.lghtcrv.rend[i].pos.b = matrix( -n, n, -n, n);
              (*dat).set[s].desc.lghtcrv.rend[i].pos.cosi = matrix( -n, n, -n, n);
              (*dat).set[s].desc.lghtcrv.rend[i].pos.cose = matrix( -n, n, -n, n);
              (*dat).set[s].desc.lghtcrv.rend[i].pos.z = matrix( -n, n, -n, n);
              (*dat).set[s].desc.lghtcrv.rend[i].pos.body = imatrix( -n, n, -n, n);
              (*dat).set[s].desc.lghtcrv.rend[i].pos.comp = imatrix( -n, n, -n, n);
              (*dat).set[s].desc.lghtcrv.rend[i].pos.f = imatrix( -n, n, -n, n);
              (*dat).set[s].desc.lghtcrv.rend[i].pos.km_per_pixel = (*par).pos_width/(2.0*n);
              (*dat).set[s].desc.lghtcrv.rend[i].pos.cosill = matrix( -n, n, -n, n);
              (*dat).set[s].desc.lghtcrv.rend[i].pos.zill = matrix( -n, n, -n, n);
              (*dat).set[s].desc.lghtcrv.rend[i].pos.bodyill = imatrix( -n, n, -n, n);
              (*dat).set[s].desc.lghtcrv.rend[i].pos.compill = imatrix( -n, n, -n, n);
              (*dat).set[s].desc.lghtcrv.rend[i].pos.fill = imatrix( -n, n, -n, n);
              (*dat).set[s].desc.lghtcrv.rend[i].pos.bistatic = 1;
            }
            break;
        default:
            bailout("set_up_pos in read_dat.c: can't do that type yet\n");
        }
      }
      break;
  default:
      bailout("set_up_pos in read_dat.c: undefined case\n");
  }
}

int read_poset( FILE *fp, struct par_t *par, struct poset_t *poset,
                          int noptlaws, int s, double *chi2_variance)
{
  int i, j, npar=0, k, jskip, kskip, bistatic=0, irow_use[2], icol_use[2],
      nrow_raw, ncol_raw, nrow, ncol, n, nmaskvals;
  char fullname[160], smearingstring[7], pixweightfile[MAXLEN];
  FILE *wp=0;
  double dist, dof, **pixweight=NULL;


  /*  Initialize degrees of freedom and variance of chi2 estimate  */

  (*poset).dof = 0.0;
  *chi2_variance = 0.0;

  /*  Read which optical scattering law to use for this dataset  */

  (*poset).ioptlaw = getint( fp);
  if ((*poset).ioptlaw < 0 || (*poset).ioptlaw >= noptlaws) {
    printf("ERROR in set %d: must have 0 <= optical scattering law <= %d\n",
           s, noptlaws-1);
    bailout("read_poset in read_dat.c\n");
  }

  /*  Read the asteroid ephemeris  */

  (*poset).astephem.n = getint( fp); /* # of points in ephemeris */
  (*poset).astephem.pnt = (struct ephpnt_t *) calloc( (*poset).astephem.n,
                                                      sizeof( struct ephpnt_t));
  for (i=0; i<(*poset).astephem.n; i++) {
    rdcal2jd( fp, &(*poset).astephem.pnt[i].t);
    (*poset).astephem.pnt[i].ra = getdouble( fp)*D2R;
    (*poset).astephem.pnt[i].dec = getdouble( fp)*D2R;
    (*poset).astephem.pnt[i].dist = getdouble( fp);
  }

  /*  Read the solar ephemeris  */

  (*poset).solephem.n = getint( fp); /* # of points in ephemeris */
  (*poset).solephem.pnt = (struct ephpnt_t *) calloc( (*poset).solephem.n,
                                                      sizeof( struct ephpnt_t));
  for (i=0; i<(*poset).solephem.n; i++) {
    rdcal2jd( fp, &(*poset).solephem.pnt[i].t);
    (*poset).solephem.pnt[i].ra = getdouble( fp)*D2R;
    (*poset).solephem.pnt[i].dec = getdouble( fp)*D2R;
    if (((*poset).solephem.pnt[i].dist = getdouble( fp)) > 0.0)
      bistatic = 1;
  }

  /*  Input angular pixel size in arcseconds, convert to radians  */

  (*poset).angle_per_pixel = (getdouble( fp)/3600)*D2R;

  /*  Read smearing information  */

  (*poset).nviews = getint( fp); /* # views per frame */
  (*poset).view_interval = getdouble( fp); /* view interval (s) */
  (*poset).view_interval /= 86400; /* convert to days */
  gettstr( fp, smearingstring);   /* smearing mode */
  if (!strcmp(smearingstring, "center")) {
      (*poset).smearing_mode = SMEAR_CENTER;
      (*poset).v0 = (*poset).nviews / 2;
  } else if (!strcmp(smearingstring, "first")) {
      (*poset).smearing_mode = SMEAR_FIRST;
      (*poset).v0 = 0;
  } else {
      bailout("read_poset in read_dat.c: can't do that smearing mode yet\n");
  }

  /*  Get the data directory and the number of frames in the dataset  */

  gettstr( fp, (*poset).dir);
  (*poset).nframes = getint( fp);
  (*poset).frame = (struct posetfrm_t *)
                   calloc( (*poset).nframes, sizeof(struct posetfrm_t));
  for (i=0; i<(*poset).nframes; i++)
    (*poset).frame[i].view = (struct posetview_t *)
                             calloc( (*poset).nviews, sizeof(struct posetview_t));

  /*  Loop through the frames  */

  for (i=0; i<(*poset).nframes; i++) {
    gettstr( fp, (*poset).frame[i].name); /* name of data file */
    sprintf( fullname, "%s/%s", (*poset).dir, (*poset).frame[i].name);
    rdcal2jd( fp, &(*poset).frame[i].t0);

    /*  Read angle at which north points (counter-clockwise from upward)  */

    (*poset).frame[i].northangle = getdouble( fp)*D2R;

    /*  The calibration factor is no longer in the obs file,
        so force it to float and assign it a dummy value      */

    (*poset).frame[i].cal.state = 'f';
    (*poset).frame[i].cal.val = -9.99;

    /*  Read the x offset (columns) and y offset (rows) of the COM for this frame  */

    npar += readparam( fp, &(*poset).frame[i].off[0]);
    npar += readparam( fp, &(*poset).frame[i].off[1]);

    /*  Read this frame's relative weight and the pixel-weighting "mask" flag  */

    (*poset).frame[i].weight = getdouble( fp);
    (*poset).frame[i].pixels_weighted = getint( fp);

    /*  Get the raw image dimensions from the image file  */

    read_poset_fits(fullname, poset, i, icol_use, irow_use, 0);
    ncol_raw = (*poset).frame[i].ncol;
    nrow_raw = (*poset).frame[i].nrow;

    /*  If a pixel-weighting mask is being used, read it in;
        then get the frame's vignetted dimensions and degrees of freedom  */

    dof = 0.0;
    if ((*poset).frame[i].pixels_weighted) {

        /*  Open the mask file, count the entries, and make sure that it's the
            expected number of entries (in case someone has changed the numbering
            of datasets without changing mask filenames accordingly)

            Note that the countdata routine resets the file position indicator to
            its initial value after it finishes reading data entries               */

        if (strcmp( (*par).maskdir, "")) {
            if ((*poset).nframes > 100)
              sprintf( pixweightfile, "%s/mask_%02d_%03d.txt", (*par).maskdir, s, i);
            else
              sprintf( pixweightfile, "%s/mask_%02d_%02d.txt", (*par).maskdir, s, i);
        } else {
            if ((*poset).nframes > 100)
              sprintf( pixweightfile, "%s/mask_%02d_%03d.txt", (*poset).dir, s, i);
            else
              sprintf( pixweightfile, "%s/mask_%02d_%02d.txt", (*poset).dir, s, i);
        }
        FOPEN( wp, pixweightfile, "r");
        nmaskvals = countdata( wp);
        if (nmaskvals != ncol_raw*nrow_raw) {
          fprintf(stderr,"ERROR: expected %d x %d = %d mask weights for set %2d frame %2d\n",
                  ncol_raw, nrow_raw, ncol_raw*nrow_raw, s, i);
          fprintf(stderr,"       -- instead there are %d weights in %s\n",
                  nmaskvals, pixweightfile);
          bailout("read_poset in read_dat.c\n");
        }

        /*  Allocate memory for the mask values and read them in.

            Note that plane-of-sky image arrays have columns as the first
            index and rows as the second: see comment in read_poset_fits.  */

        pixweight = matrix( 1, ncol_raw, 1, nrow_raw);
        icol_use[0] = ncol_raw + 1;
        icol_use[1] = 0;
        irow_use[0] = nrow_raw + 1;
        irow_use[1] = 0;
        for (k=1; k<=nrow_raw; k++)
          for (j=1; j<=ncol_raw; j++) {
            pixweight[j][k] = getdouble( wp);
            if (pixweight[j][k] > 0.0) {
              dof += (*poset).frame[i].weight;
              *chi2_variance += 2 * (*poset).frame[i].weight * (*poset).frame[i].weight;
              icol_use[0] = MIN( icol_use[0], j);
              icol_use[1] = MAX( icol_use[1], j);
              irow_use[0] = MIN( irow_use[0], k);
              irow_use[1] = MAX( irow_use[1], k);
            }
          }
        fclose( wp);
    } else {
        dof = (*poset).frame[i].weight * ncol_raw * nrow_raw;
        *chi2_variance += 2 * (*poset).frame[i].weight * (*poset).frame[i].weight
                            * ncol_raw * nrow_raw;
        icol_use[0] = 1;
        icol_use[1] = ncol_raw;
        irow_use[0] = 1;
        irow_use[1] = nrow_raw;
    }
    (*poset).frame[i].dof = dof;
    (*poset).dof += dof;
    (*poset).frame[i].ncol = ncol = icol_use[1] - icol_use[0] + 1;
    (*poset).frame[i].nrow = nrow = irow_use[1] - irow_use[0] + 1;
    (*poset).frame[i].colcom_vig = (ncol_raw + 1)/2.0 - icol_use[0] + 1;
    (*poset).frame[i].rowcom_vig = (nrow_raw + 1)/2.0 - irow_use[0] + 1;

    /*  If this node handles this dataset,
        allocate memory for observed data and fits  */

    (*poset).frame[i].obs.b = matrix( 1, ncol, 1, nrow);
    (*poset).frame[i].fit.b = matrix( 1, ncol, 1, nrow);
    (*poset).frame[i].fit.z = matrix( 1, ncol, 1, nrow);
    (*poset).frame[i].fit.f = imatrix( 1, ncol, 1, nrow);
    (*poset).frame[i].oneovervar = matrix( 1, ncol, 1, nrow);

    /*  Now read in the pixel values  */

    read_poset_fits(fullname, poset, i, icol_use, irow_use, 1);

    /*  Apply pixel weighting (if any)  */

    jskip = icol_use[0] - 1;
    kskip = irow_use[0] - 1;

    for (j=1; j<=ncol; j++)
    	for (k=1; k<=nrow; k++)
    		if ((*poset).frame[i].pixels_weighted)
    			(*poset).frame[i].oneovervar[j][k] = pixweight[j+jskip][k+kskip];
    		else
    			(*poset).frame[i].oneovervar[j][k] = 1.0;

    /*  Free memory for the pixel-weighting mask  */

    if ((*poset).frame[i].pixels_weighted)
    	free_matrix( pixweight, 1, ncol_raw, 1, nrow_raw);

    /*  Loop through all views contributing to this (smeared) frame  */

    for (k=0; k<(*poset).nviews; k++) {

      /*  Compute the epoch of this view, uncorrected for light-travel time  */

      (*poset).frame[i].view[k].t = (*poset).frame[i].t0
                                    + (k - (*poset).v0)*(*poset).view_interval;

      /*  Use this dataset's ephemeris (and linear interpolation) to get the target's
          distance (AU) at this view's epoch.  Also compute frame[i].view[k].oe, the
          transformation matrix from ecliptic to observer coordinates at that epoch;
          frame[i].view[k].se, the transformation matrix from ecliptic to source (solar)
          coordinates; frame[i].view[k].orbspin, the plane-of-sky-motion contribution to
          the apparent spin vector at that epoch (in ecliptic coordinates); and the
          solar phase angle and azimuth angle (north through east).                       */

      dist = ephem2mat( (*poset).astephem, (*poset).solephem,
                        (*poset).frame[i].view[k].t,
                        (*poset).frame[i].view[k].oe, (*poset).frame[i].view[k].se,
                        (*poset).frame[i].view[k].orbspin,
                        &(*poset).frame[i].view[k].solar_phase,
                        &(*poset).frame[i].view[k].solar_azimuth, bistatic);

      /*  If the perform_ltc parameter is turned on, use the distance
          just obtained to subtract the one-way light-travel time from
          this view's epoch; then go back to the ephemeris and recompute
          the various quantities for the corrected epoch.                  */

      if ((*par).perform_ltc) {
        (*poset).frame[i].view[k].t -= DAYSPERAU*dist;
        ephem2mat( (*poset).astephem, (*poset).solephem,
                   (*poset).frame[i].view[k].t,
                   (*poset).frame[i].view[k].oe, (*poset).frame[i].view[k].se,
                   (*poset).frame[i].view[k].orbspin,
                   &(*poset).frame[i].view[k].solar_phase,
                   &(*poset).frame[i].view[k].solar_azimuth, bistatic);
      }

    }

    /*  Convert angular pixel size to linear pixel size  */

    (*poset).frame[i].fit.km_per_pixel = (*poset).angle_per_pixel*(dist*AU);

    /*  Initialize quantities related to spin impulses  */

    (*poset).frame[i].n_integrate = -999;
    for (n=0; n<MAXIMP+2; n++) {
      (*poset).frame[i].t_integrate[n] = -HUGENUMBER;
      for (j=0; j<=2; j++)
        (*poset).frame[i].impulse[n][j] = 0.0;
    }
    printf("#     %s\n", fullname);
    fflush(stdout);
  }

  return npar;
}

int read_lghtcrv( FILE *fp, struct par_t *par, struct lghtcrv_t *lghtcrv,
                            int noptlaws, int s, unsigned char type,
                            double *chi2_variance)
{
  int i, k, npar=0, np, nunique, n, j, extrapolate_flag;
  char smearingstring[7];
  FILE *fin;
  double x, y, z, orbspin[3], dist, solar_phase, solar_azimuth, oe[3][3], se[3][3],
         t1, t2, dt, obsepoch, obsmag, obsmagerr, sunmag, obsintens, obsintenserr;
  double *obsepoch_raw, *obsepoch_unique;

  /*  Initialize some variables to avoid compilation warnings  */

  t1 = t2 = dt = 0.0;
  obsepoch_raw = NULL;

  /*  Read which optical scattering law to use for this dataset  */

  lghtcrv->ioptlaw = getint( fp);
  if (lghtcrv->ioptlaw < 0 || lghtcrv->ioptlaw >= noptlaws) {
    printf("ERROR in set %d: must have 0 <= optical scattering law <= %d\n",
           s, noptlaws-1);
    bailout("read_lghtcrv in read_dat.c\n");
  }

  /*  If this is an ordinary lightcurve dataset, read the asteroid and solar ephemerides;
      if it is a heliocurve dataset, read the ephemerides of the asteroid and
      observer as viewed from the Sun, then convert to asteroid and solar ephemerides
      as viewed by the observer                                                            */

  if (type == LGHTCRV) {

      /*  Deal with ephemeris types that won't be used  */

      lghtcrv->astephem_sun.n = lghtcrv->obsephem_sun.n = -1;
      lghtcrv->astephem_sun.pnt = lghtcrv->obsephem_sun.pnt = NULL;

      /*  Read the asteroid ephemeris  */

      lghtcrv->astephem.n = getint( fp); /* # of points in ephemeris */
      lghtcrv->astephem.pnt = (struct ephpnt_t *) calloc( lghtcrv->astephem.n,
                                                            sizeof( struct ephpnt_t));
      for (i=0; i<lghtcrv->astephem.n; i++) {
        rdcal2jd( fp, &lghtcrv->astephem.pnt[i].t);
        lghtcrv->astephem.pnt[i].ra = getdouble( fp)*D2R;
        lghtcrv->astephem.pnt[i].dec = getdouble( fp)*D2R;
        lghtcrv->astephem.pnt[i].dist = getdouble( fp);
      }

      /*  Read the solar ephemeris  */

      lghtcrv->solephem.n = getint( fp); /* # of points in ephemeris */
      lghtcrv->solephem.pnt = (struct ephpnt_t *) calloc( lghtcrv->solephem.n,
                                                            sizeof( struct ephpnt_t));
      for (i=0; i<lghtcrv->solephem.n; i++) {
        rdcal2jd( fp, &lghtcrv->solephem.pnt[i].t);
        lghtcrv->solephem.pnt[i].ra = getdouble( fp)*D2R;
        lghtcrv->solephem.pnt[i].dec = getdouble( fp)*D2R;
        lghtcrv->solephem.pnt[i].dist = getdouble( fp);
      }
  } else {

      /*  Read the asteroid ephemeris as viewed from the Sun  */

      lghtcrv->astephem_sun.n = getint( fp); /* # of points in ephemeris */
      lghtcrv->astephem_sun.pnt = (struct ephpnt_t *) calloc( lghtcrv->astephem_sun.n,
                                                                sizeof( struct ephpnt_t));
      for (i=0; i<lghtcrv->astephem_sun.n; i++) {
        rdcal2jd( fp, &lghtcrv->astephem_sun.pnt[i].t);
        lghtcrv->astephem_sun.pnt[i].ra = getdouble( fp)*D2R;
        lghtcrv->astephem_sun.pnt[i].dec = getdouble( fp)*D2R;
        lghtcrv->astephem_sun.pnt[i].dist = getdouble( fp);
      }

      /*  Read the observer ephemeris as viewed from the Sun  */

      lghtcrv->obsephem_sun.n = getint( fp); /* # of points in ephemeris */
      if (lghtcrv->obsephem_sun.n != lghtcrv->astephem_sun.n) {
        printf("ERROR for dataset %d: must have the same number of ephemeris entries\n", s);
        printf("                         for the asteroid and the observer\n");
        bailout("read_dat.c\n");
      }
      lghtcrv->obsephem_sun.pnt = (struct ephpnt_t *) calloc( lghtcrv->obsephem_sun.n,
                                                                sizeof( struct ephpnt_t));
      for (i=0; i<lghtcrv->obsephem_sun.n; i++) {
        rdcal2jd( fp, &lghtcrv->obsephem_sun.pnt[i].t);
        if (lghtcrv->obsephem_sun.pnt[i].t != lghtcrv->astephem_sun.pnt[i].t) {
          printf("ERROR for dataset %d: must have the same set of ephemeris epochs\n", s);
          printf("                         for the asteroid and the observer\n");
          bailout("read_dat.c\n");
        }
        lghtcrv->obsephem_sun.pnt[i].ra = getdouble( fp)*D2R;
        lghtcrv->obsephem_sun.pnt[i].dec = getdouble( fp)*D2R;
        lghtcrv->obsephem_sun.pnt[i].dist = getdouble( fp);
      }

      /*  Construct the asteroid ephemeris as viewed by the observer  */

      lghtcrv->astephem.n = lghtcrv->astephem_sun.n;
      lghtcrv->astephem.pnt = (struct ephpnt_t *) calloc( lghtcrv->astephem.n,
                                                             sizeof( struct ephpnt_t));
      for (i=0; i<lghtcrv->astephem.n; i++) {
        lghtcrv->astephem.pnt[i].t = lghtcrv->astephem_sun.pnt[i].t;
        x =   lghtcrv->astephem_sun.pnt[i].dist * cos( lghtcrv->astephem_sun.pnt[i].dec)
                                                  * cos( lghtcrv->astephem_sun.pnt[i].ra)
            - lghtcrv->obsephem_sun.pnt[i].dist * cos( lghtcrv->obsephem_sun.pnt[i].dec)
                                                  * cos( lghtcrv->obsephem_sun.pnt[i].ra);
        y =   lghtcrv->astephem_sun.pnt[i].dist * cos( lghtcrv->astephem_sun.pnt[i].dec)
                                                  * sin( lghtcrv->astephem_sun.pnt[i].ra)
            - lghtcrv->obsephem_sun.pnt[i].dist * cos( lghtcrv->obsephem_sun.pnt[i].dec)
                                                  * sin( lghtcrv->obsephem_sun.pnt[i].ra);
        z =   lghtcrv->astephem_sun.pnt[i].dist * sin( lghtcrv->astephem_sun.pnt[i].dec)
            - lghtcrv->obsephem_sun.pnt[i].dist * sin( lghtcrv->obsephem_sun.pnt[i].dec);
        lghtcrv->astephem.pnt[i].dist = sqrt(x*x + y*y + z*z);
        lghtcrv->astephem.pnt[i].ra = atan2( y, x);
        lghtcrv->astephem.pnt[i].dec = asin( z / lghtcrv->astephem.pnt[i].dist);
      }

      /*  Construct the solar ephemeris as viewed by the observer  */

      lghtcrv->solephem.n = lghtcrv->obsephem_sun.n;
      lghtcrv->solephem.pnt = (struct ephpnt_t *) calloc( lghtcrv->solephem.n,
                                                             sizeof( struct ephpnt_t));
      for (i=0; i<lghtcrv->solephem.n; i++) {
        lghtcrv->solephem.pnt[i].t = lghtcrv->obsephem_sun.pnt[i].t;
        lghtcrv->solephem.pnt[i].ra = lghtcrv->obsephem_sun.pnt[i].ra - PIE;
        if (lghtcrv->solephem.pnt[i].ra < 0.0)
          lghtcrv->solephem.pnt[i].ra += 2*PIE;
        lghtcrv->solephem.pnt[i].dec = -lghtcrv->obsephem_sun.pnt[i].dec;
        lghtcrv->solephem.pnt[i].dist = lghtcrv->obsephem_sun.pnt[i].dist;
      }
  }

  /*  Get the number of epochs at which to calculate the model brightness:
          positive --> an explicit list of epochs follows
          zero     --> specification of evenly spaced epochs follows
          negative --> use the same epochs as for the observations         */

  np = lghtcrv->ncalc_obsfile = getint( fp);
  if (np != 0)
    lghtcrv->jdstart = lghtcrv->jdstop = lghtcrv->jdinterval = -HUGENUMBER;

  /*  If the next lines explicitly or implicitly specify the calculation epochs,
      read them now and then generate the relevant quantities for each epoch
      (coordinate transformation matrices, POS spin components, solar phase
      angles, solar azimuth angles in the POS)                                    */

  if (np >= 0) {
    if (np > 0) {

        /*  We have an explicit list of calculated points  */

        lghtcrv->ncalc = np;
    } else {

        /*  Generate the list of calculated points from JD start/stop/interval  */

        t1 = lghtcrv->jdstart    = getdouble( fp);  /* start time */
        t2 = lghtcrv->jdstop     = getdouble( fp);  /* stop time  */
        dt = lghtcrv->jdinterval = getdouble( fp);  /* time step  */
        if (t2 < t1 || dt <= 0) {
          printf("ERROR in set %d calculation epochs: need t2 >= t1 and dt > 0\n", s);
          bailout("read_lghtcrv in read_dat.c\n");
        }
        lghtcrv->ncalc = ((int)floor((t2 - t1)/dt)) + 1;  /* # points */
    }
    lghtcrv->x0 = vector( 1, lghtcrv->ncalc);
    lghtcrv->x = vector( 1, lghtcrv->ncalc);
    lghtcrv->y = vector( 1, lghtcrv->ncalc);
    lghtcrv->y2 = vector( 1, lghtcrv->ncalc);
    lghtcrv->rotphase_calc = vector( 1, lghtcrv->ncalc);
    lghtcrv->solar_phase = vector( 1, lghtcrv->ncalc);
    lghtcrv->solar_azimuth = vector( 1, lghtcrv->ncalc);
    lghtcrv->rend = (struct crvrend_t *) calloc( lghtcrv->ncalc+1,
                                                   sizeof( struct crvrend_t));
    if (np > 0) {
        for (i=1; i<=lghtcrv->ncalc; i++)
          lghtcrv->x[i] = lghtcrv->x0[i] = getdouble( fp);
        for (i=2; i<=lghtcrv->ncalc; i++) {
          if (lghtcrv->x0[i] <= lghtcrv->x0[i-1]) {
            printf("ERROR in set %d: calculation epoch %d <= epoch %d\n", s, i, i-1);
            bailout("read_lghtcrv in read_dat.c\n");
          }
        }
    } else {
        for (i=1; i<=lghtcrv->ncalc; i++)
          lghtcrv->x[i] = lghtcrv->x0[i] = t1 + (i - 1)*dt;
    }

    for (i=1; i<=lghtcrv->ncalc; i++) {
      dist = ephem2mat( lghtcrv->astephem, lghtcrv->solephem,
                        lghtcrv->x0[i],
                        lghtcrv->rend[i].oe, lghtcrv->rend[i].se,
                        lghtcrv->rend[i].orbspin,
                        &lghtcrv->solar_phase[i], &lghtcrv->solar_azimuth[i], 1);
      if ((*par).perform_ltc) {                         /* apply light-time correction */
        lghtcrv->x[i] = lghtcrv->x0[i] - DAYSPERAU*dist;
        ephem2mat( lghtcrv->astephem, lghtcrv->solephem,
                   lghtcrv->x[i],
                   lghtcrv->rend[i].oe, lghtcrv->rend[i].se,
                   lghtcrv->rend[i].orbspin,
                   &lghtcrv->solar_phase[i], &lghtcrv->solar_azimuth[i], 1);
      }
    }
  }

  /*  Read smearing information  */

  lghtcrv->nviews = getint( fp); /* # views per point */
  lghtcrv->view_interval = getdouble( fp); /* view interval (s) */
  lghtcrv->view_interval /= 86400; /* convert to days */
  gettstr( fp, smearingstring);   /* smearing mode */
  if (!strcmp(smearingstring, "center")) {
      lghtcrv->smearing_mode = SMEAR_CENTER;
      lghtcrv->v0 = lghtcrv->nviews / 2;
  } else if (!strcmp(smearingstring, "first")) {
      lghtcrv->smearing_mode = SMEAR_FIRST;
      lghtcrv->v0 = 0;
  } else {
      bailout("read_lghtcrv in read_dat.c: can't do that smearing mode yet\n");
  }

  /*  Read the number of observed lightcurve points, the datafile name,
      and the calibration factor for this lightcurve                     */

  lghtcrv->n = getint( fp);
  gettstr( fp, lghtcrv->name);
  readparam( fp, &lghtcrv->cal);

  /*  Read the relative weight for this lightcurve, compute degrees
      of freedom contributed to this dataset, and compute weight sum  */

  lghtcrv->weight = getdouble( fp);
  lghtcrv->dof = lghtcrv->weight * lghtcrv->n;
  *chi2_variance = 2 * lghtcrv->weight * lghtcrv->weight * lghtcrv->n;
  if (lghtcrv->cal.state == 'c')
    lghtcrv->sum_opt_brightness_weights = lghtcrv->weight * lghtcrv->n;
  else
	  lghtcrv->sum_opt_brightness_weights = 0.0;

  /*  If this node handles this dataset, allocate
      memory for quantities related to observation epochs  */

  lghtcrv->t0 = vector( 1, lghtcrv->n);
  lghtcrv->t = matrix( 1, lghtcrv->n, 0, lghtcrv->nviews-1);
  lghtcrv->obs = vector( 1, lghtcrv->n);
  lghtcrv->fit = vector( 1, lghtcrv->n);
  lghtcrv->oneovervar = vector( 1, lghtcrv->n);
  lghtcrv->rotphase_obs = vector( 1, lghtcrv->n);


  /*  If we are setting calculation epochs equal to the unique observation
      epochs, create a vector to hold the raw observation epochs -- later
      to be sorted so that repeated values can be skipped over.             */

  if (np < 0)
    obsepoch_raw = vector( 1, lghtcrv->n);

  /*  If this node handles this dataset, OR if we are setting the
      calculation epochs equal to the unique observation epochs,
      read the datafile for this lightcurve                         */

  sunmag = (*par).sun_appmag[ lghtcrv->ioptlaw ];
  FOPEN( fin, lghtcrv->name, "r");
  i = 0;
  while (!feof(fin) && i < lghtcrv->n) {
	  i++;

	  /*  Read a single lightcurve point: epoch (JD), magnitude, rms error  */

	  obsepoch = getdouble( fin);
	  obsmag = getdouble( fin);
	  obsmagerr = getdouble( fin);

	  /*  Convert from magnitude to intensity (relative to solar intensity)  */

	  obsintens = exp( -0.4 * LN10 * (obsmag - sunmag) );
	  obsintenserr = (0.4 * LN10 * obsmagerr) * obsintens;

	  /*  Build up the vector containing the epochs (later to be sorted)  */

	  if (np < 0)
		  obsepoch_raw[i] = obsepoch;

	  /*  If this is the node which handles this dataset, get the
          necessary observation-related quantities for this point  */
	  lghtcrv->t0[i] = obsepoch;
	  lghtcrv->obs[i] = obsintens;
	  lghtcrv->oneovervar[i] = 1.0/(obsintenserr*obsintenserr);   /* 1/variance */

	  /*  Loop through all views contributing to this (smeared) observed point  */

	  for (k=0; k<lghtcrv->nviews; k++) {

		  /*  Compute the epoch of this view, uncorrected for light-travel time  */

		  lghtcrv->t[i][k] = lghtcrv->t0[i]
		                                     + (k - lghtcrv->v0)*lghtcrv->view_interval;

		  /*  Correct for one-way light-travel time if desired  */

		  if ((*par).perform_ltc) {
			  dist = ephem2mat( lghtcrv->astephem, lghtcrv->solephem,
					  lghtcrv->t[i][k],
					  oe, se, orbspin, &solar_phase, &solar_azimuth, 1);
			  lghtcrv->t[i][k] -= DAYSPERAU*dist;
		  }
	  }

  }
  if (i != lghtcrv->n) {
	  printf("ERROR: fix obs file: %d lightcurve pts, not %d, were read for dataset %d\n",
			  i, lghtcrv->n, s);
	  bailout("read_lghtcrv in read_dat.c\n");
  } else if (!nomoredata( fin)) {
	  printf("ERROR: fix obs file: > %d lightcurve pts were read for dataset %d\n",
			  lghtcrv->n, s);
	  bailout("read_lghtcrv in read_dat.c\n");
  }
  fclose( fin);


  /*  Sort the observation epochs, count how many unique observation epochs
      there are, and create a vector containing only these sorted, unique epochs  */

  if (np < 0) {
    hpsort( lghtcrv->n, obsepoch_raw);
    nunique = 1;
    for (i=2; i<=lghtcrv->n; i++)
      if (obsepoch_raw[i] > obsepoch_raw[i-1])
        nunique++;
    obsepoch_unique = vector( 1, nunique);
    obsepoch_unique[1] = obsepoch_raw[1];
    k = 1;
    for (i=2; i<=lghtcrv->n; i++)
      if (obsepoch_raw[i] > obsepoch_raw[i-1])
        obsepoch_unique[++k] = obsepoch_raw[i];
    free_vector( obsepoch_raw, 1, lghtcrv->n);

    /*  We now know how many unique observation epochs there are, so we can
        allocate memory for the various calculation-related arrays/vectors,
        assign epochs, and (if specified) correct them for light-travel time.  */

    lghtcrv->ncalc = nunique;
    lghtcrv->x0 = vector( 1, lghtcrv->ncalc);
    lghtcrv->x = vector( 1, lghtcrv->ncalc);
    lghtcrv->y = vector( 1, lghtcrv->ncalc);
    lghtcrv->y2 = vector( 1, lghtcrv->ncalc);
    lghtcrv->rotphase_calc = vector( 1, lghtcrv->ncalc);
    lghtcrv->solar_phase = vector( 1, lghtcrv->ncalc);
    lghtcrv->solar_azimuth = vector( 1, lghtcrv->ncalc);
    lghtcrv->rend = (struct crvrend_t *) calloc( lghtcrv->ncalc+1,
                                                   sizeof( struct crvrend_t));
    for (i=1; i<=lghtcrv->ncalc; i++) {
      lghtcrv->x[i] = lghtcrv->x0[i] = obsepoch_unique[i];
      dist = ephem2mat( lghtcrv->astephem, lghtcrv->solephem,
                        lghtcrv->x0[i],
                        lghtcrv->rend[i].oe, lghtcrv->rend[i].se,
                        lghtcrv->rend[i].orbspin,
                        &lghtcrv->solar_phase[i], &lghtcrv->solar_azimuth[i], 1);
      if ((*par).perform_ltc) {                        /* apply light-time correction */
        lghtcrv->x[i] = lghtcrv->x0[i] - DAYSPERAU*dist;
        ephem2mat( lghtcrv->astephem, lghtcrv->solephem,
                   lghtcrv->x[i],
                   lghtcrv->rend[i].oe, lghtcrv->rend[i].se,
                   lghtcrv->rend[i].orbspin,
                   &lghtcrv->solar_phase[i], &lghtcrv->solar_azimuth[i], 1);
      }
    }
    free_vector( obsepoch_unique, 1, lghtcrv->n);
  }
  printf("#     %s\n", lghtcrv->name);
  fflush(stdout);

  /*  Initialize quantities related to spin impulses  */

  for (i=1; i<=lghtcrv->ncalc; i++) {
    lghtcrv->rend[i].n_integrate = -999;
    for (n=0; n<MAXIMP+2; n++) {
      lghtcrv->rend[i].t_integrate[n] = -HUGENUMBER;
      for (j=0; j<=2; j++)
        lghtcrv->rend[i].impulse[n][j] = 0.0;
    }
  }

  /*  If this node handles this dataset, check if the interpolations needed to obtain
      observed lightcurve points from calculated lightcurve points (in the "calc_fits"
      routine) will actually involve extrapolations beyond the calculated points, and if
      so, give a warning.  This problem is most likely to arise when modeling smearing.   */

  extrapolate_flag = 0;
  for (i=1; i<=lghtcrv->n; i++)
	  for (k=0; k<lghtcrv->nviews; k++)
		  if (lghtcrv->t[i][k] < lghtcrv->x[1]
		                                        || lghtcrv->t[i][k] > lghtcrv->x[lghtcrv->ncalc])
			  extrapolate_flag = 1;
  if (extrapolate_flag) {
	  fprintf(stderr,"\n");
	  fprintf(stderr,"WARNING for dataset %2d:\n", s);
	  fprintf(stderr,"            observed lightcurve intensities will be extrapolated\n");
	  fprintf(stderr,"            outside the time span of calculated intensities\n");
	  fprintf(stderr,"\n");
  }

  return npar;
}

void read_deldop_ascii(FILE *fin, struct deldop_t *deldop, int iframe,
                                  int idel_use[2], int idop_use[2])
{
  int j, k, jskip, kskip;
  float pixelvalue;

  jskip = idel_use[0] - 1;
  kskip = idop_use[0] - 1;

  for (j=1; j<=(*deldop).ndel; j++)
    for (k=1; k<=(*deldop).ndop; k++) {
      pixelvalue = getdouble(fin);
      if (j >= idel_use[0] && j <= idel_use[1] && k >= idop_use[0] && k <= idop_use[1])
        (*deldop).frame[iframe].obs[j-jskip][k-kskip] =
                                 pixelvalue * (*deldop).frame[iframe].sdev;
    }
}

void read_deldop_binary(FILE *fin, struct deldop_t *deldop, int iframe,
                                   int idel_use[2], int idop_use[2], int swap_bytes)
{
  int j, k, jskip, kskip;
  float pixelvalue;

  jskip = idel_use[0] - 1;
  kskip = idop_use[0] - 1;

  for (j=1; j<=(*deldop).ndel; j++)
    for (k=1; k<=(*deldop).ndop; k++) {
      if (fread(&pixelvalue, sizeof(float), 1, fin) != 1)
        bailout("read_binary_dd in read_dat.c: error reading binary file\n");

      if (j >= idel_use[0] && j <= idel_use[1] && k >= idop_use[0] && k <= idop_use[1]) {
        if (swap_bytes)
          (*deldop).frame[iframe].obs[j-jskip][k-kskip] =
                                   swap_float(pixelvalue) * (*deldop).frame[iframe].sdev;
        else
          (*deldop).frame[iframe].obs[j-jskip][k-kskip] =
                                   pixelvalue * (*deldop).frame[iframe].sdev;
      }
    }
}

void read_deldop_rdf(FILE *fin, struct deldop_t *deldop, int iframe,
                                int idel_use[2], int idop_use[2], int swap_bytes)
{
  int j, k, jskip, kskip;
  float pixelvalue;
  char buffer[200] = "RDF header";

  while (buffer[0] != '.')
    if (!fgets(buffer, 200, fin))
      bailout("read_dat.c: can't read RDF data file\n");

  jskip = idel_use[0] - 1;
  kskip = idop_use[0] - 1;

  for (j=1; j<=(*deldop).ndel; j++)
    for (k=1; k<=(*deldop).ndop; k++) {
      if (fread(&pixelvalue, sizeof(float), 1, fin) != 1)
        bailout("read_dat.c: error reading RDF file\n");

      if (j >= idel_use[0] && j <= idel_use[1] && k >= idop_use[0] && k <= idop_use[1]) {
        if (swap_bytes)
          (*deldop).frame[iframe].obs[j-jskip][k-kskip] =
                                   swap_float(pixelvalue) * (*deldop).frame[iframe].sdev;
        else
          (*deldop).frame[iframe].obs[j-jskip][k-kskip] =
                                   pixelvalue * (*deldop).frame[iframe].sdev;
      }
    }
}

void read_deldop_fits(char *filename, struct deldop_t *deldop, int iframe,
                                      int idel_use[2], int idop_use[2])
{
  /* shamelessly adapted from cfitsio's readimage() in cookbook.c */
  fitsfile *fptr;
  int j, k, jskip, kskip, status,  nfound, anynull;
  long naxis, naxes[2];
  long fpixel = 1;
  long nbuffer = 1;
  float nullval = 0;
  float pixelvalue;

  status = 0;

  if ( fits_open_file(&fptr, filename, READONLY, &status) )
    fits_report_error(stderr, status);

  if ( fits_read_key_lng(fptr, "NAXIS", &naxis, NULL, &status) )
    fits_report_error(stderr, status);

  if (naxis != 2)
    bailout("read_deldop_fits in read_dat.c: FITS file is not a 2-D image\n");

  if ( fits_read_keys_lng(fptr, "NAXIS", 1, 2, naxes, &nfound, &status) )
    fits_report_error(stderr, status);

  if (naxes[0] != (*deldop).ndop || naxes[1] != (*deldop).ndel)
    fprintf(stderr,"WARNING: discrepancy bw sizes in FITS header and obs file.\n");

  jskip = idel_use[0] - 1;
  kskip = idop_use[0] - 1;

  for (j=1; j<=(*deldop).ndel; j++)
    for (k=1; k<=(*deldop).ndop; k++) {
      if ( fits_read_img(fptr, TFLOAT, fpixel++, nbuffer, &nullval,
                         &pixelvalue, &anynull, &status) )
        fits_report_error(stderr, status);
      if (j >= idel_use[0] && j <= idel_use[1] && k >= idop_use[0] && k <= idop_use[1])
        (*deldop).frame[iframe].obs[j-jskip][k-kskip] =
                                 pixelvalue * (*deldop).frame[iframe].sdev;
    }

  if ( fits_close_file(fptr, &status) )
    fits_report_error(stderr, status);
}

void read_doppler_ascii(FILE *fin, struct doppler_t *doppler, int iframe, int idop_use[2])
{
  int k, kskip;
  float spectralvalue;

  kskip = idop_use[0] - 1;

  for (k=1; k<=(*doppler).ndop; k++) {
    spectralvalue = getdouble(fin);
    if (k >= idop_use[0] && k <= idop_use[1])
      (*doppler).frame[iframe].obs[k-kskip] = spectralvalue * (*doppler).frame[iframe].sdev;
  }
}

void read_doppler_binary(FILE *fin, struct doppler_t *doppler, int iframe, int idop_use[2],
                                    int swap_bytes)
{
  int k, kskip;
  float spectralvalue;

  kskip = idop_use[0] - 1;

  for (k=1; k<=(*doppler).ndop; k++) {
    if (fread(&spectralvalue, sizeof(float), 1, fin) != 1)
      bailout("read_binary_dd in read_dat.c: error reading binary file\n");

    if (k >= idop_use[0] && k <= idop_use[1]) {
      if (swap_bytes)
        (*doppler).frame[iframe].obs[k-kskip] =
                                  swap_float(spectralvalue) * (*doppler).frame[iframe].sdev;
      else
        (*doppler).frame[iframe].obs[k-kskip] =
                                  spectralvalue * (*doppler).frame[iframe].sdev;
    }
  }
}

void read_doppler_rdf(FILE *fin, struct doppler_t *doppler, int iframe, int idop_use[2],
                                 int swap_bytes)
{
  int k, kskip;
  float spectralvalue;
  char buffer[200] = "RDF header";

  while (buffer[0] != '.')
    if (!fgets(buffer, 200, fin))
      bailout("read_doppler_rdf in read_dat.c: can't read RDF data file\n");

  kskip = idop_use[0] - 1;

  for (k=1; k<=(*doppler).ndop; k++) {
    if (fread(&spectralvalue, sizeof(float), 1, fin) != 1)
      bailout("read_doppler_rdf in read_dat.c: error reading RDF file\n");

    if (k >= idop_use[0] && k <= idop_use[1]) {
      if (swap_bytes)
        (*doppler).frame[iframe].obs[k-kskip] =
                                  swap_float(spectralvalue) * (*doppler).frame[iframe].sdev;
      else
        (*doppler).frame[iframe].obs[k-kskip] =
                                  spectralvalue * (*doppler).frame[iframe].sdev;
    }
  }
}

void read_doppler_fits(char *filename, struct doppler_t *doppler, int iframe, int idop_use[2])
{
  /* shamelessly adapted from cfitsio's readimage() in cookbook.c */
  fitsfile *fptr;
  int k, kskip, status, anynull;
  long naxis, naxes;
  long fspecbin = 1;
  long nbuffer = 1;
  float nullval = 0;
  float spectralvalue;

  status = 0;

  if ( fits_open_file(&fptr, filename, READONLY, &status) )
    fits_report_error(stderr, status);

  if ( fits_read_key_lng(fptr, "NAXIS", &naxis, NULL, &status) )
    fits_report_error(stderr, status);

  if (naxis != 1)
    bailout("read_doppler_fits in read_dat.c: FITS file is not a 1-D spectrum\n");

  if ( fits_read_key_lng(fptr, "NAXIS1", &naxes, NULL, &status) )
    fits_report_error(stderr, status);

  if (naxes != (*doppler).ndop)
    fprintf(stderr,"WARNING: discrepancy bw sizes in FITS header and obs file.\n");

  kskip = idop_use[0] - 1;

  for (k=1; k<=(*doppler).ndop; k++) {
    if ( fits_read_img(fptr, TFLOAT, fspecbin++, nbuffer, &nullval,
                       &spectralvalue, &anynull, &status) )
      fits_report_error(stderr, status);
    if (k >= idop_use[0] && k <= idop_use[1])
      (*doppler).frame[iframe].obs[k-kskip] = spectralvalue * (*doppler).frame[iframe].sdev;
  }

  if ( fits_close_file(fptr, &status) )
    fits_report_error(stderr, status);
}

void read_poset_fits(char *filename, struct poset_t *poset, int iframe,
                                     int icol_use[2], int irow_use[2], int read_data)
{

  /*  This routine is somewhat convoluted, since read_poset can't read the
      pixel-weighting mask (if any) to vignette the image until it has read
      the image to figure out the raw dimensions, but it can't read the image
      until it's read the mask and is ready to vignette the image.

      Solution: read_poset calls read_poset_fits TWICE per plane-of-sky frame.
      First it calls it with read_data = 0, so that the image dimensions are
      read but not the pixel values.  Then it reads the mask (so that it can
      get the vignetted limits irow_use and icol_use), and finally it calls
      read_poset_fits a second time with read_data = 1 so that the pixel
      values can be read in.                                                    */

  fitsfile *fptr;
  int j, k, jskip=0, kskip=0, status,  nfound, anynull;
  long naxis, naxes[2];
  long fpixel = 1;
  long nbuffer = 1;
  float nullval = 0;
  float pixelvalue;

  status = 0;

  if ( fits_open_file(&fptr, filename, READONLY, &status) )
    fits_report_error(stderr, status);

  /*  Check that this is a two-dimensional image  */

  if ( fits_read_key_lng(fptr, "NAXIS", &naxis, NULL, &status) )
    fits_report_error(stderr, status);

  if (naxis != 2)
    bailout("read_deldop_fits in read_dat.c: FITS file is not a 2-D image\n");

  /*  Get the raw image dimensions (i.e., before vignetting)  */

  if ( fits_read_keys_lng(fptr, "NAXIS", 1, 2, naxes, &nfound, &status) )
    fits_report_error(stderr, status);

  /*  If the data (pixel values) will be read, read them and vignette the image  */

  if (read_data) {
      jskip = icol_use[0] - 1;
      kskip = irow_use[0] - 1;

      /*  Read pixels one at a time: each column within the first row, then each
          column within the second row, etc.

          NOTE that we will store this array with the first index representing
          columns and the second index rows, because that's how shape's plane-of-sky
          renderings are set up: shape needs to resample POS renderings to get
          plane-of-sky fit images, and it can handle rotations of +/- northangle
          when it resamples but it can't handle axis flipping.                        */

      for (k=1; k<=naxes[1]; k++)
        for (j=1; j<=naxes[0]; j++) {
          if ( fits_read_img(fptr, TFLOAT, fpixel++, nbuffer, &nullval,
                             &pixelvalue, &anynull, &status) )
            fits_report_error(stderr, status);
          if (j >= icol_use[0] && j <= icol_use[1] && k >= irow_use[0] && k <= irow_use[1])
            (*poset).frame[iframe].obs.b[j-jskip][k-kskip] = pixelvalue;
        }
  } else {
      (*poset).frame[iframe].ncol = naxes[0];   /* read_poset may change this later */
      (*poset).frame[iframe].nrow = naxes[1];   /* read_poset may change this later */
  }

  if ( fits_close_file(fptr, &status) )
    fits_report_error(stderr, status);
}

void initialize_delcor( struct dat_t *dat)
{
  int nmax, s, i, s_delcor, n_delcor, n;
  double t0_delcor, t0;
  struct deldop_t *deldop;
  struct doppler_t *doppler;

  nmax = -1;
  for (s=0; s<(*dat).nsets; s++)
    if ((*dat).set[s].type == DELAY)
      nmax = MAX( nmax, (*dat).set[s].desc.deldop.delcor.n);
    if ((*dat).set[s].type == DOPPLER)
      nmax = MAX( nmax, (*dat).set[s].desc.doppler.delcor.n);
  if (nmax < 0)
    return;

  for (i=0; i<=nmax; i++) {
    s_delcor = -1;
    n_delcor = -1;
    t0_delcor = -1.0;
    for (s=0; s<(*dat).nsets; s++)
      if ((*dat).set[s].type == DELAY) {
          deldop = &(*dat).set[s].desc.deldop;
          n = (*deldop).delcor.n;
          t0 = (*deldop).delcor.t0;
          if ((*deldop).delcor.a[i].state == '=') {
              if (s_delcor < 0)
                bailout("can't use \"=\" state for the first delay polynomial\n");
              else if (n != n_delcor)
                bailout("delay polynomials must have same degree if state = \"=\"\n");
              else if (fabs(t0 - t0_delcor) > HALFSECOND)
                bailout("delay polynomials must have same t0 if state = \"=\"\n");
          } else {
              s_delcor = s;
              n_delcor = n;
              t0_delcor = t0;
          }
          (*deldop).delcor.equals_set[i] = s_delcor;

      } else if ((*dat).set[s].type == DOPPLER) {
          doppler = &(*dat).set[s].desc.doppler;
          n = (*doppler).delcor.n;
          t0 = (*doppler).delcor.t0;
          if ((*doppler).delcor.a[i].state == '=') {
              if (s_delcor < 0)
                bailout("can't use \"=\" state for the first delay polynomial\n");
              else if (n != n_delcor)
                bailout("delay polynomials must have same degree if state = \"=\"\n");
              else if (fabs(t0 - t0_delcor) > HALFSECOND)
                bailout("delay polynomials must have same t0 if state = \"=\"\n");
          } else {
              s_delcor = s;
              n_delcor = n;
              t0_delcor = t0;
          }
          (*doppler).delcor.equals_set[i] = s_delcor;
      }
  }
}
