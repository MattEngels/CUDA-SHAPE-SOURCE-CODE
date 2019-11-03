/*****************************************************************************************
                                                                              write_dat.c

Writes observations (obs) file.

Modified 2015 June 10 by CM:
    Implement smearing for the "fit" and "write" actions by writing number of views per
        frame (or lightcurve point), view interval, and smearing mode for all datasets

Modified 2014 February 12 by CM:
    Implement multiple radar and optical scattering laws

Modified 2012 March 24 by CM:
    In write_deldop and write_doppler, write "dopscale" parameter
    Call "realize_dopscale" routine in case Doppler scaling factors with the '=' state
        need to be updated

Modified 2009 March 24 by CM:
    In write_lghtcrv, write the list of calculated lightcurve points
       exactly as it was specified in the input obs file rather than always
       changing it to an explicit list of Julian dates

Modified 2007 August 18 by CM:
    Don't include build.h (which is now included in head.h)

Modified 2006 October 1 by CM:
    Add two new arguments to realize_delcor
    Write lightcurve "calfact" as %12.6e rather than %f format
        (since it's now multiplicative rather than additive)

Modified 2006 June 21 by CM:
    In write_deldop, changed delres to del_per_pixel and dopres to
        dop_per_pixel, and improved comments
    In write_doppler, changed dopres to dop_per_bin and improved comments
    In write_poset, changed angres to angle_per_pixel and improved comments

Modified 2006 June 18 by CM:
    Write "mask" pixel-weighting flag for all delay-Doppler, Doppler, and
        plane-of-sky frames
    Fix error in write_poset: solar ephemeris wasn't being written to
        increased precision (see 2006 April 14 modification below)
    Eliminate range datasets

Modified 2006 April 14 by PT:
    Increase precision when writing RA, dec, and distance to 5, 5, and 8
        decimal places to prevent discontinuities in the values of 
        apparent rotation due to sky motion

Modified 2005 November 18 by MCN:
    Change nlooks (# of looks) from integer to floating-point, to handle
        zero-filled data

Modified 2005 July 22 by CM:
    Don't change the values of the mpi processing nodes when rewriting the
        obs file for any action other than "fit"

Modified 2005 July 17 by CM:
    COM delay bin is now written in %13.6f format rather than %11.6f,
        thus allowing for very large positive or negative values

Modified 2005 March 10 by CM:
    Weights are now floating-point rather than integer

Modified 2005 March 6 by CM:
    For plane-of-sky datasets, don't write "calfact" to the obs file

Modified 2005 March 1 by CM:
    For plane-of-sky datasets, eliminate "sdev" and add "northangle"
        for each frame
    Add call to new realize_xyoff routine

Modified 2005 January 25 by CM:
    Eliminate unused variables

Modified 2005 January 21 by CM:
    Always write the mpi processing node: if read_node = 0, write -1
        as the node, thus telling shape to assign the node sequentially
        if it reads this file on a subsequent run
    Slight adjustments to output format for POS datasets
    Convert POS angular resolution from radians to arcseconds on output

Modified 2004 March 15 by CM:
    Slight adjustment to output format for delay correction
        polynomial coefficients

Modified 2004 February 29 by CM:
    Label lightcurve calculation epochs 0 through (ncalc-1)
        rather than 1 through ncalc, to be consistent with
        Doppler and delay-Doppler usage

Modified 2004 February 16 by CM:
    Include calls to realize_angleoff and realize_omegaoff in case
        angle offset and spin offset components with state = '='
        need to be updated.
    Increase maximum length of datafile names to 25 characters
        for delay-Doppler, Doppler, POS, and range datasets

Modified 2003 October 27 by CM:
    Read version number from build.h

Modified 2003 April 24 by CM:
    Include a call to realize_delcor in case delay polynomial
        coefficients with state = '=' need to be updated.
    Move "delcom" from delay-Doppler datasets to individual frames
    Add "weight" to delay-Doppler, Doppler, POS, and range frames
        and to lightcurve datasets
    Fix a typo at the end of write_poset
*****************************************************************************************/

#include "head.h"

void write_deldop( FILE *fp, struct deldop_t *deldop);
void write_doppler( FILE *fp, struct doppler_t *doppler);
void write_poset( FILE *fp, struct poset_t *poset);
void write_lghtcrv( FILE *fp, struct lghtcrv_t *lghtcrv);

void write_dat( struct par_t *par, struct dat_t *dat)
{
  FILE *fp;
  int s, output_node;

  /*  Make sure that obs file parameters with state = '=' are updated  */
  realize_angleoff( dat);
  realize_omegaoff( dat);
  realize_delcor( dat, 0.0, 0);
  realize_dopscale( par, dat, 1.0, 0);
  realize_xyoff( dat);

  /*  Now write the data  */
  FOPEN( fp, dat->name, "w");
  printf("# writing data to file: %s ...\n", dat->name);
  fprintf( fp, "{DATA FILE FOR SHAPE.C VERSION %s BUILD %s}\n\n", VERSION, BUILD);
  fprintf( fp, "%14d {number of sets}\n", dat->nsets);
  for (s=0; s<dat->nsets; s++) {
    fprintf( fp, "\n\n{SET %d}\n\n", s);
    if (par->action != FIT || (HMT))
      output_node = dat->set[s].inputnode;
    else
      output_node = -1;
    fprintf( fp, "%2d {is mpi node responsible for this set}\n\n", output_node);
    fprintf( fp, " %c %14e %c %14e %c %14e {Euler angle offsets}\n",
                 dat->set[s].angleoff[0].state, dat->set[s].angleoff[0].val*R2D,
                 dat->set[s].angleoff[1].state, dat->set[s].angleoff[1].val*R2D,
                 dat->set[s].angleoff[2].state, dat->set[s].angleoff[2].val*R2D);
    fprintf( fp, " %c %14e %c %14e %c %14e {spin vector offsets}\n",
                 dat->set[s].omegaoff[0].state, dat->set[s].omegaoff[0].val*R2D,
                 dat->set[s].omegaoff[1].state, dat->set[s].omegaoff[1].val*R2D,
                 dat->set[s].omegaoff[2].state, dat->set[s].omegaoff[2].val*R2D);
    switch (dat->set[s].type) {
    case DELAY:
        write_deldop( fp, &dat->set[s].desc.deldop);
        break;
    case DOPPLER:
        write_doppler( fp, &dat->set[s].desc.doppler);
        break;
    case POS:
        write_poset( fp, &dat->set[s].desc.poset);
        break;
    case LGHTCRV:
        write_lghtcrv( fp, &dat->set[s].desc.lghtcrv);
        break;
    default:
        bailout("write_dat.c: can't write that data type yet\n");
    }
  }
  fclose( fp);
  printf("# writing completed\n");
  fflush(stdout);
}


void write_deldop( FILE *fp, struct deldop_t *deldop)
{
  int i, cd[6];
  char codestring[3][10] = {"short", "long_orig", "long_mod"};
  char smearingstring[2][7] = {"center", "first"};

  fprintf( fp, "\n%14s {set type}\n", "delay-doppler");
  fprintf( fp, "\n%14d {radar scattering law for this set}\n", deldop->iradlaw);
  fprintf( fp, "\n%14d {number of ephemeris points}\n", deldop->astephem.n);
  fprintf( fp, "{%3s %2s %2s %2s %2s %2s %9s %9s %9s}\n", "yr", "mo",
               "dd", "hh", "mm", "ss", "ra", "dec", "dist");
  for (i=0; i<deldop->astephem.n; i++) {
    jd2cal( &cd[0], &cd[1], &cd[2], &cd[3], &cd[4], &cd[5],
            deldop->astephem.pnt[i].t);
    fprintf( fp, "%4d %2d %2d %2d %2d %2d %9.5f %9.5f %10.8f\n",
                 cd[0], cd[1], cd[2], cd[3], cd[4], cd[5],
                 deldop->astephem.pnt[i].ra*R2D, deldop->astephem.pnt[i].dec*R2D,
                 deldop->astephem.pnt[i].dist);
  }
  fprintf( fp, "\n%14.6f {transmitter frequency (MHz)}\n", deldop->Ftx);
  fprintf( fp, "\n%d %f %d %d %s {delay: # rows, pixel height (usec), spb, stride, code method}\n",
               deldop->ndel, deldop->del_per_pixel, deldop->spb, deldop->stride,
               codestring[deldop->codemethod]);
  fprintf( fp, "\n%d %f %f %f %d {dop: # cols, pixel width (Hz), COM col, DC col, fftlen}\n\n",
               deldop->ndop, deldop->dop_per_pixel, deldop->dopcom,
               deldop->dopDC, deldop->dopfftlen);

  jd2cal( &cd[0], &cd[1], &cd[2], &cd[3], &cd[4], &cd[5],
          deldop->delcor.t0);
  fprintf( fp, "%4d %2d %2d %2d %2d %2d {t0 of delcor poly}\n",
               cd[0], cd[1], cd[2], cd[3], cd[4], cd[5]);
  fprintf( fp, "%14d {order of polynomial}\n", deldop->delcor.n);
  for (i=0; i<=deldop->delcor.n; i++)
    fprintf( fp, " %c %13.6e {coefficient %d}\n",
                 deldop->delcor.a[i].state, deldop->delcor.a[i].val, i);
  fprintf( fp, "\n %c %13.6e {Doppler scaling factor}\n",
               deldop->dopscale.state, deldop->dopscale.val);
  fprintf( fp, "\n%d %8.3f %s {smearing: # views per frame, view interval (s), mode}\n",
               deldop->nviews, deldop->view_interval*86400,
               smearingstring[deldop->smearing_mode]);
  fprintf( fp, "\n%14s {data directory}\n\n", deldop->dir);
  fprintf( fp, "%14d {number of frames}\n", deldop->nframes);
  fprintf( fp, "{%24s %4s %2s %2s %2s %2s %2s %12s %14s %7s %13s %12s %4s}\n",
               "name", "year", "mo", "dd", "hh", "mm", "ss", "sdev",
               "calfact", "looks", "COM del row", "weight", "mask");
  for (i=0; i<deldop->nframes; i++) {
    jd2cal( &cd[0], &cd[1], &cd[2], &cd[3], &cd[4], &cd[5],
            deldop->frame[i].t0);
    fprintf( fp, "%25s %4d %2d %2d %2d %2d %2d %12.6e %c %12.6e %7.1f %13.6f %12.6e %4d\n",
                 deldop->frame[i].name,
                 cd[0], cd[1], cd[2], cd[3], cd[4], cd[5], 
                 deldop->frame[i].sdev, deldop->frame[i].cal.state,
                 deldop->frame[i].cal.val, deldop->frame[i].nlooks,
                 deldop->frame[i].delcom, deldop->frame[i].weight,
                 deldop->frame[i].pixels_weighted);
  }
}


void write_doppler( FILE *fp, struct doppler_t *doppler)
{
  int i, cd[6];
  char smearingstring[2][7] = {"center", "first"};

  fprintf( fp, "\n%14s {set type}\n", "doppler");
  fprintf( fp, "\n%14d {radar scattering law for this set}\n", doppler->iradlaw);
  fprintf( fp, "\n%14d {number of ephemeris points}\n", doppler->astephem.n);
  fprintf( fp, "{%3s %2s %2s %2s %2s %2s %9s %9s %9s}\n", "yr", "mo",
               "dd", "hh", "mm", "ss", "ra", "dec", "dist");
  for (i=0; i<doppler->astephem.n; i++) {
    jd2cal( &cd[0], &cd[1], &cd[2], &cd[3], &cd[4], &cd[5],
            doppler->astephem.pnt[i].t);
    fprintf( fp, "%4d %2d %2d %2d %2d %2d %9.5f %9.5f %10.8f\n",
                 cd[0], cd[1], cd[2], cd[3], cd[4], cd[5],
                 doppler->astephem.pnt[i].ra*R2D,
                 doppler->astephem.pnt[i].dec*R2D,
                 doppler->astephem.pnt[i].dist);
  }
  fprintf( fp, "\n%14.6f {transmitter frequency (MHz)}\n", doppler->Ftx);
  fprintf( fp, "\n%d %f %f {dop: # bins, bin width (Hz), COM bin}\n\n",
               doppler->ndop, doppler->dop_per_bin, doppler->dopcom);

  jd2cal( &cd[0], &cd[1], &cd[2], &cd[3], &cd[4], &cd[5],
          doppler->delcor.t0);
  fprintf( fp, "%4d %2d %2d %2d %2d %2d {t0 of delcor poly}\n",
               cd[0], cd[1], cd[2], cd[3], cd[4], cd[5]);
  fprintf( fp, "%14d {order of polynomial}\n", doppler->delcor.n);
  for (i=0; i<=doppler->delcor.n; i++)
    fprintf( fp, " %c %13.6e {coefficient %d}\n",
                 doppler->delcor.a[i].state, doppler->delcor.a[i].val, i);
  fprintf( fp, "\n %c %13.6e {Doppler scaling factor}\n",
               doppler->dopscale.state, doppler->dopscale.val);
  fprintf( fp, "\n%d %8.3f %s {smearing: # views per frame, view interval (s), mode}\n",
               doppler->nviews, doppler->view_interval*86400,
               smearingstring[doppler->smearing_mode]);
  fprintf( fp, "\n%14s {data directory}\n\n", doppler->dir);
  fprintf( fp, "%14d {number of frames}\n", doppler->nframes);
  fprintf( fp, "{%24s %4s %2s %2s %2s %2s %2s %12s %14s %7s %12s %4s}\n",
               "name", "year", "mo", "dd", "hh", "mm", "ss", "sdev",
               "calfact", "looks", "weight", "mask");
  for (i=0; i<doppler->nframes; i++) {
    jd2cal( &cd[0], &cd[1], &cd[2], &cd[3], &cd[4], &cd[5],
            doppler->frame[i].t0);
    fprintf( fp, "%25s %4d %2d %2d %2d %2d %2d %12.6e %c %12.6e %7.1f %12.6e %4d\n",
                 doppler->frame[i].name,
                 cd[0], cd[1], cd[2], cd[3], cd[4], cd[5], 
                 doppler->frame[i].sdev, doppler->frame[i].cal.state,
                 doppler->frame[i].cal.val, doppler->frame[i].nlooks,
                 doppler->frame[i].weight, doppler->frame[i].pixels_weighted);
  }
}


void write_poset( FILE *fp, struct poset_t *poset)
{
  int i, cd[6];
  char smearingstring[2][7] = {"center", "first"};

  fprintf( fp, "\n%14s {set type}\n", "plane-of-sky");
  fprintf( fp, "\n%14d {optical scattering law for this set}\n", poset->ioptlaw);
  fprintf( fp, "\n%14d {number of asteroid ephemeris points}\n",
               poset->astephem.n);
  fprintf( fp, "{%3s %2s %2s %2s %2s %2s %9s %9s %9s}\n", "yr", "mo",
               "dd", "hh", "mm", "ss", "ra", "dec", "dist");
  for (i=0; i<poset->astephem.n; i++) {
    jd2cal( &cd[0], &cd[1], &cd[2], &cd[3], &cd[4], &cd[5],
            poset->astephem.pnt[i].t);
    fprintf( fp, "%4d %2d %2d %2d %2d %2d %9.5f %9.5f %10.8f\n",
                 cd[0], cd[1], cd[2], cd[3], cd[4], cd[5],
                 poset->astephem.pnt[i].ra*R2D, poset->astephem.pnt[i].dec*R2D,
                 poset->astephem.pnt[i].dist);
  }
  fprintf( fp, "\n%14d {number of solar ephemeris points}\n",
               poset->solephem.n);
  fprintf( fp, "{%3s %2s %2s %2s %2s %2s %6s %6s %7s}\n", "yr", "mo",
               "dd", "hh", "mm", "ss", "ra", "dec", "dist");
  for (i=0; i<poset->solephem.n; i++) {
    jd2cal( &cd[0], &cd[1], &cd[2], &cd[3], &cd[4], &cd[5],
            poset->solephem.pnt[i].t);
    fprintf( fp, "%4d %2d %2d %2d %2d %2d %9.5f %9.5f %10.8f\n",
                 cd[0], cd[1], cd[2], cd[3], cd[4], cd[5],
                 poset->solephem.pnt[i].ra*R2D, poset->solephem.pnt[i].dec*R2D,
                 poset->solephem.pnt[i].dist);
  }
  fprintf( fp, "\n%14.6f {angular pixel size (arcsec)}\n", poset->angle_per_pixel*R2D*3600);
  fprintf( fp, "\n%d %8.3f %s {smearing: # views per frame, view interval (s), mode}\n",
               poset->nviews, poset->view_interval*86400,
               smearingstring[poset->smearing_mode]);
  fprintf( fp, "\n%14s {data directory}\n\n", poset->dir);
  fprintf( fp, "%14d {number of frames}\n", poset->nframes);
  fprintf( fp, "{%24s %4s %2s %2s %2s %2s %2s %13s %15s %15s %12s %4s}\n",
               "name", "year", "mo", "dd", "hh", "mm", "ss", "northangle",
               "xoffset", "yoffset", "weight", "mask");
  for (i=0; i<poset->nframes; i++) {
    jd2cal( &cd[0], &cd[1], &cd[2], &cd[3], &cd[4], &cd[5],
            poset->frame[i].t0);
    fprintf( fp, "%25s %4d %2d %2d %2d %2d %2d %13.6e %c %13.6e %c %13.6e %12.6e %4d\n",
                 poset->frame[i].name,
                 cd[0], cd[1], cd[2], cd[3], cd[4], cd[5], 
                 poset->frame[i].northangle*R2D,
                 poset->frame[i].off[0].state, poset->frame[i].off[0].val,
                 poset->frame[i].off[1].state, poset->frame[i].off[1].val,
                 poset->frame[i].weight, poset->frame[i].pixels_weighted);
  }
}


void write_lghtcrv( FILE *fp, struct lghtcrv_t *lghtcrv)
{
  int i, cd[6];
  char smearingstring[2][7] = {"center", "first"};

  fprintf( fp, "\n%14s {set type}\n", "lightcurve");
  fprintf( fp, "\n%14d {optical scattering law for this set}\n", lghtcrv->ioptlaw);
  fprintf( fp, "\n%14d {number of asteroid ephemeris points}\n",
               lghtcrv->astephem.n);
  fprintf( fp, "{%3s %2s %2s %2s %2s %2s %9s %9s %9s}\n", "yr", "mo",
               "dd", "hh", "mm", "ss", "ra", "dec", "dist");
  for (i=0; i<lghtcrv->astephem.n; i++) {
    jd2cal( &cd[0], &cd[1], &cd[2], &cd[3], &cd[4], &cd[5],
            lghtcrv->astephem.pnt[i].t);
    fprintf( fp, "%4d %2d %2d %2d %2d %2d %9.5f %9.5f %10.8f\n",
                 cd[0], cd[1], cd[2], cd[3], cd[4], cd[5],
                 lghtcrv->astephem.pnt[i].ra*R2D,
                 lghtcrv->astephem.pnt[i].dec*R2D,
                 lghtcrv->astephem.pnt[i].dist);
  }
  fprintf( fp, "\n%14d {number of solar ephemeris points}\n",
               lghtcrv->solephem.n);
  fprintf( fp, "{%3s %2s %2s %2s %2s %2s %9s %9s %9s}\n", "yr", "mo",
               "dd", "hh", "mm", "ss", "ra", "dec", "dist");
  for (i=0; i<lghtcrv->solephem.n; i++) {
    jd2cal( &cd[0], &cd[1], &cd[2], &cd[3], &cd[4], &cd[5],
            lghtcrv->solephem.pnt[i].t);
    fprintf( fp, "%4d %2d %2d %2d %2d %2d %9.5f %9.5f %10.8f\n",
                 cd[0], cd[1], cd[2], cd[3], cd[4], cd[5],
                 lghtcrv->solephem.pnt[i].ra*R2D,
                 lghtcrv->solephem.pnt[i].dec*R2D,
                 lghtcrv->solephem.pnt[i].dist);
  }

  fprintf( fp, "\n%14d {number of calculated points}\n", lghtcrv->ncalc_obsfile);
  if (lghtcrv->ncalc_obsfile > 0) {
      for (i=1; i<=lghtcrv->ncalc; i++)
        fprintf( fp, "%f {JD[%d]}\n", lghtcrv->x0[i], i-1);
  } else if (lghtcrv->ncalc_obsfile == 0) {
        fprintf( fp, "%f %f %f {JD start, stop, interval for calculated points}\n",
                 lghtcrv->jdstart, lghtcrv->jdstop, lghtcrv->jdinterval);
  }
  fprintf( fp, "\n%d %8.3f %s {smearing: # views per point, view interval (s), mode}\n",
               lghtcrv->nviews, lghtcrv->view_interval*86400,
               smearingstring[lghtcrv->smearing_mode]);
  fprintf( fp, "\n%14d {number of samples in lightcurve}\n", lghtcrv->n);
  fprintf( fp, "%s %c %12.6e %12.6e {name, calfact, weight}\n", lghtcrv->name,
               lghtcrv->cal.state, lghtcrv->cal.val, lghtcrv->weight);
}
