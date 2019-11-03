/*****************************************************************************************
                                                                             pos2deldop.c

Takes a pos_t structure and a deldop_t structure and "fills in" a deldopfrm_t structure
indexed by frm.  In other words, pos2deldop works from a model plane-of-sky image (with an
observer z-coordinate and a scattering angle assigned to each pixel) to produce a model
delay-Doppler image corresponding to data frame frm.

In the case of an orbiting binary system (the "orbit" action), pos2deldop only computes
power contributions from the orbiting body denoted by the "body" parameter: the routine is
called twice, once for each body.

pos2deldop takes contributions only from the rectangular plane-of-sky region defined by
pos.xlim and pos.ylim -- the smallest rectangle which completely "contains" the model in
the plane of the sky.  No power is contributed by parts of the model which extend beyond
the POS window; this is why such models are heavily penalized (the objective function is
doubled -- see function "objective" in file bestfit.c).

idellim and idoplim are updated for frame frm to show the model delay-Doppler
(rectangular) region that contains nonzero power.

Modified 2015 June 3 by CM:
    Implement smearing for the "fit" and "write" actions by adding "v" (view) parameter
        and applying each run of pos2deldop to a single view rather than to an entire
        (smeared) frame

Modified 2014 February 10 by CM:
    Add "ilaw" argument to the radlaw routine to implement multiple radar scattering laws

Modified 2012 March 23 by CM:
    Implement Doppler scaling

Modified 2010 September 1 by CM:
    Add braces to an if-then-else statement to avoid compilation warning

Modified 2010 August 26 by CM:
    For the "map" action, change the "map_forward" parameter to "map_mode"
        and implement map_mode = 'facets'
    For the "map" action, implement the "map_verbose" parameter

Modified 2010 June 15 by CM:
    Pass the entire par_t parameter structure as an argument rather than
        just selected parameters
    Implement the map action

Modified 2009 November 15 by CM:
    Fix argument type in printf statement

Modified 2009 April 3 by CM:
    Add the "warn_badradar" parameter (see below)
    Rather than halting the program if the sinc^2 Doppler response function
        extends to too many Doppler columns, set a flag and compute a factor
        by which the objective function should be increased (actually the
        logarithm of this factor).  If the "warn_badradar" parameter is
        turned on, print an explicit warning.
    If the model is too wide in delay-Doppler space even for the overflow
        image, set a flag and compute a factor by which the objective
        function should be increased (actually the logarithm of this
        factor).  If the "warn_badradar" parameter is turned on, print an
        explicit warning.
    Make pos2deldop int rather than void in order to return the flag
        described above to the calling procedure

Modified 2007 October 22 by CM:
    Use the vignetted DC column rather than the DC column for the unvignetted
        image

Modified 2007 August 4 by CM:
    Add orbit_xoff, orbit_yoff, and orbit_dopoff parameters, the x offset
        (POS image rows), y offset (POS image columns), and Doppler offset
        (delay-Doppler image columns) of the center of mass due to orbital
        motion.  Note that the delay offset due to orbital motion is already
        taken care of by the posvis routine.
    Add body parameter to indicate (for the "orbit" action) which of the two
        orbiting bodies' power contributions should be computed
    Add c (component) argument to radlaw routine

Modified 2006 September 14 by CM:
    If the overflow region is too small, print a warning rather than
        halting the program

Modified 2006 June 21 by CM:
    Change delres to del_per_pixel and dopres to dop_per_pixel
    For POS renderings, change res to km_per_pixel

Modified 2006 June 18 by CM:
    Allow each delay-Doppler frame in a dataset to have different dimensions
        after vignetting

Modified 2006 March 10 by CM:
    The bug fix of 2005 July 25 inadvertently caused "codefactor" to cancel
        out, so that the sensitivity falloff of short-code images away from
        DC was not properly modeled; fix this so that the sum of each
        POS pixel's contributions to the image is equal to codefactor*sigma
        rather than to sigma, where sigma is the cross-section equivalent
        of the echo power on the sky within that POS pixel.
    Pass the "speckle" parameter so that self-noise can be included when
        computing the chi squared contribution of the overflow region
    Compute overflow_xsec, overflow_delmean and overflow_dopmean so that
        these quantities can be used by the "delcorinit" action

Modified 2005 July 25 by CM:
    Fix bug in overall cross-section scale factor: return to Scott's scheme
        of normalizing the cross-section contributions from a given POS
        pixel so that they sum to the cross section actually present on the
        sky in that pixel

Modified 2005 July 20 by CM:
    Fix bug in computing floating-point delay-Doppler limits in usec and Hz
    Add "facet" argument to radlaw routine

Modified 2005 July 5 by CM:
    Eliminate "dir" argument (since we always add power to the model image
       and never subtract it)
    Add "set" (set number) argument in order to improve error messages

Modified 2005 June 27 by CM:
    Rename INFINITY constant to HUGENUMBER to avoid conflicts

Modified 2005 June 25 by CM:
    Rename old "dellim" to "idellim" and old "doplim" to "idoplim";
        these are delay-Doppler limits in (integer) row and column numbers,
        respectively
    Add new "dellim" and "doplim" which are floating-point delay-Doppler
        limits in usec and Hz, respectively, obtained PRIOR to convolution
        with the delay and Doppler response functions

Modified 2005 January 25 by CM:
    Take care of uninitialized variables to avoid compilation warnings

Modified 2003 May 11 by CM:
    Compute contributions to chi squared by model power which lies
        outside the limits of the data frame.

Modified 2003 May 5 by CM:
    For each POS pixel, compute the entire pixel's contribution to a
        given Doppler column in the model image so long as even one point
        at which we evaluate the sinc^2 response function is less than
        sinc2width/2.0 bins away from that Doppler column.  In other words,
        err on the side of computing too many small contributions to each
        column in the model image, so as not to omit significant
        contributions just because a POS pixel's *center* isn't close
        enough in Doppler.

Modified 2003 April 29 by CM:
    Evaluate the sinc^2 Doppler response function at nsinc2 points
        per POS pixel dimension, not just at the pixel center.
        The sinc^2 function varies rapidly -- one null per Doppler bin
        away from the central peak -- so if the pixel width is more than
        about half the Doppler resolution, we want to take the mean of
        several points within the pixel.

Modified 2003 April 26 by CM:
    Zero out the sinc^2 Doppler response function beyond the
        nearest sinc2width bins rather than beyond the nearest 2 bins

Modified 2003 April 24 by CM:
    Move "delcom" (COM delay bin) from delay-Doppler datasets to
        individual frames

Modified 2003 April 14 by CM:
    Now handles multiple samples per baud.
    Now handles short code vs. long code (original reduction method)
                           vs. long code (modified reduction method)
    Now correctly scales the model delay-Doppler image to account
        for delay mismatching and correlations between image rows
        (i.e., does NOT require the cross-section contributions from
        each plane-of-sky pixel to sum to the cross section actually
        present in that POS pixel)
*****************************************************************************************/

#include "head.h"

int pos2deldop( struct par_t *par, struct photo_t *photo,
                double orbit_xoff, double orbit_yoff, double orbit_dopoff,
                struct deldop_t *deldop, int body, int set, int frm, int v)
{
  int x, y, idel, idel_min, idel_max, idop, idop_min, idop_max,
      codemethod, dopfftlen, spb, stride, spb_sq, m, m_min, m_max,
      nsinc2_sq, i, j, k, any_overflow, in_bounds, idel0, idop0,
      spb_over_stride, ndel, ndop, idel1, idel2, idop1, idop2, i1, i2,
      j1, j2, badradar, c, f;
  double delPOS, dopPOS, dopDC_vig, delfact, dopfact, w[3], ax, ay, delshift,
         dopshift, const1, const2, codefactor, dop_contribution[MAXBINS],
         del_contribution[MAXBINS], tmp, amp, xincr, yincr, arg_left,
         sinc2arg, sinc2_mean, arg_bl, dopdiff_bl, dopdiff_max,
         one_over_spb, arg_sample, lookfact, sdev_sq, variance,
         dop_extra, sumweights, delfactor, dopfactor, fit_contribution;
  struct deldopfrm_t *frame;
  struct pos_t *pos;

  /*  Initialize variables to avoid compilation warnings  */

  idel0 = idop0 = 0;

  /*  Initialize other variables  */

  static double fit_overflow[MAXOVERFLOW][MAXOVERFLOW];
  any_overflow = 0;

  frame = &deldop->frame[frm];
  pos = &frame->pos;

  ndel = frame->ndel;
  ndop = frame->ndop;
  frame->idellim[0] = ndel + 999999;
  frame->idellim[1] = -999999;
  frame->idoplim[0] = ndop + 999999;
  frame->idoplim[1] = -999999;
  frame->dellim[0] =  HUGENUMBER;
  frame->dellim[1] = -HUGENUMBER;
  frame->doplim[0] =  HUGENUMBER;
  frame->doplim[1] = -HUGENUMBER;

  badradar = 0;
  frame->badradar_logfactor = 0.0;

  /*
      Get parameters related to data sampling and data reduction;
      then compute two more (both in units of delay bins = image rows):

      const1: half of the base width of the delay response function
      const2: half the delay difference between the first and last
                  image rows within each baud
  */

  codemethod = deldop->codemethod;
  spb = deldop->spb;
  stride = deldop->stride;
  dopfftlen = deldop->dopfftlen;
  dopDC_vig = frame->dopDC_vig;
  spb_over_stride = spb/stride;
  one_over_spb = 1.0/spb;
  spb_sq = spb*spb;
  if (codemethod != LONG_ORIG) {
      const1 = (3*spb - 1)/(2.0*stride);
      const2 = (spb - 1)/(2.0*stride);
  } else {
      const1 = (double) spb_over_stride;
      const2 = 0.0;  /* not used with this code + reduction method */
  }

  /*  Converts from km towards radar to delay bins  */
  delfact = -KM2US/deldop->del_per_pixel;

  /*  Get w, the apparent spin vector in observer coordinates  */
  cotrans( w, frame->view[v].oe, frame->view[v].spin, 1);

  /*  Compute the Doppler bin increment per plane-of-sky pixel westward (ax)
      and northward (ay); these values are scaled by the "dopscale" parameter
      for this dataset.  Then compute km2Hz, the Doppler increment (Hz) per
      km perpendicular to the projected spin axis in the plane of the sky.     */

  dopfact = deldop->dopscale.val * KM2HZFACT * pos->km_per_pixel
                      * deldop->Ftx / deldop->dop_per_pixel;
  ax = -w[1]*dopfact;
  ay = w[0]*dopfact;
  frame->view[v].km2Hz = sqrt(ax*ax + ay*ay) * deldop->dop_per_pixel
                                               / pos->km_per_pixel;

  /*  Compute the absolute value of the difference between the
      maximum (or minimum) Doppler on any given POS pixel's edge
      and the Doppler at its center                               */

  if (w[0] != 0.0 || w[1] != 0.0)
    dop_extra = frame->view[v].km2Hz * 0.5 * pos->km_per_pixel
                * sqrt(w[0]*w[0] + w[1]*w[1]) / MAX( fabs(w[0]), fabs(w[1]));
  else
    dop_extra = 0.0;

  /*  We may be evaluating the sinc^2 Doppler response function at
      more than one point per POS pixel.  xincr and yincr are the
      Doppler bin increments between adjacent evaluation points in the
      x and y directions.  dopdiff_bl is the Doppler bin difference
      between the bottom-leftmost (southeasternmost) evaluation point
      and the pixel center.  dopdiff_max is the maximum positive
      Doppler bin difference between any evaluation point and the
      pixel center.                                                     */

  nsinc2_sq = par->nsinc2 * par->nsinc2;
  xincr = ax / par->nsinc2;
  yincr = ay / par->nsinc2;
  dopdiff_bl = -(par->nsinc2 - 1)*(xincr + yincr)/2;
  dopdiff_max = (par->nsinc2 - 1)*(fabs(xincr) + fabs(yincr))/2;
  if (2*dopdiff_max + par->sinc2width + 1 > MAXBINS) {
    badradar = 1;
    frame->badradar_logfactor += log((2*dopdiff_max + par->sinc2width + 1) / MAXBINS);
    if (par->warn_badradar) {
      printf("\nWARNING in pos2deldop.c for set %2d frame %2d:\n", set, frm);
      printf("        sinc^2 function evaluated at %d Doppler bins, must be <= %d\n",
             (int) ceil(2*dopdiff_max + par->sinc2width + 1), MAXBINS);
      fflush(stdout);
    }
  }

  /*  Get the COM delay and Doppler bins, corrected for ephemeris drift
      and adjusted for orbital motion; the delay adjustment for orbital
      motion is done implicitly (the posvis routine has already adjusted
      the "z" values for all POS pixels), whereas the Doppler adjustment
      for orbital motion must be done here explicitly.                    */

  delshift = frame->delcom_vig + frame->view[v].deloff;
  dopshift = frame->dopcom_vig + frame->view[v].dopoff + orbit_dopoff;

  /*
      Loop through all POS pixels within the rectangular plane-of-sky region
      spanned by the model; for each such pixel which isn't blank sky, compute
      the cross-section contributions to pixels in the model delay-Doppler frame.
      Note that functions posclr and posvis flag blank-sky pixels by assigning
      "cose" = cos(scattering angle) = 0.

      Only compute contributions from POS pixels that project onto the right body,
      in case this is the "orbit" action (for which this routine is called twice,
      once for each of the two orbiting bodies).
  */

  for (x=pos->xlim[0]; x<=pos->xlim[1]; x++)
    for (y=pos->ylim[0]; y<=pos->ylim[1]; y++) {
      if (pos->cose[x][y] > 0.0 && pos->body[x][y] == body) {

        /*
            Get the (floating-point) delay and Doppler bin of the POS pixel center:
            delPOS and dopPOS.  Also get the minimum and maximum (integer) delay
            and Doppler bins to which this pixel contributes power: idel_min and
            idel_max, idop_min and idop_max.  Strictly speaking, each POS pixel
            contributes power to *all* Doppler columns, but here we're zeroing out
            the sinc^2 response function beyond the nearest sinc2width columns.

            Actually, if nsinc2 > 1, we'll distribute power to *at least*
            sinc2width Doppler bins: For pixels which span multiple bins we'll
            err on the side of computing more contributions rather than fewer.
        */

        delPOS = pos->z[x][y]*delfact + delshift;
        idel_min = (int) floor(delPOS - const1) + 1;
        idel_max = (int) ceil(delPOS + const1) - 1;
        dopPOS = ax*(x - orbit_xoff) + ay*(y - orbit_yoff) + dopshift;
        idop_min = (int) floor(dopPOS - dopdiff_max + 1 - par->sinc2width/2.0);
        idop_max = (int) floor(dopPOS + dopdiff_max + par->sinc2width/2.0);

        /*  For the short code, sensitivity drops as we move away from DC.
            (This variation is slow, so we can just evaluate the response
            at the center of the POS pixel.)

            Note that the SINC2 macro multiplies its argument by pi.        */

        codefactor = (codemethod == SHORT) ? SINC2( (dopPOS-dopDC_vig)/dopfftlen ) : 1.0;

        /*  Update the rectangular delay-Doppler region (row and
            column numbers) with nonzero power according to the model  */

        if (idel_min < frame->idellim[0])
            frame->idellim[0] = idel_min;
        if (idel_max > frame->idellim[1])
            frame->idellim[1] = idel_max;
        if (idop_min < frame->idoplim[0])
            frame->idoplim[0] = idop_min;
        if (idop_max > frame->idoplim[1])
            frame->idoplim[1] = idop_max;

        /*
            Update the model's floating-point delay-Doppler limits, as determined
            prior to convolution with the delay and Doppler response functions

            At this point in the code, dellim and doplim are pairs of
            floating-point row and column numbers which apply to POS pixel
            centers; when the loop over POS pixels is finished we will convert
            them to usec and Hz, and will widen the Doppler limits to account for
            nonzero POS pixel width.
        */

        frame->dellim[0] = MIN( frame->dellim[0], delPOS);
        frame->dellim[1] = MAX( frame->dellim[1], delPOS);
        frame->doplim[0] = MIN( frame->doplim[0], dopPOS);
        frame->doplim[1] = MAX( frame->doplim[1], dopPOS);

        /*  Check whether or not all delay-Doppler pixels which will
            receive power from this POS pixel fall within the data frame;
            if not, initialize the "overflow" image if necessary.          */

        if ( (idel_min >= 1) && (idel_max <= ndel) &&
             (idop_min >= 1) && (idop_max <= ndop)    )
          in_bounds = 1;
        else {
          in_bounds = 0;
          if (!any_overflow) {
            any_overflow = 1;
            for (i=0; i<MAXOVERFLOW; i++)
              for (j=0; j<MAXOVERFLOW; j++)
                fit_overflow[i][j] = 0.0;

            /*  Center the COM in the overflow image:
                pixel [idel][idop] in the fit frame corresponds to
                pixel [idel+idel0][idop+idop0] in the fit_overflow frame.  */

            idel0 = MAXOVERFLOW/2 - (int) floor(delshift + 0.5);
            idop0 = MAXOVERFLOW/2 - (int) floor(dopshift + 0.5);
          }
        }

        /*  Loop through all delay bins to which this POS pixel
            contributes power (cross section), and compute the
            delay response function for each bin                 */

        for (idel=idel_min; idel<=idel_max; idel++) {
          if (codemethod != LONG_ORIG) {

              /*
                  Get the delay response function for image row idel:
                  sum the triangle-function contributions from each
                  sample per baud, then divide the sum by spb and square.

                  The triangle function for sample m  (0 <= m <= spb-1) has
                  unit height and a half-width of spb/stride image rows, and
                  is centered [-const2 + m/stride] rows later than the row
                  center (idel).

                  In the code block below, the arguments to macros TRI and
                  TRI2 have been divided by half-width spb/stride, since
                  those two macros are defined to give nonzero values for
                  arguments between -1 and 1.  Each argument, then, is just
                  (delPOS - [triangle-function center]) / half-width.

                  Do the two most common cases (spb = 1 or 2) without
                  loops in order to gain a bit of speed.  For the other
                  cases, set m_min and m_max so as not to waste time
                  computing contributions that are zero.
              */

              switch (spb) {
                case 1:
                  del_contribution[idel-idel_min] = TRI2( delPOS - idel );
                  break;
                case 2:
                  arg_sample = (delPOS - (idel - const2)) / spb_over_stride;
                  del_contribution[idel-idel_min] = TRI( arg_sample )
                                                    + TRI( arg_sample - 0.5 );
                  del_contribution[idel-idel_min] *= del_contribution[idel-idel_min]/4;
                  break;
                default:
                  del_contribution[idel-idel_min] = 0.0;
                  m_min = MAX( (int) floor((delPOS - idel - const2)*stride) , 0 );
                  m_max = MIN( (int) ceil((delPOS - idel + const1)*stride) , spb ) - 1;
                  arg_sample = (delPOS - (idel - const2)) / spb_over_stride
                               - m_min*one_over_spb;
                  for (m=m_min; m<=m_max; m++) {
                    del_contribution[idel-idel_min] += TRI( arg_sample );
                    arg_sample -= one_over_spb;
                  }
                  del_contribution[idel-idel_min] *= del_contribution[idel-idel_min]/spb_sq;
                  break;
              }

          } else {

              /*  Long code with original (Harmon) reduction method:
                  data for each sample per baud are reduced separately,
                  as if datataking were just one sample per baud; then
                  the image rows for spb/stride samples are interleaved.  */

              del_contribution[idel-idel_min] = TRI2( (delPOS - idel)/spb_over_stride );
          }
        }

        /*  Next include the sinc^2 factor for Doppler mismatching:
            Take the mean of nsinc2^2 points interior to the POS pixel.
            Do the two most common cases (nsinc2 = 1 or 2) without
            loops in order to gain a bit of speed.

            Note that the SINC2 macro multiplies its argument by pi.        */

        for (idop=idop_min; idop<=idop_max; idop++) {
          switch (par->nsinc2) {
            case 1:
              sinc2_mean = SINC2( dopPOS - idop );
              break;
            case 2:
              arg_bl = dopPOS + dopdiff_bl - idop;   /* bl = bottom left */
              sinc2_mean = ( SINC2( arg_bl ) +
                             SINC2( arg_bl+xincr ) +
                             SINC2( arg_bl+yincr ) +
                             SINC2( arg_bl+xincr+yincr ) ) / 4;
              break;
            default:
              arg_left = dopPOS + dopdiff_bl - idop;
              sinc2_mean = 0.0;
              for (i=0; i<par->nsinc2; i++) {
                sinc2arg = arg_left;
                for (j=0; j<par->nsinc2; j++) {
                  sinc2_mean += SINC2( sinc2arg );
                  sinc2arg += xincr;
                }
                arg_left += yincr;
              }
              sinc2_mean /= nsinc2_sq;
              break;
          }
          k = MIN( idop - idop_min, MAXBINS);
          dop_contribution[k] = sinc2_mean;
        }

        /*  Compute the sum of delay-Doppler weighting factors  */

        sumweights = 0.0;
        for (idel=idel_min; idel<=idel_max; idel++)
          for (idop=idop_min; idop<=idop_max; idop++) {
            k = MIN( idop - idop_min, MAXBINS);
            sumweights += del_contribution[idel-idel_min]*dop_contribution[k];
          }

        /*
            The radar cross section within this plane-of-sky pixel is

                [differential radar scattering law]*[POS pixel area in km^2]

            The differential radar scattering law (function radlaw
            = d[cross section]/d[area] ) includes a sec(theta) factor to
            account for the fact that the POS pixel area is projected area
            rather than physical area on the target surface.
        */

        amp = radlaw( photo, deldop->iradlaw, pos->cose[x][y],
                      pos->comp[x][y], pos->f[x][y])
              * pos->km_per_pixel * pos->km_per_pixel
              * codefactor / sumweights;

        /*  Only add this POS pixel's power contributions to the model
            delay-Doppler frame if NONE of those contributions fall
            outside the frame limits.                                   */

        if (in_bounds) {

            /*  Add the cross-section contributions to the model frame  */

            for (idel=idel_min; idel<=idel_max; idel++)
              for (idop=idop_min; idop<=idop_max; idop++) {
                k = MIN( idop - idop_min, MAXBINS);
                fit_contribution = amp * del_contribution[idel-idel_min]
                                       * dop_contribution[k];
                frame->fit[idel][idop] += fit_contribution;

                if (par->action == MAP) {
                  if (par->map_mode == MAPMODE_DELDOP) {
                      if (frame->map_fit[idel][idop] > 0.0) {
                        frame->map_pos[x][y] += fit_contribution;
                        c = pos->comp[x][y];
                        f = pos->f[x][y];
                        frame->map_facet_power[c][f] += fit_contribution;
                        if (par->map_verbose)
                          printf("# POS (%3d, %3d) comp %d facet %4d contributes %e to d-D (%3d, %3d)\n",
                                 x+pos->n, y+pos->n, c, f, fit_contribution, idel-1, idop-1);
                      }
                  } else if (par->map_mode == MAPMODE_POS) {
                      if (frame->map_pos[x][y] > 0.0) {
                        frame->map_fit[idel][idop] += fit_contribution;
                        c = pos->comp[x][y];
                        f = pos->f[x][y];
                        frame->map_facet_power[c][f] += fit_contribution;
                        if (par->map_verbose)
                          printf("# POS (%3d, %3d) comp %d facet %4d contributes %e to d-D (%3d, %3d)\n",
                                 x+pos->n, y+pos->n, c, f, fit_contribution, idel-1, idop-1);
                      }
                  } else {
                      if (frame->map_pos[x][y] > 0.0) {
                        frame->map_fit[idel][idop] += fit_contribution;
                        if (par->map_verbose) {
                          c = pos->comp[x][y];
                          f = pos->f[x][y];
                          printf("# POS (%3d, %3d) comp %d facet %4d contributes %e to d-D (%3d, %3d)\n",
                                 x+pos->n, y+pos->n, c, f, fit_contribution, idel-1, idop-1);
                        }
                      }
                  }
                }
              }

        } else {

            /*  Add the cross-section contributions to the "overflow" image  */

            if (par->action == MAP && par->map_mode != MAPMODE_DELDOP)
              if (frame->map_pos[x][y] > 0.0)
                par->map_overflow = 1;

            idel1 = MAX( idel_min, -idel0);
            idel2 = MIN( idel_max, -idel0 + MAXOVERFLOW - 1);
            idop1 = MAX( idop_min, -idop0);
            idop2 = MIN( idop_max, -idop0 + MAXOVERFLOW - 1);
            for (idel=idel1; idel<=idel2; idel++)
              for (idop=idop1; idop<=idop2; idop++) {
                k = MIN( idop - idop_min, MAXBINS);
                fit_contribution = amp * del_contribution[idel-idel_min]
                                       * dop_contribution[k];
                fit_overflow[idel+idel0][idop+idop0] += fit_contribution;
                if (par->action == MAP && par->map_mode == MAPMODE_DELDOP)
                  if (idel >= par->map_dellim[0] && idel <= par->map_dellim[1] &&
                      idop >= par->map_doplim[0] && idop <= par->map_doplim[1]    ) {
                    frame->map_pos[x][y] += fit_contribution;
                    c = pos->comp[x][y];
                    f = pos->f[x][y];
                    frame->map_facet_power[c][f] += fit_contribution;
                    if (par->map_verbose)
                      printf("# POS (%3d, %3d) comp %d facet %4d contributes %e to d-D (%3d, %3d)\n",
                             x+pos->n, y+pos->n, c, f, fit_contribution, idel-1, idop-1);
                  }
              }
        }

      }  /* if cos(scattering angle) > 0 and POS pixel projects onto the right body */
    }  /* go to the next POS image pixel */

  /*  --- end loop over POS pixels ---  */

  /*  Convert the model's floating-point delay-Doppler limits from
      floating-point row and column numbers to usec and Hz, and
      widen the Doppler limits to account for nonzero POS pixel width  */

  frame->dellim[0] = (frame->dellim[0] - delshift)*deldop->del_per_pixel;
  frame->dellim[1] = (frame->dellim[1] - delshift)*deldop->del_per_pixel;
  frame->doplim[0] = (frame->doplim[0] - dopshift)*deldop->dop_per_pixel
                       - dop_extra;
  frame->doplim[1] = (frame->doplim[1] - dopshift)*deldop->dop_per_pixel
                       + dop_extra;

  /*  Calculate the overflow contributions to chi squared:
      o2 = obs^2 contribution, m2 = model^2 contribution.

      Also compute the summed cross section and the mean delay and Doppler
      bins for the overflow region, for use with the "delcorinit" action    */

  frame->overflow_o2 = 0.0;
  frame->overflow_m2 = 0.0;
  frame->overflow_xsec = 0.0;
  frame->overflow_delmean = 0.0;
  frame->overflow_dopmean = 0.0;
  sdev_sq = frame->sdev*frame->sdev;
  variance = sdev_sq;
  lookfact = (frame->nlooks > 0.0) ? 1.0/frame->nlooks : 0.0;
  if (any_overflow) {
	i1 = MAX( frame->idellim[0] + idel0, 0);
    i2 = MIN( frame->idellim[1] + idel0, MAXOVERFLOW - 1);
    j1 = MAX( frame->idoplim[0] + idop0, 0);
    j2 = MIN( frame->idoplim[1] + idop0, MAXOVERFLOW - 1);
    for (i=i1; i<=i2; i++)
      for (j=j1; j<=j2; j++) {
        if (fit_overflow[i][j] != 0.0) {
          if (par->speckle)
            variance = sdev_sq + lookfact*fit_overflow[i][j]*fit_overflow[i][j];
          frame->overflow_o2 += 1.0;
          frame->overflow_m2 += fit_overflow[i][j]*fit_overflow[i][j]/variance;
          frame->overflow_xsec += fit_overflow[i][j];
          frame->overflow_delmean += (i - idel0)*fit_overflow[i][j];
          frame->overflow_dopmean += (j - idop0)*fit_overflow[i][j];
        }
      }
    if (frame->overflow_xsec != 0.0) {
      frame->overflow_delmean /= frame->overflow_xsec;
      frame->overflow_dopmean /= frame->overflow_xsec;
    }

    /*  Print a warning if the model extends even beyond the overflow image  */

    if ( ((frame->idellim[0] + idel0) < 0)            ||
         ((frame->idellim[1] + idel0) >= MAXOVERFLOW) ||
         ((frame->idoplim[0] + idop0) < 0)            ||
         ((frame->idoplim[1] + idop0) >= MAXOVERFLOW)    ) {
      badradar = 1;
      delfactor = (MAX( frame->idellim[1] + idel0, MAXOVERFLOW)
                   - MIN( frame->idellim[0] + idel0, 0)         )
                  / (1.0*MAXOVERFLOW);
      dopfactor = (MAX( frame->idoplim[1] + idop0, MAXOVERFLOW)
                   - MIN( frame->idoplim[0] + idop0, 0)         )
                  / (1.0*MAXOVERFLOW);
      frame->badradar_logfactor += log(delfactor*dopfactor);
      if (par->warn_badradar) {
        printf("\nWARNING in pos2deldop.c for set %2d frame %2d:\n", set, frm);
        printf("        model delay-Doppler image extends too far beyond the data image\n");
        printf("             data:  rows %2d to %2d , cols %2d to %2d\n", 1, ndel, 1, ndop);
        printf("            model:  rows %2d to %2d , cols %2d to %2d\n",
               frame->idellim[0], frame->idellim[1],
               frame->idoplim[0], frame->idoplim[1]);
        fflush(stdout);
      }
    }
  }

  return badradar;
}
