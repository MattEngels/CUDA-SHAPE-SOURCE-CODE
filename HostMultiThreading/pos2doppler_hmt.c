/*****************************************************************************************
                                                                            pos2doppler.c

Takes a pos_t structure and a doppler_t structure and "fills in" a dopfrm_t structure
indexed by frm.  In other words, pos2doppler works from a model plane-of-sky image (with
an observer z-coordinate and a scattering angle assigned to each pixel) to produce a model
Doppler spectrum corresponding to data frame frm.

In the case of an orbiting binary system (the "orbit" action), pos2doppler only computes
power contributions from the orbiting body denoted by the "body" parameter: the routine is
called twice, once for each body.

pos2doppler takes contributions only from the rectangular plane-of-sky region defined by
pos.xlim and pos.ylim -- the smallest rectangle which completely "contains" the model in
the plane of the sky.  No power is contributed by parts of the model which extend beyond
the POS window; this is why such models are heavily penalized (the objective function is
doubled -- see function f in file bestfit.c).

idoplim is updated for frame frm to show the model Doppler region that contains nonzero
power.

Modified 2015 June 3 by CM:
    Implement smearing for the "fit" and "write" actions by adding "v" (view) parameter
        and applying each run of pos2doppler to a single view rather than to an entire
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
    Fix argument type in a printf statement

Modified 2009 April 3 by CM:
    Add the "warn_badradar" parameter (see below)
    If the sinc^2 Doppler response function extends to too many Doppler
        bins, set a flag and compute a factor by which the objective
        function should be increased (actually the logarithm of this
        factor).  If the "warn_badradar" parameter is turned on, print an
        explicit warning.
    If the model is too wide in Doppler space even for the overflow image,
        set a flag and compute a factor by which the objective function
        should be increased (actually the logarithm of this factor).  If
        the "warn_badradar" parameter is turned on, print an explicit
        warning.
    Make pos2doppler int rather than void in order to return the flag
        described above to the calling procedure

Modified 2007 August 4 by CM:
    Add orbit_xoff, orbit_yoff, and orbit_dopoff parameters, the x offset
        (POS image rows), y offset (POS image columns), and Doppler offset
        (spectral bins) of the center of mass due to orbital motion.
    Add body parameter to indicate (for the "orbit" action) which of the two
        orbiting bodies' power contributions should be computed
    Add c (component) argument to radlaw routine

Modified 2006 September 14 by CM:
    If the overflow region is too small, print a warning rather than
        halting the program

Modified 2006 June 21 by CM:
    Change dopres to dop_per_bin
    For POS renderings, change res to km_per_pixel

Modified 2006 June 18 by CM:
    Allow each Doppler frame in a dataset to have different dimensions
        after vignetting

Modified 2006 March 10 by CM:
    Pass the "speckle" parameter so that self-noise can be included when
        computing the chi squared contribution of the overflow region
    Compute overflow_xsec and overflow_dopmean so that these quantities
        can be used by the "delcorinit" action

Modified 2005 July 25 by CM:
    Fix bug in overall cross-section scale factor: return to Scott's scheme
        of normalizing the cross-section contributions from a given POS 
        pixel so that they sum to the cross section actually present on the 
        sky in that pixel

Modified 2005 July 20 by CM:
    Fix bug in computing floating-point Doppler limits in Hz
    Add "facet" argument to radlaw routine

Modified 2005 July 5 by CM:
    Eliminate "dir" argument (since we always add power to the model image
       and never subtract it)
    Add "set" (set number) argument in order to improve error messages

Modified 2005 June 27 by CM:
    Rename INFINITY constant to HUGENUMBER to avoid conflicts

Modified 2005 June 25 by CM:
    Rename old "doplim" to "idoplim"; this is the Doppler limits in
        (integer) bin numbers
    Add new "doplim" which is the floating-point Doppler limits in Hz,
        obtained PRIOR to convolution with the Doppler response function

Modified 2005 January 25 by CM:
    Take care of uninitialized variable

Modified 2003 May 11 by CM:
    Compute contributions to chi squared by model power which lies
    outside the limits of the data frame.

Modified 2003 May 5 by CM:
    For each POS pixel, compute the entire pixel's contribution to a
    given Doppler bin in the model spectrum so long as even one point
    at which we evaluate the sinc^2 response function is less than
    sinc2width/2.0 bins away from that Doppler bin.  In other words,
    err on the side of computing too many small contributions to each
    bin in the model spectrum, so as not to omit significant contributions
    just because a POS pixel's *center* isn't close enough in Doppler.

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

Modified 2003 April 17 by CM:
    Now correctly scales the model Doppler spectrum to account for
    Doppler mismatching
 *****************************************************************************************/

#include "../shape/head.h"

int pos2doppler_hmt( struct par_t *par, struct photo_t *photo, double orbit_xoff,
		double orbit_yoff, double orbit_dopoff, struct doppler_t *doppler,
		int body, int set, int frm, int v)
{
	int x, y, idop_min, idop_max, idop, i, j, k, nsinc2_sq, any_overflow,
	in_bounds, idop0, ndop, idop1, idop2, j1, j2, badradar, c, f;
	double dopPOS, dopfact, w[3], ax, ay, dopshift, tmp, amp, xincr, yincr,
	arg_left, sinc2arg, sinc2_mean, arg_bl, dopdiff_bl, dopdiff_max,
	lookfact, sdev_sq, variance, dop_extra, dop_contribution[MAXBINS],
	sumweights, dopfactor, fit_contribution;
	struct dopfrm_t *frame;
	struct pos_t *pos;

	/*  Initialize variable to avoid compilation warning  */
	idop0 = 0;

	/*  Initialize other variables  */
	any_overflow = 0;
	frame = &doppler->frame[frm];
	pos = &frame->pos;
	ndop = frame->ndop;
	frame->idoplim[0] = ndop + 999999;
	frame->idoplim[1] = -999999;
	frame->doplim[0] =  HUGENUMBER;
	frame->doplim[1] = -HUGENUMBER;

	badradar = 0;
	frame->badradar_logfactor = 0.0;

	/*  Get w, the apparent spin vector in observer coordinates  */
	cotrans( w, frame->view[v].oe, frame->view[v].spin, 1);

	/*  Compute the Doppler bin increment per plane-of-sky pixel westward (ax)
      and northward (ay); these values are scaled by the "dopscale" parameter
      for this dataset.  Then compute km2Hz, the Doppler increment (Hz) per
      km perpendicular to the projected spin axis in the plane of the sky.     */

	dopfact = doppler->dopscale.val * KM2HZFACT * pos->km_per_pixel
			* doppler->Ftx / doppler->dop_per_bin;
	ax = -w[1]*dopfact;
	ay = w[0]*dopfact;
	frame->view[v].km2Hz = sqrt(ax*ax + ay*ay) * doppler->dop_per_bin
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
			printf("\nWARNING in pos2doppler.c for set %2d frame %2d:\n", set, frm);
			printf("        sinc^2 function evaluated at %d Doppler bins, must be <= %d\n",
					(int) ceil(2*dopdiff_max + par->sinc2width + 1), MAXBINS);
			fflush(stdout);
		}
	}

	/*  Get the COM Doppler bin, corrected for ephemeris drift
      and adjusted for orbital motion                         */

	dopshift = frame->dopcom_vig + frame->view[v].dopoff + orbit_dopoff;

/*	Loop through all POS pixels within the rectangular plane-of-sky region
    spanned by the model; for each such pixel which isn't blank sky, compute
    the cross-section contributions to pixels in the model Doppler spectrum.
    Note that functions posclr and posvis flag blank-sky pixels by assigning
    "cose" = cos(scattering angle) = 0.

    Only compute contributions from POS pixels that project onto the right body,
    in case this is the "orbit" action (for which this routine is called twice,
    once for each of the two orbiting bodies).	 */

	for (x=pos->xlim[0]; x<=pos->xlim[1]; x++)
		for (y=pos->ylim[0]; y<=pos->ylim[1]; y++) {
			if (pos->cose[x][y] > 0.0 && pos->body[x][y] == body) {

		/*  Get the (floating-point) Doppler bin of the POS pixel center: dopPOS.
            Also get the minimum and maximum (integer) Doppler bins to which this
            pixel contributes power: idop_min and idop_max.  Strictly speaking,
            each POS pixel contributes power to *all* Doppler bins, but here
            we're zeroing out the sinc^2 response function beyond the nearest
            sinc2width bins.

            Actually, if nsinc2 > 1, we'll distribute power to *at least*
            sinc2width Doppler bins: For pixels which span multiple bins we'll
            err on the side of computing more contributions rather than fewer. */

				dopPOS = ax*(x - orbit_xoff) + ay*(y - orbit_yoff) + dopshift;
				idop_min = (int) floor(dopPOS - dopdiff_max + 1 - par->sinc2width/2.0);
				idop_max = (int) floor(dopPOS + dopdiff_max + par->sinc2width/2.0);

		/*  Update the rectangular delay-Doppler region with
            nonzero power according to the model              */

				if (idop_min < frame->idoplim[0])	frame->idoplim[0] = idop_min;
				if (idop_max > frame->idoplim[1])	frame->idoplim[1] = idop_max;

		/* 	Update the model's floating-point Doppler limits, as determined
            prior to convolution with the Doppler response function

            At this point in the code, doplim is a pair of floating-point bin
            numbers which applies to POS pixel centers; when the loop over
            POS pixels is finished we will convert these limits to Hz and
            will widen the limits to account for nonzero POS pixel width. */

				frame->doplim[0] = MIN( frame->doplim[0], dopPOS);
				frame->doplim[1] = MAX( frame->doplim[1], dopPOS);

		/*  Check whether or not all Doppler bins which will receive
            power from this POS pixel fall within the data frame;
            if not, initialize the "overflow" spectrum if necessary.  */

				if ( (idop_min >= 1) && (idop_max <= ndop) )
					in_bounds = 1;
				else {
					in_bounds = 0;
					if (!any_overflow) {
						any_overflow = 1;
						for (j=0; j<MAXOVERFLOW; j++)
							doppler->fit_overflow[j] = 0.0;

						/*  Center the COM in the overflow spectrum:
                bin [idop] in the fit frame corresponds to
                bin [idop+idop0] in the fit_overflow frame.  */

						idop0 = MAXOVERFLOW/2 - (int) floor(dopshift + 0.5);
					}
				}

				/*  Compute the sinc^2 factors for Doppler mismatching:
            Take the mean of nsinc2^2 points interior to the POS pixel.
            Do the two most common cases (nsinc2 = 1 or 2) without
            loops in order to gain a bit of speed.  Note that the SINC2
            macro multiplies its argument by pi.

            Then add the cross-section contributions to the model spectrum.  */

				for (idop=idop_min; idop<=idop_max; idop++) {
					switch (par->nsinc2) {
					case 1:
						sinc2_mean = SINC2(dopPOS - idop);
						break;
					case 2:
						arg_bl = dopPOS + dopdiff_bl - idop;   /* bl = bottom left */
						sinc2_mean = (SINC2(arg_bl) +
								SINC2(arg_bl+xincr) +
								SINC2(arg_bl+yincr) +
								SINC2(arg_bl+xincr+yincr)) / 4;
						break;
					default:
						arg_left = dopPOS + dopdiff_bl - idop;
						sinc2_mean = 0.0;
						for (i=0; i<par->nsinc2; i++) {
							sinc2arg = arg_left;
							for (j=0; j<par->nsinc2; j++) {
								sinc2_mean += SINC2(sinc2arg);
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

				/*  Compute the sum of Doppler weighting factors  */

				sumweights = 0.0;
				for (idop=idop_min; idop<=idop_max; idop++) {
					k = MIN( idop - idop_min, MAXBINS);
					sumweights += dop_contribution[k];
				}


				/*
            The radar cross section within this plane-of-sky pixel is

                [differential radar scattering law]*[POS pixel area in km^2]

            The differential radar scattering law (function radlaw
            = d[cross section]/d[area] ) includes a sec(theta) factor to
            account for the fact that the POS pixel area is projected area
            rather than physical area on the target surface.
				 */

				amp = radlaw( photo, doppler->iradlaw, pos->cose[x][y],
						pos->comp[x][y], pos->f[x][y])
            						  * pos->km_per_pixel * pos->km_per_pixel
            						  / sumweights;

				/*  Only add this POS pixel's power contributions to the model
            Doppler spectrum if NONE of those contributions fall
            outside the spectrum limits.                                 */

				if (in_bounds) {

					/*  Add the cross-section contributions to the model frame  */
					for (idop=idop_min; idop<=idop_max; idop++) {
						k = MIN( idop - idop_min, MAXBINS);
						fit_contribution = amp * dop_contribution[k];
						frame->fit[idop] += fit_contribution;

						if (par->action == MAP) {
							if (par->map_mode == MAPMODE_DELDOP) {
								if (frame->map_fit[idop] > 0.0) {
									frame->map_pos[x][y] += fit_contribution;
									c = pos->comp[x][y];
									f = pos->f[x][y];
									frame->map_facet_power[c][f] += fit_contribution;
									if (par->map_verbose)
										printf("# POS (%3d, %3d) comp %d facet %4d contributes %e to Dop (%3d)\n",
												x+pos->n, y+pos->n, c, f, fit_contribution, idop-1);
								}
							} else if (par->map_mode == MAPMODE_POS) {
								if (frame->map_pos[x][y] > 0.0) {
									frame->map_fit[idop] += fit_contribution;
									c = pos->comp[x][y];
									f = pos->f[x][y];
									frame->map_facet_power[c][f] += fit_contribution;
									if (par->map_verbose)
										printf("# POS (%3d, %3d) comp %d facet %4d contributes %e to Dop (%3d)\n",
												x+pos->n, y+pos->n, c, f, fit_contribution, idop-1);
								}
							} else {
								if (frame->map_pos[x][y] > 0.0) {
									frame->map_fit[idop] += fit_contribution;
									if (par->map_verbose) {
										c = pos->comp[x][y];
										f = pos->f[x][y];
										printf("# POS (%3d, %3d) comp %d facet %4d contributes %e to Dop (%3d)\n",
												x+pos->n, y+pos->n, c, f, fit_contribution, idop-1);
									}
								}
							}
						}
					}

				} else {

					/*  Add the cross-section contributions to the "overflow" spectrum  */

					if (par->action == MAP && par->map_mode != MAPMODE_DELDOP)
						if (frame->map_pos[x][y] > 0.0)
							par->map_overflow = 1;

					idop1 = MAX( idop_min, -idop0);
					idop2 = MIN( idop_max, -idop0 + MAXOVERFLOW - 1);
					for (idop=idop1; idop<=idop2; idop++) {
						k = MIN( idop - idop_min, MAXBINS);
						fit_contribution = amp * dop_contribution[k];
						doppler->fit_overflow[idop+idop0] += fit_contribution;
						if (par->action == MAP && par->map_mode == MAPMODE_DELDOP)
							if (idop >= par->map_doplim[0] && idop <= par->map_doplim[1]) {
								frame->map_pos[x][y] += fit_contribution;
								c = pos->comp[x][y];
								f = pos->f[x][y];
								frame->map_facet_power[c][f] += fit_contribution;
								if (par->map_verbose)
									printf("# POS (%3d, %3d) comp %d facet %4d contributes %e to Dop (%3d)\n",
											x+pos->n, y+pos->n, c, f, fit_contribution, idop-1);
							}
					}
				}

			}  /* if cos(scattering angle) > 0 and POS pixel projects onto the right body */
		}  /* go to the next POS image pixel */

	/*  --- end loop over POS pixels ---  */

	/*  Convert the model's floating-point Doppler limits from
      floating-point bin numbers to Hz, and widen the limits
      to account for nonzero POS pixel width                  */

	frame->doplim[0] = (frame->doplim[0] - dopshift)*doppler->dop_per_bin
			- dop_extra;
	frame->doplim[1] = (frame->doplim[1] - dopshift)*doppler->dop_per_bin
			+ dop_extra;

	/*  Calculate the overflow contributions to chi squared:
      o2 = obs^2 contribution, m2 = model^2 contribution.

      Also compute the summed cross section and the mean Doppler bin
      for the overflow region, for use with the "delcorinit" action   */

	frame->overflow_o2 = 0.0;
	frame->overflow_m2 = 0.0;
	frame->overflow_xsec = 0.0;
	frame->overflow_dopmean = 0.0;
	sdev_sq = frame->sdev*frame->sdev;
	variance = sdev_sq;
	lookfact = (frame->nlooks > 0.0) ? 1.0/frame->nlooks : 0.0;
	if (any_overflow) {
		j1 = MAX( frame->idoplim[0] + idop0, 0);
		j2 = MIN( frame->idoplim[1] + idop0, MAXOVERFLOW - 1);
		for (j=j1; j<=j2; j++) {
			if (doppler->fit_overflow[j] != 0.0) {
				if (par->speckle)
					variance = sdev_sq + lookfact*doppler->fit_overflow[j]*doppler->fit_overflow[j];
				frame->overflow_o2 += 1.0;
				frame->overflow_m2 += doppler->fit_overflow[j]*doppler->fit_overflow[j]/variance;
				frame->overflow_xsec += doppler->fit_overflow[j];
				frame->overflow_dopmean += (j - idop0)*doppler->fit_overflow[j];
			}
		}
		if (frame->overflow_xsec != 0.0)
			frame->overflow_dopmean /= frame->overflow_xsec;

		/*  Print a warning if the model extends even beyond the overflow spectrum  */

		if ( ((frame->idoplim[0] + idop0) < 0) ||
				((frame->idoplim[1] + idop0) >= MAXOVERFLOW) ) {
			badradar = 1;
			dopfactor = (MAX( frame->idoplim[1] + idop0, MAXOVERFLOW)
					- MIN( frame->idoplim[0] + idop0, 0)         )
                						  / (1.0*MAXOVERFLOW);
			frame->badradar_logfactor += log(dopfactor);
			if (par->warn_badradar) {
				printf("\nWARNING in pos2doppler.c for set %2d frame %2d:\n", set, frm);
				printf("        model Doppler spectrum extends too far beyond the data spectrum\n");
				printf("             data:  bins %2d to %2d\n", 1, ndop);
				printf("            model:  bins %2d to %2d\n",
						frame->idoplim[0], frame->idoplim[1]);
				fflush(stdout);
			}
		}
	}
	return badradar;
}
