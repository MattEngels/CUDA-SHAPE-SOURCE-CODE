/*****************************************************************************************
                                                                              penalties.c
 
Compute the penalty functions for all penalties being applied to this model

Modified 2014 February 15 by CM:
    Adjust several penalty functions to accommodate multiple radar and optical
        scattering laws within a mod file

Modified 2013 July 6 by CM:
    Add the "pa3tilt" penalty

Modified 2011 September 2 by CM:
    Add the "harmlambert" and "inholambert" cases for the "optalbdel" and "optalbvar"
        penalties

Modified 2011 August 22 by CM:
    Add the "impulse" penalty

Modified 2010 August 31 by CM:
    Fix bug in the "noncosine" penalty: forgot to assign value to n

Modified 2010 June 1 by CM:
    Revise the "rdev" and "maxrdev" penalties now that the "scalefactor"
        parameter is a 3-component vector rather than a scalar

Modified 2010 May 12 by CM:
    Revise the "bifurcation" penalty so that it's not sensitive to the
        exact positions of vertices that lie near zone boundaries

Modified 2010 April 27 by CM:
    Add the "noncosine" and "bifurcation" penalties

Modified 2009 November 15 by CM:
    Remove unused variable

Modified 2009 August 2 by CM:
    Adjust the nonsmooth and concavity penalties to pay attention to the
        "act" (active) flags of model sides rather than of model facets

Modified 2009 July 2 by CM:
    For various penalties, only sum over active facets/vertices/sides,
        thus excluding interior regions for multiple-component models
    For the "rdev" "maxrdev" and "maxellipdev" penalties, scale to the
        overall model's effective radius, not to the effective radius for
        each component
    For the "maxellipdev" penalty, compute deviations from the overall
        model's DEEVE, not from each component's DEEVE

Modified 2007 February 21 by CM:
    Add the "maxrdev" and "maxellipdev" penalties
    Fix bugs in "rdev" penalty, and change this penalty so that each
        vertex deviation is normalized to the effective radius of that
        component
    Change the "comdev" penalty so that the COM displacement is normalized
        to the model's effective radius

Modified 2005 September 7 by CM:
    Add the "harmlommel" "harmhapke" and "harmkaas" cases for the
        "optalbdel" and "optalbvar" penalties
    Add the "harmhapke" case for the "thetadel" and "thetavar" penalties
    Add the "harmcosine" case for the "radalbdel" "radalbvar" "rad_c_del"
        and "rad_c_var" penalties

Modified 2005 August 10 by CM:
    Add the "inhokaas" case for the "optalbdel" and "optalbvar" penalties

Modified 2005 July 20 by CM:
    Add the "thetadel" penalty for the "inhohapke" optical scattering law
    Add four penalties for the "inhocosine" radar scattering law:
        "radalbdel" "radalbvar" "rad_c_del" "rad_c_var"
    Don't display "changed negative penalty to 0.0" messages at all, since
        these situations are always due to slight roundoff error

Modified 2005 July 7 by CM:
    Don't display "changed negative penalty to 0.0" messages to the screen
        unless the par->showstate flag is turned on

Modified 2005 July 4 by CM:
    Adjust the structure for the "inholommel" optical scattering law
    Enable the "optalbdel" penalty for the "inhohapke" optical scattering
        law
    Protect against division by zero for "rdev" penalty

Modified 2005 March 8 by CM:
    Fix bug with negative penalty weights

Modified 2005 February 28 by CM:
    Eliminate checks that photometric parameters are valid, since these
        checks are now performed in routine realize_photo

Modified 2005 January 25 by CM:
    Initialize variable to avoid compilation warning

Modified 2004 May 4 by CM:
    Added "flattening" penalty

Modified 2004 April 25 by CM:
    Added "inertiadev_uni" and "nonpa_uni" penalties for PA rotators

Modified 2003 April 21 by CM:
    Added comments
    Protected against division by zero (for negative penalty weights)

Modified 2003 April 17 by CM:
    Removed the large code block for computing 0th, 1st, and 2nd-order
        moments, partly because it wasn't executed if none of the three
        penalties VOLUME and COMDEV and INERTIADEV were used, partly
        because it would be nice to know volumes and COM displacements
        and inertia tensors even when there's no need to evaluate
        penalties.  (Added this same code to function realize_mod.)
*****************************************************************************************/

#include "head.h"

#define RHOCUTOFF 1.0e-20
#define NBIFURCATION 20
#define NZONES (NBIFURCATION - 1)
#define TINYPEN 1.0e-10

double penalties( struct par_t *par, struct mod_t *mod, struct dat_t *dat)
{
  int i, c, s, ns, v, f, j, k, ntot, f1, f2, v1, v2, v3, got_pa, n, nv, jmax, j1, j2,
      k1, k2, ilaw;
  double pen, sum=0.0, a, b, adotb, av, av2, len, x, y, disp[3], pmoment[3],
         ap[3][3], b_over_c, volume, scale, radius[3], DEEVE_radius[3],
         vcoord[3], r, theta, phi, r_DEEVE, xtemp, ytemp, ztemp, r_eff,
         ymean, x2sum, y2sum, xysum, slope, intercept, resid2sum, rho,
         rho_fit, min_extent, max_extent, axis_increment, sumrho2[NBIFURCATION],
         nrho2[NBIFURCATION], meanrho2[NBIFURCATION], rho2, deepest_minimum,
         w1, w2;
  char name[80];
  struct vertices_t *real;

  /*  Initialize variable to avoid compilation warning  */

  v3 = 0;

  /*  Initialize other variables  */

  got_pa = 0;

  /*  Loop through the penalties and calculate each  */
  /*  contribution to the penalty-function sum       */

  for (i=1; i<=par->pen.n; i++) {

    pen = 0.0;
    switch (par->pen.type[i]) {
    case OPTALBDEL:

        /*  pen = (weighted mean over model "sides" [edges] of |albedo difference|
                             for the two facets sharing that side)
                  / (weighted mean over model "sides" of albedo sum for those facets)

            where the weighting factor is the length of the side  */

        strcpy( name, "optalbdel");
        a = b = 0.0;
        for (c=0; c<mod->shape.ncomp; c++) {
          for (s=0; s<mod->shape.comp[c].real.ns; s++) {
            f1 = mod->shape.comp[c].real.s[s].f[0];
            f2 = mod->shape.comp[c].real.s[s].f[1];
            if (mod->shape.comp[c].real.f[f1].act &&
                mod->shape.comp[c].real.f[f2].act    ) {
              v1 = mod->shape.comp[c].real.s[s].v[0];
              v2 = mod->shape.comp[c].real.s[s].v[1];
              len = distance( mod->shape.comp[c].real.v[v1].x,
                              mod->shape.comp[c].real.v[v2].x);
              for (ilaw=0; ilaw<mod->photo.noptlaws; ilaw++) {
                switch (mod->photo.opttype[ilaw]) {
                case HARMLAMBERT:
                case HARMLOMMEL:
                    x = mod->photo.optical[ilaw].harmR.local[c][f1].R.val;
                    y = mod->photo.optical[ilaw].harmR.local[c][f2].R.val;
                    a += len*fabs(x + y);
                    b += len*fabs(x - y);
                    break;
                case INHOLAMBERT:
                case INHOLOMMEL:
                    x = mod->photo.optical[ilaw].inhoR.local[c][f1].R.val;
                    y = mod->photo.optical[ilaw].inhoR.local[c][f2].R.val;
                    a += len*fabs(x + y);
                    b += len*fabs(x - y);
                    break;
                case HARMHAPKE:
                    x = mod->photo.optical[ilaw].harmhapke.local[c][f1].w.val;
                    y = mod->photo.optical[ilaw].harmhapke.local[c][f2].w.val;
                    a += len*fabs(x + y);
                    b += len*fabs(x - y);
                    break;
                case INHOHAPKE:
                    x = mod->photo.optical[ilaw].inhohapke.local[c][f1].w.val;
                    y = mod->photo.optical[ilaw].inhohapke.local[c][f2].w.val;
                    a += len*fabs(x + y);
                    b += len*fabs(x - y);
                    break;
                case HARMKAAS:
                    x = mod->photo.optical[ilaw].harmkaas.local[c][f1].R.val;
                    y = mod->photo.optical[ilaw].harmkaas.local[c][f2].R.val;
                    a += len*fabs(x + y);
                    b += len*fabs(x - y);
                    break;
                case INHOKAAS:
                    x = mod->photo.optical[ilaw].inhokaas.local[c][f1].R.val;
                    y = mod->photo.optical[ilaw].inhokaas.local[c][f2].R.val;
                    a += len*fabs(x + y);
                    b += len*fabs(x - y);
                    break;
                }
              }
            }
          }
        }
        if (a == 0.0)
          bailout("penalties.c: 'optalbdel' can't be used with this radar scattering law\n");
        pen = b/a;
        break;
    case OPTALBVAR:

        /*  pen = (facet albedo variance) / (mean facet albedo)^2  */

        strcpy( name, "optalbvar");
        ntot = 0;
        av = av2 = 0.0;
        for (c=0; c<mod->shape.ncomp; c++) {
          for (f=0; f<mod->shape.comp[c].real.nf; f++) {
            if (mod->shape.comp[c].real.f[f].act) {
              for (ilaw=0; ilaw<mod->photo.noptlaws; ilaw++) {
                switch (mod->photo.opttype[ilaw]) {
                case HARMLAMBERT:
                case HARMLOMMEL:
                    x = mod->photo.optical[ilaw].harmR.local[c][f].R.val;
                    av += x;
                    av2 += x*x;
                    ++ntot;
                    break;
                case INHOLAMBERT:
                case INHOLOMMEL:
                    x = mod->photo.optical[ilaw].inhoR.local[c][f].R.val;
                    av += x;
                    av2 += x*x;
                    ++ntot;
                    break;
                case HARMHAPKE:
                    x = mod->photo.optical[ilaw].harmhapke.local[c][f].w.val;
                    av += x;
                    av2 += x*x;
                    break;
                case INHOHAPKE:
                    x = mod->photo.optical[ilaw].inhohapke.local[c][f].w.val;
                    av += x;
                    av2 += x*x;
                    ++ntot;
                    break;
                case HARMKAAS:
                    x = mod->photo.optical[ilaw].harmkaas.local[c][f].R.val;
                    av += x;
                    av2 += x*x;
                    ++ntot;
                    break;
                case INHOKAAS:
                    x = mod->photo.optical[ilaw].inhokaas.local[c][f].R.val;
                    av += x;
                    av2 += x*x;
                    ++ntot;
                    break;
                }
              }
            }
          }
        }
        if (ntot == 0)
          bailout("penalties.c: 'optalbvar' can't be used with this radar scattering law\n");
        pen = ntot*(av2/(av*av)) - 1.0; /* fractional variance */
        if (pen < 0.0)
          pen = 0.0;    /* roundoff error */
        break;
    case THETADEL:

        /*  pen = (weighted mean over model "sides" [edges] of |slope angle difference|
                             for the two facets sharing that side)
                  / (weighted mean over model "sides" of slope angle sum for those facets)

            where the weighting factor is the length of the side

            More precisely, theta is the mean slope angle for intrafacet topographic
            roughness, so "mean facet slope angle" is actually "mean over all model
            facets of the mean slope angle for roughness within each facet," and
            similarly for the variance.                                               */

        strcpy( name, "thetadel");
        a = b = 0.0;
        for (c=0; c<mod->shape.ncomp; c++) {
          for (s=0; s<mod->shape.comp[c].real.ns; s++) {
            f1 = mod->shape.comp[c].real.s[s].f[0];
            f2 = mod->shape.comp[c].real.s[s].f[1];
            if (mod->shape.comp[c].real.f[f1].act &&
                mod->shape.comp[c].real.f[f2].act    ) {
              v1 = mod->shape.comp[c].real.s[s].v[0];
              v2 = mod->shape.comp[c].real.s[s].v[1];
              len = distance( mod->shape.comp[c].real.v[v1].x,
                              mod->shape.comp[c].real.v[v2].x);
              for (ilaw=0; ilaw<mod->photo.noptlaws; ilaw++) {
                switch (mod->photo.opttype[ilaw]) {
                case HARMHAPKE:
                    x = mod->photo.optical[ilaw].harmhapke.local[c][f1].theta.val;
                    y = mod->photo.optical[ilaw].harmhapke.local[c][f2].theta.val;
                    a += len*fabs(x + y);
                    b += len*fabs(x - y);
                    break;
                case INHOHAPKE:
                    x = mod->photo.optical[ilaw].inhohapke.local[c][f1].theta.val;
                    y = mod->photo.optical[ilaw].inhohapke.local[c][f2].theta.val;
                    a += len*fabs(x + y);
                    b += len*fabs(x - y);
                    break;
                }
              }
            }
          }
        }
        if (a == 0.0)
          bailout("penalties.c: 'thetadel' can't be used with this optical scattering law\n");
        pen = b/a;
        break;
    case THETAVAR:

        /*  pen = (facet slope angle variance) / (mean facet slope angle)^2

            More precisely, theta is the mean slope angle for intrafacet topographic
            roughness, so "mean facet slope angle" is actually "mean over all model
            facets of the mean slope angle for roughness within each facet," and
            similarly for the variance.                                               */

        strcpy( name, "thetavar");
        ntot = 0;
        av = av2 = 0.0;
        for (c=0; c<mod->shape.ncomp; c++) {
          for (f=0; f<mod->shape.comp[c].real.nf; f++) {
            if (mod->shape.comp[c].real.f[f].act) {
              for (ilaw=0; ilaw<mod->photo.noptlaws; ilaw++) {
                switch (mod->photo.opttype[ilaw]) {
                case HARMHAPKE:
                    x = mod->photo.optical[ilaw].harmhapke.local[c][f].theta.val;
                    av += x;
                    av2 += x*x;
                    ++ntot;
                    break;
                case INHOHAPKE:
                    x = mod->photo.optical[ilaw].inhohapke.local[c][f].theta.val;
                    av += x;
                    av2 += x*x;
                    ++ntot;
                    break;
                }
              }
            }
          }
        }
        if (ntot == 0)
          bailout("penalties.c: 'thetavar' can't be used with this optical scattering law\n");
        pen = ntot*(av2/(av*av)) - 1.0; /* fractional variance */
        if (pen < 0.0)
          pen = 0.0;    /* roundoff error */
        break;
    case RADALBDEL:

        /*  pen = (weighted mean over model "sides" [edges] of |albedo difference|
                             for the two facets sharing that side)
                  / (weighted mean over model "sides" of albedo sum for those facets)

            where the weighting factor is the length of the side  */

        strcpy( name, "radalbdel");
        a = b = 0.0;
        for (c=0; c<mod->shape.ncomp; c++) {
          for (s=0; s<mod->shape.comp[c].real.ns; s++) {
            f1 = mod->shape.comp[c].real.s[s].f[0];
            f2 = mod->shape.comp[c].real.s[s].f[1];
            if (mod->shape.comp[c].real.f[f1].act &&
                mod->shape.comp[c].real.f[f2].act    ) {
              v1 = mod->shape.comp[c].real.s[s].v[0];
              v2 = mod->shape.comp[c].real.s[s].v[1];
              len = distance( mod->shape.comp[c].real.v[v1].x,
                              mod->shape.comp[c].real.v[v2].x);
              for (ilaw=0; ilaw<mod->photo.nradlaws; ilaw++) {
                switch (mod->photo.radtype[ilaw]) {
                case HARMCOSINE_DIFF:
                    x = mod->photo.radar[ilaw].harmcosine.local[c][f1].R.val;
                    y = mod->photo.radar[ilaw].harmcosine.local[c][f2].R.val;
                    a += len*fabs(x + y);
                    b += len*fabs(x - y);
                    break;
                case INHOCOSINE_DIFF:
                    x = mod->photo.radar[ilaw].inhocosine.local[c][f1].R.val;
                    y = mod->photo.radar[ilaw].inhocosine.local[c][f2].R.val;
                    a += len*fabs(x + y);
                    b += len*fabs(x - y);
                    break;
                }
              }
            }
          }
        }
        if (a == 0.0)
          bailout("penalties.c: 'radalbdel' can't be used with this radar scattering law\n");
        pen = b/a;
        break;
    case RADALBVAR:

        /*  pen = (facet albedo variance) / (mean facet albedo)^2  */

        strcpy( name, "radalbvar");
        ntot = 0;
        av = av2 = 0.0;
        for (c=0; c<mod->shape.ncomp; c++) {
          for (f=0; f<mod->shape.comp[c].real.nf; f++) {
            if (mod->shape.comp[c].real.f[f].act) {
              for (ilaw=0; ilaw<mod->photo.nradlaws; ilaw++) {
                switch (mod->photo.radtype[ilaw]) {
                case HARMCOSINE_DIFF:
                    x = mod->photo.radar[ilaw].harmcosine.local[c][f].R.val;
                    av += x;
                    av2 += x*x;
                    ++ntot;
                    break;
                case INHOCOSINE_DIFF:
                    x = mod->photo.radar[ilaw].inhocosine.local[c][f].R.val;
                    av += x;
                    av2 += x*x;
                    ++ntot;
                    break;
                }
              }
            }
          }
        }
        if (ntot == 0)
          bailout("penalties.c: 'radalbvar' can't be used with this radar scattering law\n");
        pen = ntot*(av2/(av*av)) - 1.0; /* fractional variance */
        if (pen < 0.0)
          pen = 0.0;    /* roundoff error */
        break;
    case RAD_C_DEL:

        /*  pen = (weighted mean over model "sides" [edges] of |C difference|
                             for the two facets sharing that side)
                  / (weighted mean over model "sides" of C sum for those facets)

            where the weighting factor is the length of the side  */

        strcpy( name, "rad_c_del");
        a = b = 0.0;
        for (c=0; c<mod->shape.ncomp; c++) {
          for (s=0; s<mod->shape.comp[c].real.ns; s++) {
            f1 = mod->shape.comp[c].real.s[s].f[0];
            f2 = mod->shape.comp[c].real.s[s].f[1];
            if (mod->shape.comp[c].real.f[f1].act &&
                mod->shape.comp[c].real.f[f2].act    ) {
              v1 = mod->shape.comp[c].real.s[s].v[0];
              v2 = mod->shape.comp[c].real.s[s].v[1];
              len = distance( mod->shape.comp[c].real.v[v1].x,
                              mod->shape.comp[c].real.v[v2].x);
              for (ilaw=0; ilaw<mod->photo.nradlaws; ilaw++) {
                switch (mod->photo.radtype[ilaw]) {
                case HARMCOSINE_DIFF:
                    x = mod->photo.radar[ilaw].harmcosine.local[c][f1].C.val;
                    y = mod->photo.radar[ilaw].harmcosine.local[c][f2].C.val;
                    a += len*fabs(x + y);
                    b += len*fabs(x - y);
                    break;
                case INHOCOSINE_DIFF:
                    x = mod->photo.radar[ilaw].inhocosine.local[c][f1].C.val;
                    y = mod->photo.radar[ilaw].inhocosine.local[c][f2].C.val;
                    a += len*fabs(x + y);
                    b += len*fabs(x - y);
                    break;
                }
              }
            }
          }
        }
        if (a == 0.0)
          bailout("penalties.c: 'rad_c_del' can't be used with this radar scattering law\n");
        pen = b/a;
        break;
    case RAD_C_VAR:

        /*  pen = (facet C variance) / (mean facet C)^2  */

        strcpy( name, "rad_c_var");
        ntot = 0;
        av = av2 = 0.0;
        for (c=0; c<mod->shape.ncomp; c++) {
          for (f=0; f<mod->shape.comp[c].real.nf; f++) {
            if (mod->shape.comp[c].real.f[f].act) {
              for (ilaw=0; ilaw<mod->photo.nradlaws; ilaw++) {
                switch (mod->photo.radtype[ilaw]) {
                case HARMCOSINE_DIFF:
                    x = mod->photo.radar[ilaw].harmcosine.local[c][f].C.val;
                    av += x;
                    av2 += x*x;
                    ++ntot;
                    break;
                case INHOCOSINE_DIFF:
                    x = mod->photo.radar[ilaw].inhocosine.local[c][f].C.val;
                    av += x;
                    av2 += x*x;
                    ++ntot;
                    break;
                }
              }
            }
          }
        }
        if (ntot == 0)
          bailout("penalties.c: 'rad_c_var' can't be used with this radar scattering law\n");
        pen = ntot*(av2/(av*av)) - 1.0; /* fractional variance */
        if (pen < 0.0)
          pen = 0.0;    /* roundoff error */
        break;
    case NONCOSINE:

        /*  pen = mean squared residual about fit to cosine scattering law
                  (unweighted linear fit of log rho vs. log cos theta)

            The fit is done over the range 0 <= theta < 90 deg.  The point at 90 deg
            must be treated separately, since log(cos(90)) is undefined.  If it is
            less than the preceding rho value then the final point contributes nothing
            to the penalty function; if it is greater than the preceding value then
            the final point's squared residual is computed about the value predicted
            by the fit for the preceding point.                                         */

        strcpy( name, "noncosine");
        ntot = 0;
        resid2sum = 0.0;
        for (ilaw=0; ilaw<mod->photo.nradlaws; ilaw++) {
          switch (mod->photo.radtype[ilaw]) {
          case TABULARLAW:

              /*  Using the points with theta < 90 deg, compute the best-fit slope and
                  intercept and the sum of squared residuals.  The x-values and their
                  mean were computed by the read_mod routine, since they never change.

                  For trial rho values that are less than a tiny positive cutoff value,
                  avoid NaN errors by continuing the log(rho) curve toward negative rho
                  values as a straight line that matches the slope at the cutoff point   */

              n = mod->photo.radar[ilaw].tabular.n;
              ntot += n;
              ymean = 0.0;
              for (j=0; j<(n-1); j++) {
                rho = mod->photo.radar[ilaw].tabular.rho[j].val;
                if (rho > RHOCUTOFF)
                  par->pen.y_noncosine[j] = log10(rho);
                else
                  par->pen.y_noncosine[j] = log10(RHOCUTOFF) + (rho/RHOCUTOFF - 1)/LN10;
                ymean += par->pen.y_noncosine[j];
              }
              ymean /= (n - 1);
              x2sum = y2sum = xysum = 0.0;
              for (j=0; j<(n-1); j++) {
                x = par->pen.x_noncosine[j] - par->pen.xmean_noncosine;
                y = par->pen.y_noncosine[j] - ymean;
                x2sum += x*x;
                y2sum += y*y;
                xysum += x*y;
              }
              slope = xysum/x2sum;
              intercept = ymean - slope * par->pen.xmean_noncosine;
              resid2sum += y2sum - xysum*xysum/x2sum;

              /*  If the rho value for 90 deg is greater than for the next smallest angle,
                  add in its squared residual about the fit value for that smaller angle;
                  otherwise treat this final point as having zero residual                  */

              rho = mod->photo.radar[ilaw].tabular.rho[n-1].val;
              if (rho > mod->photo.radar[ilaw].tabular.rho[n-2].val) {
                rho_fit = slope * par->pen.x_noncosine[n-2] + intercept;
                if (rho > RHOCUTOFF)
                  y = log10(rho) - rho_fit;
                else
                  y = log10(RHOCUTOFF) + (rho/RHOCUTOFF - 1)/LN10 - rho_fit;
                resid2sum += y*y;
              }
              break;
          }
        }

        /*  Set the penalty function equal to the mean squared residual about the fit  */

        if (ntot == 0)
          bailout("penalties.c: 'noncosine' can't be used with this radar scattering law\n");
        pen = MAX( resid2sum/ntot, 0.0);
        break;
    case NONSMOOTH:

        /*  pen = mean over all "sides" (edges) of [1 - cos(theta)]^4

                  where theta = angle between the normals to the
                                two facets adjoining a given side

            Even an ellipsoid model has a nonzero "nonsmooth" penalty, but the
            high power (4) ensures that the penalty will be much greater if
            facet-scale topography is present.               

            Since this penalty depends on facet-scale roughness, it is smaller
            (for a given shape) when the model is subdivided into a larger
            number of smaller facets.  To be precise, when realizing an
            ellipsoid or harmonic model as a vertex model, the nonsmooth penalty
            is roughly proportional to 1/(number of vertices)^4.  One must
            adjust the "nonsmooth" penalty weight accordingly.                    */

        strcpy( name, "nonsmooth");
        ntot = 0;
        for (c=0; c<mod->shape.ncomp; c++) { /* visit each component */
          ns = mod->shape.comp[c].real.ns;
          for (s=0; s<ns; s++)
            if (mod->shape.comp[c].real.s[s].act) {
              x = 1 - dot( mod->shape.comp[c].real.f
                                  [mod->shape.comp[c].real.s[s].f[0] ].n,
                           mod->shape.comp[c].real.f
                                  [mod->shape.comp[c].real.s[s].f[1] ].n );
              pen += x*x*x*x;
              ntot++;
            }
        }
        pen /= ntot;
        break;
    case CONCAVITY:

        /*  pen = mean over all "sides" (edges) of the following quantity:

                      [1 - cos(theta)]^2  if side is "concave"
                              0           if side is "convex"

                      where theta = angle between the normals to the
                                    two facets adjoining a given side

            To determine whether or not a given side represents a concavity:

            Look at the two facets adjoining that side; construct a vector from one
            end of the side to the far vertex of facet 1; and take its dot product
            with the normal to facet 2.  If the dot product is positive, these two
            facets are tilted relative to each other in the concave sense.

            Note that while ellipsoids are convex-definite, the vertex realization
            of an ellipsoid model CAN have some shallow concavities.

            Since this penalty depends on facet-scale concavities, it is smaller
            (for a given shape) when the model is subdivided into a larger
            number of smaller facets.  To be precise, when realizing an
            ellipsoid or harmonic model as a vertex model, the concavity penalty
            is roughly proportional to 1/(number of vertices)^2.  One must
            adjust the "concavity" penalty weight accordingly.                       */

        strcpy( name, "concavity");
        ntot = 0;
        for (c=0; c<mod->shape.ncomp; c++) { /* visit each component */
          ns = mod->shape.comp[c].real.ns;
          for (s=0; s<ns; s++)
            if (mod->shape.comp[c].real.s[s].act) {
              f1 = mod->shape.comp[c].real.s[s].f[0];
              f2 = mod->shape.comp[c].real.s[s].f[1];
              v1 = mod->shape.comp[c].real.s[s].v[0];
              v2 = mod->shape.comp[c].real.s[s].v[1];
              for (j=0; j<=2; j++)
                if ( (mod->shape.comp[c].real.f[f1].v[j] != v1) &&
                     (mod->shape.comp[c].real.f[f1].v[j] != v2)    )
                  v3 = mod->shape.comp[c].real.f[f1].v[j];
              for (j=0; j<=2; j++)
                disp[j] = mod->shape.comp[c].real.v[v3].x[j] -
                          mod->shape.comp[c].real.v[v1].x[j];
              if (dot( disp, mod->shape.comp[c].real.f[f2].n) > 0.0) {
                x = 1 - dot( mod->shape.comp[c].real.f[f1].n,
                             mod->shape.comp[c].real.f[f2].n );
                pen += x*x;
              }
              ntot++;
            }
        }
        pen /= ntot;
        break;
    case RDEV:

        /*  pen = mean squared vertex deviation length
                  (where each vertex deviation is expressed
                  as a fraction of the model's effective radius)  */

        strcpy( name, "rdev");
        pen = 0.0;
        ntot = 0;

        /*  Get the model's effective radius  */

        volume = mod->shape.volume;
        r_eff = pow( 3*volume/(4*PIE), 1.0/3.0);

        for (c=0; c<mod->shape.ncomp; c++) { /* visit each component */
          if (mod->shape.comp[c].type == VERTEX) {

            /*  Loop through vertices to build up mean squared deviation  */

            for (v=0; v<mod->shape.comp[c].real.nv; v++) {
              if (mod->shape.comp[c].real.v[v].act) {
                scale = 0.0;
                for (j=0; j<=2; j++) {
                  y = mod->shape.comp[c].real.v[v].u[j] *
                      mod->shape.comp[c].real.scalefactor[j].val;
                  scale += y*y;
                }
                scale = sqrt(scale);
                x = scale * mod->shape.comp[c].real.v[v].r.val / r_eff;
                pen += x*x;
                ntot++;
              }
            }
          }
        }
        if (ntot == 0)
          bailout("penalties.c: need at least one vertex component for 'rdev'\n");
        pen /= ntot;
        break;
    case MAXRDEV:

        /*  pen = maximum squared vertex deviation length
                  (where each vertex deviation is expressed
                  as a fraction of the model's effective radius)  */

        strcpy( name, "maxrdev");
        pen = 0.0;

        /*  Get the model's effective radius  */

        volume = mod->shape.volume;
        r_eff = pow( 3*volume/(4*PIE), 1.0/3.0);

        for (c=0; c<mod->shape.ncomp; c++) { /* visit each component */
          if (mod->shape.comp[c].type == VERTEX) {

            /*  Loop through vertices to find maximum deviation  */

            for (v=0; v<mod->shape.comp[c].real.nv; v++) {
              if (mod->shape.comp[c].real.v[v].act) {
                scale = 0.0;
                for (j=0; j<=2; j++) {
                  y = mod->shape.comp[c].real.v[v].u[j] *
                      mod->shape.comp[c].real.scalefactor[j].val;
                  scale += y*y;
                }
                scale = sqrt(scale);
                x = scale * mod->shape.comp[c].real.v[v].r.val / r_eff;
                x *= x;
                pen = MAX( pen, x);
              }
            }
          }
        }
        break;
    case MAXELLIPDEV:

        /*  pen = maximum squared deviation from the model's DEEVE
                  (where each deviation is expressed as a fraction
                  of the model's effective radius)                  */

        strcpy( name, "maxellipdev");
        pen = 0.0;

        /*  Get the model's volume and effective radius  */

        volume = mod->shape.volume;
        r_eff = pow( 3*volume/(4*PIE), 1.0/3.0);

        /*  Diagonalize the inertia tensor to get pmoments, the principal
            moments of inertia, and ap, the transformation matrix taking us
            from principal-axis to body-fixed coordinates                    */

        if (!got_pa) {
          diag_inertia( mod->shape.inertia, pmoment, ap);
          got_pa = 1;
        }

        /*  Given a unit-density ellipsoid with volume V_ell and axis radii a, b, c
            along x, y, and z, respectively, the moment of inertia about the x-axis
            is (V_ell/5)*(b^2 + c^2), and similarly for the other two axes.  Hence
            the three parameters below are the three radii of an ellipsoid whose
            principal moments of inertia are the same as our model's.                 */

        radius[0] = sqrt( (5/volume) * (-pmoment[0] + pmoment[1] + pmoment[2]) / 2 );
        radius[1] = sqrt( (5/volume) * ( pmoment[0] - pmoment[1] + pmoment[2]) / 2 );
        radius[2] = sqrt( (5/volume) * ( pmoment[0] + pmoment[1] - pmoment[2]) / 2 );

        /*  Take those inertia ellipsoid radii and multiply them by the
            cube root of ( V_model / V_ell ): These are the DEEVE radii.  */

        scale = pow( 3*volume/(4*PIE*radius[0]*radius[1]*radius[2]), 1.0/3.0);
        for (j=0; j<=2; j++)
          DEEVE_radius[j] = scale*radius[j];

        /*  Loop through components and compute each vertex's deviation from
            the DEEVE, expressed as a fraction of the model's effective radius  */

        for (c=0; c<mod->shape.ncomp; c++) {

          /*  Loop through this component's vertices to get the maximum deviation  */

          for (v=0; v<mod->shape.comp[c].real.nv; v++) {
            if (mod->shape.comp[c].real.v[v].act) {

              /*  Transform this vertex's Cartesian body-fixed coordinates 
                  to spherical principal-axis coordinates                   */

              for (j=0; j<=2; j++)
                vcoord[j] = mod->shape.comp[c].real.v[v].x[j];
              cotrans( vcoord, ap, vcoord, -1);
              r = vecnorm( vcoord);
              theta = atan2(sqrt(vcoord[0]*vcoord[0] + vcoord[1]*vcoord[1]), vcoord[2]);
              phi = atan2(vcoord[1], vcoord[0]);

              /*  For these angular coordinates theta and phi, compute the
                  radial coordinate r_DEEVE of the point that lies on the DEEVE  */

              xtemp = sin(theta)*cos(phi)/DEEVE_radius[0];    /*  (x/a)/r  */
              ytemp = sin(theta)*sin(phi)/DEEVE_radius[1];    /*  (y/b)/r  */
              ztemp = cos(theta)/DEEVE_radius[2];             /*  (z/c)/r  */
              r_DEEVE = 1/sqrt(xtemp*xtemp + ytemp*ytemp + ztemp*ztemp);

              /*  Compute the difference between r and r_DEEVE, expressed as a
                  fraction of the model's effective radius, and then square it  */

              x = (r - r_DEEVE)/r_eff;
              x *= x;

              /*  The penalty is the maximum value of this quantity  */

              pen = MAX( pen, x);
            }
          }
        }
        break;
    case VOLUME:

        /*  pen = model's total volume (km^3)  */

        strcpy( name, "volume");
        pen = mod->shape.volume;
        break;
    case COMDEV:

        /*  pen = squared length of the center-of-mass displacement
                  divided by the square of the model's effective radius  */

        strcpy( name, "comdev");

        /*  Get the model's volume and effective radius  */

        volume = mod->shape.volume;
        r_eff = pow( 3*volume/(4*PIE), 1.0/3.0);

        /*  Compute the squared magnitude of the model's COM displacement  */
        for (k=0; k<=2; k++) {
          pen += mod->shape.com[k]*mod->shape.com[k];
        }

        pen /= (r_eff*r_eff);

        break;
    case INERTIADEV:

        /*
            pen = 1 - (dot product of two vectors described below):

            Create a vector using the diagonal elements of the inertia
            tensor (in body coordinates) as the three components; then
            divide this vector by the square root of the sum of squares
            of all nine tensor elements.  (The latter quantity appears as
            "sqrt(b)" below, and is invariant under rotation -- in
            particular, under transformation to principal-axis coordinates.)
            The resulting vector has unit length if the inertia tensor is
            diagonal, but is shorter otherwise.

            Create a second vector -- this one certainly a unit vector --
            by carrying out the same procedure with the three principal
            moments of inertia, treating them as the diagonal elements of
            a 3x3 diagonal matrix.  Here we use the three "spin.inertia"
            parameters listed in the "spin" section of the model file,
            NOT the principal moments we would obtain by diagonalizing the
            inertia tensor.  Since the inertia tensor was computed assuming
            uniform density, these two sets of principal moments will differ
            if the density is nonuniform.

            The resulting penalty function is zero iff these two vectors are
            identical -- that is, iff the inertia tensor is diagonal AND the
            model density is uniform.  It is positive for any other case.
            Hence this penalty pushes the model towards uniform density with
            principal axes remaining close to the body-coordinate axes.

            Note that this penalty is confined to the range [0, 2].

            Note also that the three "spin.inertia" moments are the parameters
            used in Euler's equations for evolving the spin state of NPA
            rotators.  The "inertiadev" penalty, then, is the only link between
            the model's principal moments and the model's *shape*.
        */ 

        strcpy( name, "inertiadev");
        a = b = adotb = 0.0;
        for (j=0; j<=2; j++) {
          a += mod->spin.inertia[j].val*mod->spin.inertia[j].val;
          adotb += mod->spin.inertia[j].val*mod->shape.inertia[j][j];
          for (k=0; k<=2; k++) {
            b += mod->shape.inertia[j][k]*mod->shape.inertia[j][k];
          }
        }
        pen = 1 - adotb/sqrt(a*b);
        break;
    case INERTIADEV_UNI:

        /*
            Same as the inertiadev penalty (see above), except that instead of
            comparing the inertia tensor (computed assuming uniform density)
            to the three "spin.inertia" principal moments, we compare the
            inertia tensor to the three diagonal elements of the diagonalized
            inertia tensor; this is appropriate for a principal-axis rotator,
            since the data can't constrain the spin.inertia principal moments
            (i.e., we have no choice but to assume uniform density).
        */ 

        strcpy( name, "inertiadev_uni");
        if (!got_pa) {

          /*  Diagonalize the inertia tensor to get pmoments, the principal
              moments of inertia, and ap, the transformation matrix taking us
              from principal-axis to body-fixed coordinates                    */

          diag_inertia( mod->shape.inertia, pmoment, ap);
          got_pa = 1;
        }
        a = b = adotb = 0.0;
        for (j=0; j<=2; j++) {
          a += pmoment[j]*pmoment[j];
          adotb += pmoment[j]*mod->shape.inertia[j][j];
          for (k=0; k<=2; k++) {
            b += mod->shape.inertia[j][k]*mod->shape.inertia[j][k];
          }
        }
        pen = 1 - adotb/sqrt(a*b);
        break;
    case PA3TILT:

        /*  pen = sin^2 (angle between third principal axis and body-fixed z-axis)

            pa3tilt is to be used for a principal-axis rotator instead of the
            inertiadev_uni penalty (see above) if we don't care how the first two
            principal axes are oriented relative to the body-fixed x and y axes
            but we still want to enforce the physical requirement that the third
            principal axis be parallel to the spin vector (i.e., to the body-fixed
            z-axis).  The principal axes are determined assuming uniform density.   */

        strcpy( name, "pa3tilt");
        if (!got_pa) {

          /*  Diagonalize the inertia tensor to get pmoments, the principal
              moments of inertia, and ap, the transformation matrix taking us
              from principal-axis to body-fixed coordinates                    */

          diag_inertia( mod->shape.inertia, pmoment, ap);
          got_pa = 1;
        }

        /*  ap[2][2] = cos(angle between PA3 and the body-fixed z-axis)  */

        pen = MAX( 0.0, 1 - ap[2][2]*ap[2][2]);
        break;
    case NONPA:

        /*  pen = MAX( 0, (0.01 + fraction by which the largest of
                                  the first two principal moments of
                                  inertia exceeds the third moment   ) )

            This penalty drives the first two principal moments
            to be at least 1% smaller than the third.                     */

        strcpy( name, "nonpa");

        /*  The 0.01 term below avoids biaxial inertia ellipsoids  */

        pen = MAX( (mod->spin.inertia[0].val/mod->spin.inertia[2].val) - 1,
                   (mod->spin.inertia[1].val/mod->spin.inertia[2].val) - 1 )
              + 0.01;
        if (pen < 0.0)
          pen = 0.0;
        break;
    case NONPA_UNI:

        /*
            Same as the nonpa penalty (see above), except that instead of
            comparing the three "spin.inertia" principal moments, we take
            the inertia tensor (computed assuming uniform density),
            diagonalize it, and compare the three diagonal elements; this is
            appropriate for a principal-axis rotator, since the data can't
            constrain the spin.inertia principal moments (i.e., we have no
            choice but to assume uniform density).
        */ 

        strcpy( name, "nonpa_uni");
        if (!got_pa) {

          /*  Diagonalize the inertia tensor to get pmoments, the principal
              moments of inertia, and ap, the transformation matrix taking us
              from principal-axis to body-fixed coordinates                    */

          diag_inertia( mod->shape.inertia, pmoment, ap);
          got_pa = 1;
        }

        /*  The 0.01 term below avoids biaxial inertia ellipsoids  */

        pen = MAX( (pmoment[0]/pmoment[2]) - 1, 
                   (pmoment[1]/pmoment[2]) - 1 )
              + 0.01;
        if (pen < 0.0)
          pen = 0.0;
        break;
    case EULEROFFS:

        /*  pen = mean squared *component* of angle and spin Euler
                  offsets lumped together, with component values
                  in degrees (angle) and degrees/day (spin).
                  That is, for nsets datasets, sum 6*nsets squared
                  component values and then divide by 6*nsets.      */

        strcpy( name, "euleroffs");
        pen = 0.0;
        ntot = 0;
        for (k=0; k<dat->nsets; k++)
          for (j=0; j<=2; j++) {
            pen += dat->set[k].angleoff[j].val*dat->set[k].angleoff[j].val;
            pen += dat->set[k].omegaoff[j].val*dat->set[k].omegaoff[j].val;
            ntot += 2;
          }
        pen *= R2D*R2D/ntot;
        break;
    case FLATTENING:

        /*
            Discourage flattening by setting penalty to (b/c - 1)^4,
            where b is the smaller of the first two DEEVE diameters
            and c is the third DEEVE diameter; both diameters are
            estimated via the inertia tensor computed under the
            assumption of uniform density.
        */ 

        strcpy( name, "flattening");
        if (!got_pa) {

          /*  Diagonalize the inertia tensor to get pmoments, the principal
              moments of inertia, and ap, the transformation matrix taking us
              from principal-axis to body-fixed coordinates                    */

            diag_inertia( mod->shape.inertia, pmoment, ap);
            got_pa = 1;
          }
          b_over_c = sqrt( MIN( (-pmoment[0] + pmoment[1] + pmoment[2]),
                                ( pmoment[0] - pmoment[1] + pmoment[2]) )
                           / ( pmoment[0] + pmoment[1] - pmoment[2])      );
          if (b_over_c > 1.0) {
              x = b_over_c - 1;
              pen = x*x*x*x;
          } else {
              pen = 0.0;
          }
          break;
    case BIFURCATION:

        /*
            Discourage bifurcation along the longest DEEVE diameter -- that is,
            along the axis corresponding to the smallest principal moment of
            inertia, computed under the assumption of uniform density

            Divide this axis into NBIFURCATION equal-width zones.  For each active
            vertex, compute the squared distance from the axis, then add this
            contribution to TWO adjacent zones (except at the model's ends)
            according to a 1 - x^6 "response function" whose "base" is two zones
            wide.  Then compute the mean squared distance S for each zone.  This
            procedure produces correlated S values for adjacent zones, but it
            prevents the penalty function from being sensitive to the exact
            positions of vertices that lie near zone boundaries.

            Now consider each possible set of three zones k1 < k < k2 for which the
            mean squared distance S for zone k is less than that for both k1 and k2.
            Find the deepest fractional minimum, that is, the minimum value of ratio
            r = S(k) / <mean of S(k1) and S(k2)>.  Set the penalty function equal to
            (1/r - 1)^2 (unless r > 1, in which case set it to zero).
        */ 

        strcpy( name, "bifurcation");
        if (!got_pa) {

          /*  Diagonalize the inertia tensor to get pmoments, the principal
              moments of inertia, and ap, the transformation matrix taking us
              from principal-axis to body-fixed coordinates                    */

          diag_inertia( mod->shape.inertia, pmoment, ap);
          got_pa = 1;
        }

        /*  Figure out which axis corresponds to the smallest principal moment  */

        if (pmoment[0] <= pmoment[1] && pmoment[0] <= pmoment[2])
          jmax = 0;
        else if (pmoment[1] <= pmoment[0] && pmoment[1] <= pmoment[2])
          jmax = 1;
        else
          jmax = 2;

        j1 = (jmax + 1) % 3;
        j2 = (jmax + 2) % 3;

        /*  Compute the model's maximum extent along this axis  */

        min_extent = HUGENUMBER;
        max_extent = -HUGENUMBER;
        for (c=0; c<mod->shape.ncomp; c++) {
          real = &mod->shape.comp[c].real;
          nv = real->nv;
          for (v=0; v<nv; v++) {
            min_extent = MIN( min_extent, real->v[v].x[jmax]);
            max_extent = MAX( max_extent, real->v[v].x[jmax]);
          }
        }
        axis_increment = (max_extent - min_extent + SMALLVAL) / NBIFURCATION;

        /*  Loop over all "active" (exterior) vertices of all model components,
            computing the mean value of the squared distance from the chosen axis
            in each of NZONES zones along that axis.  Except for vertices near
            the model's ends, each vertex contributes to TWO adjacent zones
            according to a 1 - x^6 "response function" whose "base" is two zones
            wide.  This implies fractional contributions and hence the "nrho2"
            vector is floating-point rather than integer.                          */

        for (k=0; k<NZONES; k++)
          sumrho2[k] = nrho2[k] = 0.0;

        for (c=0; c<mod->shape.ncomp; c++) {
          real = &mod->shape.comp[c].real;
          nv = real->nv;

          for (v=0; v<nv; v++)
            if (real->v[v].act) {
              rho2 = real->v[v].x[j1] * real->v[v].x[j1] +
                     real->v[v].x[j2] * real->v[v].x[j2];
              x = (real->v[v].x[jmax] - min_extent)/axis_increment;
              k = (int) floor(x);
              y = x - k;
              if (k == 0) {
                  k1 = k2 = 0;
                  w1 = 0.0;
                  w2 = 1.0;
              } else if (k == NZONES) {
                  k1 = k2 = NZONES - 1;
                  w1 = 1.0;
                  w2 = 0.0;
              } else {
                  k1 = k - 1;
                  k2 = k;
                  w1 = 1 - pow(y, 6);
                  w2 = 1 - pow(1-y, 6);
              }
              sumrho2[k1] += w1*rho2;
              sumrho2[k2] += w2*rho2;
              nrho2[k1] += w1;
              nrho2[k2] += w2;
            }
        }

        for (k=0; k<NZONES; k++)
          meanrho2[k] = sumrho2[k]/nrho2[k];

        /*  Look for the deepest fractional minimum in the mean squared distance
            from the longest axis: compare the value for a given zone k to the
            mean value for zones k1 and k2 on opposite sides of zone k, where the
            values for both zone k1 and zone k2 are greater than for zone k        */

        deepest_minimum = HUGENUMBER;

        for (k=1; k<NZONES-1; k++)
          for (k1=0; k1<k; k1++)
            if (meanrho2[k1] > meanrho2[k])
              for (k2=k+1; k2<NZONES; k2++)
                if (meanrho2[k2] > meanrho2[k])
                  deepest_minimum = MIN( deepest_minimum, 
                                         meanrho2[k]/((meanrho2[k1] + meanrho2[k2])/2));

        /*  Set penalty equal to (the reciprocal of this fractional minimum, minus 1)^2  */

        if (deepest_minimum < 1.0)
          pen = (1/deepest_minimum - 1)*(1/deepest_minimum - 1);
        else
          pen = 0.0;
        break;
    case IMPULSE:

        /*  pen = mean squared spin impulse *component* in degrees/day  */

        strcpy( name, "impulse");
        pen = 0.0;
        for (n=0; n<mod->spin.n_impulse; n++)
          for (j=0; j<=2; j++)
            pen += mod->spin.impulse[n][j].val * mod->spin.impulse[n][j].val;
        pen *= R2D*R2D/(3 * mod->spin.n_impulse);
        break;
    default:
        bailout("penalties.c: haven't coded for that penalty yet\n");
    }

    /*  A negative penalty weight yields the reciprocal of the usual
        penalty function, leading the corresponding model property
        to be pushed toward large rather than small values.           */

    if (par->pen.weight[i] >= 0.0)
      par->pen.base[i] = pen;
    else
      par->pen.base[i] = 1.0 / MAX( pen, TINYPEN);

    /*  Add this penalty to the penalty sum; if desired,
        display the penalty weight and value to the user.  */
    sum += fabs(par->pen.weight[i])*par->pen.base[i];
    if (par->showstate)
      printf("# %15s %e = fabs(%13.6e) * %e\n", name,
             fabs(par->pen.weight[i])*par->pen.base[i],
             par->pen.weight[i], par->pen.base[i]);
  }
  return sum;
}

#undef RHOCUTOFF
#undef NBIFURCATION
#undef NZONES
#undef TINYPEN
