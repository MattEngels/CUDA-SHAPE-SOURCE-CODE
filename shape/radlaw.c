/*****************************************************************************************
                                                                                 radlaw.c

Calculates the differential radar scattering law, d(cross section)/d(area), which involves
R, the Fresnel reflectivity at normal incidence, and C, which is related to the r.m.s.
slope angle.  The expression for each scattering law includes an extra factor of
1/cos(scattering angle), because routines pos2deldop and pos2doppler will multiply the
return value by projected (plane-of-sky) area rather than by physical area along the
model's surface.

Modified 2014 February 10 by CM:
    Add "ilaw" parameter in order to implement multiple radar scattering laws

Modified 2010 April 27 by CM:
    Add "tabular" radar scattering law, with reflectivity tabulated at
        regular intervals in incidence angle and linear interpolation
        between those values

Modified 2007 August 4 by CM:
    Add "c" parameter (model component) and rename "facet" parameter as "f"

Modified 2006 September 1 by CM and MCN:
    For inhomogeneous laws, add check that "facet" parameter is
        nonnegative

Modified 2005 September 7 by CM:
    Add "harmcosine" radar scattering law

Modified 2005 July 20 by CM:
    Add "gaussian" and "hagfors" and "cosine_qs" and "gauss+cosine" and
        "hagfors+cosine" and "cosine+cosine" and inhomogeneous "inhocosine"
        radar scattering laws
    Add "facet" argument, which is needed for the "inhocosine" law

Modified 2005 March 1 by CM:
    Add NOLAW case

Modified 2005 January 25 by CM:
    Put return statement at end to avoid compilation warning
*****************************************************************************************/

#include "head.h"

double radlaw( struct photo_t *photo, int ilaw, double cosinc, int c, int f)
{
  int i;
  double tan2inc, sin2inc, hagforsarg, incidence, rho1, rho2, angres, angratio;
  double diffxsec_proj = -9.99;  /* dummy value */

  switch (photo->radtype[ilaw]) {
  case COSINELAW_DIFF:
      diffxsec_proj = photo->radar[ilaw].RC.R.val*(photo->radar[ilaw].RC.C.val + 1)
                      * pow( cosinc, 2*photo->radar[ilaw].RC.C.val - 1);
      break;
  case TABULARLAW:
      if (f < 0) {
          diffxsec_proj = 0.0;   /* blank sky */
      } else {
          incidence = acos(cosinc);
          angres = (PIE/2) / (photo->radar[ilaw].tabular.n - 1);
          angratio = incidence/angres;
          i = (int) floor(angratio);
          rho1 = photo->radar[ilaw].tabular.rho[i].val;
          rho2 = photo->radar[ilaw].tabular.rho[i+1].val;
          diffxsec_proj = (rho1 + (angratio - i)*(rho2 - rho1)) / cosinc;
      }
      break;
  case GAUSSIANLAW:
      if (cosinc >= photo->radar[ilaw].quasispec.cos_cutoff) {
          tan2inc = 1/(cosinc*cosinc) - 1;
          diffxsec_proj = photo->radar[ilaw].quasispec.R.val*photo->radar[ilaw].quasispec.C.val
                          * exp( -photo->radar[ilaw].quasispec.C.val * tan2inc)
                          / (cosinc*cosinc*cosinc*cosinc*cosinc);
      } else {
          diffxsec_proj = 0.0;
      }
      break;
  case HAGFORSLAW:
      if (cosinc >= photo->radar[ilaw].quasispec.cos_cutoff) {
          sin2inc = 1 - cosinc*cosinc;
          hagforsarg = cosinc*cosinc*cosinc*cosinc + photo->radar[ilaw].quasispec.C.val*sin2inc;
          diffxsec_proj = 0.5*photo->radar[ilaw].quasispec.R.val*photo->radar[ilaw].quasispec.C.val
                          * pow( hagforsarg, -1.5) / cosinc;
      } else {
          diffxsec_proj = 0.0;
      }
      break;
  case COSINELAW_QS:
      if (cosinc >= photo->radar[ilaw].quasispec.cos_cutoff)
        diffxsec_proj = photo->radar[ilaw].quasispec.R.val*(photo->radar[ilaw].quasispec.C.val + 1)
                        * pow( cosinc, 2*photo->radar[ilaw].quasispec.C.val - 1);
      else
        diffxsec_proj = 0.0;
      break;
  case GAUSSIAN_COSINE:
      diffxsec_proj = photo->radar[ilaw].hybrid.diff.R.val*(photo->radar[ilaw].hybrid.diff.C.val + 1)
                      * pow( cosinc, 2*photo->radar[ilaw].hybrid.diff.C.val - 1);
      if (cosinc >= photo->radar[ilaw].hybrid.qs.cos_cutoff) {
        tan2inc = 1/(cosinc*cosinc) - 1;
        diffxsec_proj += photo->radar[ilaw].hybrid.qs.R.val*photo->radar[ilaw].hybrid.qs.C.val
                         * exp( -photo->radar[ilaw].hybrid.qs.C.val * tan2inc)
                         / (cosinc*cosinc*cosinc*cosinc*cosinc);
      }
      break;
  case HAGFORS_COSINE:
      diffxsec_proj = photo->radar[ilaw].hybrid.diff.R.val*(photo->radar[ilaw].hybrid.diff.C.val + 1)
                      * pow( cosinc, 2*photo->radar[ilaw].hybrid.diff.C.val - 1);
      if (cosinc >= photo->radar[ilaw].hybrid.qs.cos_cutoff) {
        sin2inc = 1 - cosinc*cosinc;
        hagforsarg = cosinc*cosinc*cosinc*cosinc + photo->radar[ilaw].hybrid.qs.C.val*sin2inc;
        diffxsec_proj += 0.5*photo->radar[ilaw].hybrid.qs.R.val*photo->radar[ilaw].hybrid.qs.C.val
                         * pow( hagforsarg, -1.5) / cosinc;
      }
      break;
  case COSINE_COSINE:
      diffxsec_proj = photo->radar[ilaw].hybrid.diff.R.val*(photo->radar[ilaw].hybrid.diff.C.val + 1)
                      * pow( cosinc, 2*photo->radar[ilaw].hybrid.diff.C.val - 1);
      if (cosinc >= photo->radar[ilaw].hybrid.qs.cos_cutoff)
        diffxsec_proj += photo->radar[ilaw].hybrid.qs.R.val*(photo->radar[ilaw].hybrid.qs.C.val + 1)
                         * pow( cosinc, 2*photo->radar[ilaw].hybrid.qs.C.val - 1);
      break;
  case HARMCOSINE_DIFF:
      if (f < 0) {
          diffxsec_proj = 0.0;   /* blank sky */
      } else {
          diffxsec_proj = photo->radar[ilaw].harmcosine.local[c][f].R.val
                          * (photo->radar[ilaw].harmcosine.local[c][f].C.val + 1)
                          * pow( cosinc, 2*photo->radar[ilaw].harmcosine.local[c][f].C.val - 1);
      }
      break;
  case INHOCOSINE_DIFF:
      if (f < 0) {
          diffxsec_proj = 0.0;   /* blank sky */
      } else {
          diffxsec_proj = photo->radar[ilaw].inhocosine.local[c][f].R.val
                          * (photo->radar[ilaw].inhocosine.local[c][f].C.val + 1)
                          * pow( cosinc, 2*photo->radar[ilaw].inhocosine.local[c][f].C.val - 1);
      }
      break;
  case NOLAW:
      bailout("radlaw.c: can't set radar scattering law = \"none\" when radar data are used\n");
      break;
  default:
      bailout("radlaw.c: can't handle that radar scattering law yet\n");
  }

  return diffxsec_proj;
}
