/*****************************************************************************************
                                                                                  hapke.c

Implements the bidirectional reflectance function for a rough surface as developed by
Hapke.  See Hapke, Bidirectional Reflectance Spectroscopy, Icarus 59, 41-59, 1984.

See also Chapter 12 of Hapke's book: Hapke, B., Theory of Reflectance and Emittance
Spectroscopy, New York: Cambridge U. Press, 1993

Arguments: cosi  = cos(incidence angle)
           cose  = cos(scattering angle)
           phase = solar phase angle (rad)
           w     = single-scattering albedo
           h     = dimensionless parameter roughly equal to double
                       the width (in radians) of the opposition surge
           B0    = amplitude of the opposition surge
           g     = asymmetry factor for Henyey-Greenstein phase function
           theta = mean slope angle for macroscopic roughness (rad)

Modified 2011 September 3 by CM:
    Use the improved approximation to the Chandrasekhar H function from Hapke 2002,
        Icarus, 157, 523-534
    Guard against division by zero (or by infinity) in various places
    Revise routines to require fewer calculations for the simple case where B0, g, and
        theta are zero

Modified 2007 March 18 by CM:
    Removed assignment statements within statements and otherwise made
        things more legible

Modified 2005 January 25 by CM:
    Split sin2psiovr2 computation into two statements to avoid
        compilation warnings

Modified 2004 March 25 by CM:
    Give phase rather than cos(phase) as argument
    Rename phase as "phase" (rather than "g") and the Henyey-Greenstein
        asymmetry factor as "g" (rather than "a")
    Added some header comments
*****************************************************************************************/

extern "C" {
#include "../shape/head.h"
}
#define TINYMU 1e-40

__device__ double dev_hapke( double cosi, double cose, double phase,
              double w, double h, double B0, double g, double theta)
{

  double sqrt1minw, r0, i, e, sini, psi, cospsi, sin2psiovr2, tmp, P, B, S, cosphase, pi,
         r, Hmu0p, Hmup, sine, E1e, E1i, E2e, E2i, f, C, cottheta, arg, mu0, mu,
         tantheta, mu0p, mup, mup0, mu0p0;

  pi = 4*atan(1.0);

  if ( w <= 0.0 || w >= 1.0 || h <= 0.0 || B0 < 0.0 || fabs(g) >= 1.0
                || theta < 0.0 || theta >= pi/2 || phase < 0.0 || phase > pi )
    return 0.0;

  mu0 = cosi;
  mu = cose;
  if (g != 0.0 || theta != 0.0)
    cosphase = cos(phase);
  else
    cosphase = -999.99;

  /*  Compute the bihemispherical reflectance of a semiinfinite medium
      of isotropic scatterers with single-scattering albedo w           */

  sqrt1minw = sqrt(1 - w);
  r0 = (1 - sqrt1minw)/(1 + sqrt1minw);

  /*  Evaluate the Henyey-Greenstein phase function  */

  if (g == 0.0)
    P = 1.0;
  else
    P = (1 - g*g)/pow(1 + 2*g*cosphase + g*g, 1.5);

  /*  Compute the strength of the opposition surge  */

  if (B0 == 0.0)
    B = 0.0;
  else
    B = B0/(1 + tan(phase/2.0)/h);

  /*  Correct for macroscopic roughness  */

  if (theta == 0.0) {

      /*  smooth surface  */

      mu0p = mu0;
      mup = mu;
      S = 1.0;

  } else {

      /*  rough surface  */

      i = acos(cosi);
      sini = sin(i);
      e = acos(cose);
      sine = sin(e);
      if (sini > 0.0 && sine > 0.0) {
          cospsi = (cosphase - cosi*cose)/(sini*sine);
          if (fabs(cospsi) > 1.0)
            cospsi /= fabs(cospsi);
      } else {
          cospsi = 1.0;
      }
      psi = acos(cospsi);
      tmp = sin(psi/2.0);
      sin2psiovr2 = tmp*tmp;

      tantheta = tan(theta);
      cottheta = 1/tantheta;
      f = (psi < pi) ? exp(-2.0*tan(psi/2.0)) : 0.0;
      C = 1/sqrt(1 + pi*tantheta*tantheta);

      /*  E functions  */

      if (e >= pi/2) {
          E1e = E2e = 1.0;
      } else if (e > 0.0) {
          arg = cottheta/tan(e);
          E1e = exp(-(2/pi)*arg);
          E2e = exp(-(1/pi)*arg*arg);
      } else {
          E1e = E2e = 0.0;
      }
      if (i >= pi/2) {
          E1i = E2i = 1.0;
      } else if (i > 0.0) {
          arg = cottheta/tan(i);
          E1i = exp(-(2/pi)*arg);
          E2i = exp(-(1/pi)*arg*arg);
      } else {
          E1i = E2i = 0.0;
      }

      mu0p0 = C*(mu0 + sini*tantheta*E2i/(2 - E1i));
      mup0 = C*(mu + sine*tantheta*E2e/(2 - E1e));
      if (i < e) {
          mu0p = C*(mu0 + sini*tantheta
                          * (cospsi*E2e + sin2psiovr2*E2i)/(2 - E1e - (psi/pi)*E1i));
          mup = C*(mu + sine*tantheta
                        * (E2e - sin2psiovr2*E2i)/(2 - E1e - (psi/pi)*E1i));
          S = ((mup*mu0)/(mup0*mu0p0))*C / (1 - f*(1 - C*mu0/mu0p0));
      } else {
          mu0p = C*(mu0 + sini*tantheta
                          * (E2i - sin2psiovr2*E2e)/(2 - E1i - (psi/pi)*E1e));
          mup = C*(mu + sine*tantheta
                        * (cospsi*E2i + sin2psiovr2*E2e)/(2 - E1i - (psi/pi)*E1e));
          S = ((mup*mu0)/(mup0*mu0p0))*C / (1 - f*(1 - C*mu/mup0));
      }
  }

  /*  Evaluate Hapke's analytical approximation to the Chandrasekhar H functions
      for the incidence and scattering angles as corrected for macroscopic roughness;
      this is the improved approximation given by Hapke 2002, Icarus, 157, 523-534     */

  if (mu0p > TINYMU)
    Hmu0p = 1/(1 - w*mu0p*(r0 + 0.5*(1 - 2*r0*mu0p)*log(1 + 1/mu0p)));
  else
    Hmu0p = 1.0;
  if (mup > TINYMU)
    Hmup = 1/(1 - w*mup*(r0 + 0.5*(1 - 2*r0*mup)*log(1 + 1/mup)));
  else
    Hmup = 1.0;

  /*  Compute the bidirectional reflectance  */

  r = ((w*mu0p)/(4*pi*(mu0p + mup))) * ( (1 + B)*P + Hmu0p*Hmup - 1.0 ) * S;

  if (r > 0.0)
    return r;
  else
    return 0.0;
}

#undef TINYMU
