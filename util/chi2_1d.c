/***************************************************************************

                                                                  chi2_1d.c

You give me two vectors o[imin..imax] and f[imin..imax], and a scalar or
vector of standard errors, s[imin..imax] or s[0] (flag stype tells
which).  I return the chi-square error.  If flag scale is set, I optimally
scale the f[] data.

Modified 2005 January 2005 by CM:
    Split a couple of complex statements into pairs of statements to
        avoid compilation warnings
***************************************************************************/

#include "basic.h"

double chi2_1d( double *o, double *f, int imin, int imax,
                double *s, int stype, int scale)
{
  double sc, num, den, err, tmp;
  int i;

  num = den = err = 0.0;
  if (scale) {
    if (stype == 0)				/* single sigma for all data */
        for (i=imin; i<=imax; i++) {
          num += o[i]*f[i];
          den += f[i]*f[i];
        }
    else
        for (i=imin; i<=imax; i++) {
          num += o[i]*f[i]*(sc = 1/(s[i]*s[i]));
          den += f[i]*f[i]*sc;
        }
    sc = num/den;
    for (i=imin; i<=imax; i++)
      f[i] *= sc;
  }

  if (stype == 0) {
      for (i=imin; i<=imax; i++) {
        tmp = o[i] - f[i];
        err += tmp*tmp;
      }
      err /= (s[0]*s[0]);
  } else {
      for (i=imin; i<=imax; i++) {
        tmp = (o[i] - f[i])/(s[i]*s[i]);
        err += tmp*tmp;
      }
  }

  return err;
}
