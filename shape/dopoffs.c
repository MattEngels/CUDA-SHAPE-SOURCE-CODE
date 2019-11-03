/*****************************************************************************************
                                                                                dopoffs.c

Takes the delay-correction polynomial for a Doppler dataset and figures out the COM
Doppler corrections (in units of Doppler bins) for each frame.

Modified 2015 June 3 by CM:
    Implement smearing for the "fit" and "write" actions

Modified 2006 June 21 by CM:
    Changed dopres to dop_per_bin

Modified 2003 April 26 by CM:
    Removed delay computation
*****************************************************************************************/

#include "head.h"

void dopoffs( struct doppler_t *doppler)
{
  int f, k, n;
  double dop, arg, x;

  for (f=0; f<(*doppler).nframes; f++)
    for (k=0; k<(*doppler).nviews; k++) {
      x = 1.0;
      dop = 0.0;
      arg = (*doppler).frame[f].view[k].t - (*doppler).delcor.t0;
      for (n=1; n<=(*doppler).delcor.n; n++) {
        dop += n*(*doppler).delcor.a[n].val*x;
        x *= arg;
      }

      /*  dop has units of usec/day and there are 86400 sec/day  */

      (*doppler).frame[f].view[k].dopoff = -dop*(*doppler).Ftx
                                            / ((*doppler).dop_per_bin*86400.0);
    }
}
