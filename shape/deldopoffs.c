/*****************************************************************************************
                                                                             deldopoffs.c

Takes the delay-correction polynomial for a delay-doppler set and figures out the COM
delay and doppler corrections (in units of image rows and columns) for each frame.

Modified 2015 June 3 by CM:
    Implement smearing for the "fit" and "write" actions

Modified 2006 June 21 by CM:
    Changed delres to del_per_pixel and dopres to dop_per_pixel
*****************************************************************************************/

#include "head.h"


void deldopoffs( struct deldop_t *deldop)
{
  int f, k, n;
  double del, dop, arg, x;

  for (f=0; f<(*deldop).nframes; f++)
    for (k=0; k<(*deldop).nviews; k++) {
      x = 1.0;
      dop = 0.0;
      del = (*deldop).delcor.a[0].val;
      arg = (*deldop).frame[f].view[k].t - (*deldop).delcor.t0;
      for (n=1; n<=(*deldop).delcor.n; n++) {
        dop += n*(*deldop).delcor.a[n].val*x;
        del += (*deldop).delcor.a[n].val*(x*=arg);
      }

      /*  del has units of usec  */

      (*deldop).frame[f].view[k].deloff = del/(*deldop).del_per_pixel;

      /*  dop has units of usec/day and there are 86400 sec/day  */

      (*deldop).frame[f].view[k].dopoff = -dop*(*deldop).Ftx
                                           / ((*deldop).dop_per_pixel*86400.0);
    }
}

void deldopoffs_MFS_initial( struct deldop_t *deldop)
{
  int f, k, n;
  double del, dop, arg, x;

  for (f=0; f<deldop->nframes; f++)
    for (k=0; k<deldop->nviews; k++) {
      x = 1.0;
      dop = 0.0;
      del = deldop->frame[f].delcor.a[0].val;
      arg = deldop->frame[f].view[k].t - deldop->frame[f].delcor.t0;
      for (n=1; n<=deldop->frame[f].delcor.n; n++) {
        dop += n*deldop->frame[f].delcor.a[n].val*x;
        del += deldop->frame[f].delcor.a[n].val*(x*=arg);
      }

      /*  del has units of usec  */
      deldop->frame[f].view[k].deloff = del/deldop->frame[f].del_per_pixel;

      /*  dop has units of usec/day and there are 86400 sec/day  */
      deldop->frame[f].view[k].dopoff = -dop*deldop->frame[f].Ftx
                                           / (deldop->frame[f].dop_per_pixel*86400.0);
    }
}
