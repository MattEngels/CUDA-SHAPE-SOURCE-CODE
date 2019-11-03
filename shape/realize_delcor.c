/***************************************************************************
                                                           realize_delcor.c
  
Implements the '=' state for delay correction polynomial coefficients:

For each delay-Doppler or Doppler dataset whose coefficients have the
'=' state, go backwards in the datafile until we find a delay-Doppler
or Doppler dataset whose coefficients have state 'f' and/or 'c', and
copy its coefficient values.

Setting zeroth-order coefficients is more complicated than setting other
coefficients, since the "vary_delcor0" fitting parameter, which permits
zeroth-order coefficients to be varied jointly with shape/spin parameters
that are being fit, might be turned on.  The "delcor0_mode" parameter to
realize_delcor determines how zeroth-order coefficients are handled:

    delcor0_mode = 0:
        Zeroth-order coefficients are not being varied jointly with
        shape/spin parameters, or else we aren't fitting a shape/spin
        parameter right now; just update each dataset's delcor0_save by
        setting it equal to that dataset's zeroth-order coefficient,
        in case joint variation is needed later in the fit

    delcor0_mode = 1:
        Zeroth-order coefficients are being varied jointly with shape/spin
        parameters, and we're in the process of fitting some shape/spin
        parameter p (i.e., we're realizing the model for a trial value
        of p); set each dataset's zeroth-order coefficient equal to the sum
        of the corresponding delcor0_save and "delta_delcor0"

    delcor0_mode = 2:
        Zeroth-order coefficients are being varied jointly with shape/spin
        parameters, we've just obtained the best-fit value for shape/spin
        parameter p, and now we need to set the zeroth-order coefficients
        to their best-fit values (i.e., to the values which "go with" the
        best-fit value of p); set each dataset's zeroth-order coefficient
        equal to the sum of the corresponding delcor0_save and
        "delta_delcor0," then update delcor0_save by setting it equal to
        this same sum

Written 2003 April 23 by CM

Modified 2006 October 1 by CM:
    Add "delta_delcor0" and "delcor0_mode" arguments
    Implement the "vary_delcor0" parameter
***************************************************************************/

#include "head.h"

void realize_delcor( struct dat_t *dat, double delta_delcor0, int delcor0_mode)
{
  int s_delcor, type_delcor, n_delcor, s, n, i;
  double t0_delcor, t0;

  /*
      If a dataset has delay correction polynomial coefficients with
      state = '=', go backwards in the datafile until we reach a
      delay-Doppler or Doppler dataset whose polynomial coefficients
      have states 'f' and/or 'c' rather than '='.

      s_delcor is the number of the dataset we find.
      type_delcor tells whether that dataset is delay-Doppler or Doppler.
      n_delcor is the number of coefficients in that dataset's polynomial.
      t0_delcor is the reference epoch for the polynomial.
  */

  s_delcor = -1;
  type_delcor = -1;
  n_delcor = -1;
  t0_delcor = -1.0;

  for (s=0; s<(*dat).nsets; s++) {

    if ((*dat).set[s].type == DELAY) {

        n = (*dat).set[s].desc.deldop.delcor.n;
        t0 = (*dat).set[s].desc.deldop.delcor.t0;
        if ((*dat).set[s].desc.deldop.delcor.a[0].state != '=') {
            s_delcor = s;
            type_delcor = DELAY;
            n_delcor = n;
            t0_delcor = t0;
            if ((*dat).set[s].desc.deldop.delcor.a[0].state == 'f') {
              if (delcor0_mode != 0)
                (*dat).set[s].desc.deldop.delcor.a[0].val =
                     (*dat).set[s].desc.deldop.delcor.delcor0_save + delta_delcor0;
              if (delcor0_mode != 1)
                (*dat).set[s].desc.deldop.delcor.delcor0_save =
                             (*dat).set[s].desc.deldop.delcor.a[0].val;
            }
        } else if (s_delcor < 0)
            bailout("can't use \"=\" state for the first delay polynomial\n");
        else if (n != n_delcor)
            bailout("delay polynomials must have same degree if state = \"=\"\n");
        else if (fabs(t0 - t0_delcor) > HALFSECOND)
            bailout("delay polynomials must have same t0 if state = \"=\"\n");
        else if (type_delcor == DELAY) {
            (*dat).set[s].desc.deldop.delcor.t0 = t0_delcor;
            for (i=0; i<=n; i++)
              (*dat).set[s].desc.deldop.delcor.a[i].val =
                            (*dat).set[s_delcor].desc.deldop.delcor.a[i].val;
        } else {
            (*dat).set[s].desc.deldop.delcor.t0 = t0_delcor;
            for (i=0; i<=n; i++)
              (*dat).set[s].desc.deldop.delcor.a[i].val =
                            (*dat).set[s_delcor].desc.doppler.delcor.a[i].val;
        }

        /*  Compute frame[*].deloff and frame[*].dopoff for this delay-Doppler
            dataset: the (floating-point) number of delay and Doppler bins
            corresponding to the delay correction polynomial at the epoch of
            each data frame.*/

        deldopoffs( &(*dat).set[s].desc.deldop);

    } else if ((*dat).set[s].type == DOPPLER) {

        n = (*dat).set[s].desc.doppler.delcor.n;
        t0 = (*dat).set[s].desc.doppler.delcor.t0;
        if ((*dat).set[s].desc.doppler.delcor.a[0].state != '=') {
            s_delcor = s;
            type_delcor = DOPPLER;
            n_delcor = n;
            t0_delcor = t0;
        } else if (s_delcor < 0)
            bailout("can't use \"=\" state for the first delay polynomial\n");
        else if (n != n_delcor)
            bailout("delay polynomials must have same degree if state = \"=\"\n");
        else if (fabs(t0 - t0_delcor) > HALFSECOND)
            bailout("delay polynomials must have same t0 if state = \"=\"\n");
        else if (type_delcor == DELAY) {
            (*dat).set[s].desc.deldop.delcor.t0 = t0_delcor;
            for (i=0; i<=n; i++)
              (*dat).set[s].desc.doppler.delcor.a[i].val =
                            (*dat).set[s_delcor].desc.deldop.delcor.a[i].val;
        } else {
            (*dat).set[s].desc.deldop.delcor.t0 = t0_delcor;
            for (i=0; i<=n; i++)
              (*dat).set[s].desc.doppler.delcor.a[i].val =
                            (*dat).set[s_delcor].desc.doppler.delcor.a[i].val;
        }

        /*  Compute frame[*].dopoff for this Doppler dataset:
            the (floating-point) number of Doppler bins corresponding to
            the delay correction polynomial at the epoch of each data frame.  */

        dopoffs( &(*dat).set[s].desc.doppler);

    }
  }
}
