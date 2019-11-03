/*****************************************************************************************
                                                                       realize_dopscale.c
  
Implements the '=' state for Doppler scaling factors, and also checks that the values of
all such factors lie within the allowed range

For each delay-Doppler or Doppler dataset whose Doppler scaling factor has the '=' state,
go backwards in the datafile until we find a delay-Doppler or Doppler dataset whose
Doppler scaling factor has state 'f' and/or 'c', and copy its value.

Since the "vary_dopscale" parameter permits Doppler scaling factors to be varied jointly
with shape/spin parameters that are being fit, the "dopscale_mode" parameter to
realize_dopscale determines how to handle this possibility:

    dopscale_mode = 0:
        Doppler scaling factors are not being varied jointly with shape/spin parameters,
        or else we aren't fitting a shape/spin parameter right now; just update each
        dataset's dopscale_save by setting it equal to that dataset's Doppler scaling
        factor, in case joint variation is needed later in the fit

    dopscale_mode = 1:
        Doppler scaling factors are being varied jointly with shape/spin parameters, and
        we're in the process of fitting some shape/spin parameter p (i.e., we're realizing
        the model for a trial value of p); set each dataset's Doppler scaling factor equal
        to the product of the corresponding dopscale_save and "dopscale_factor"

    dopscale_mode = 2:
        Doppler scaling factors are being varied jointly with shape/spin parameters, we've
        just obtained the best-fit value for shape/spin parameter p, and now we need to
        set the Doppler scaling factors to their best-fit values (i.e., to the values that
        "go with" the best-fit value of p); set each dataset's Doppler scaling factor
        equal to the product of the corresponding dopscale_save and "dopscale_factor,"
        then update dopscale_save by setting it equal to this same product

Modified 2016 July 7 by Matt Engels:
	Adapted for use in shape-cuda.

Written 2012 March 24 by CM, based on the "realize_delcor" routine for implementing the
    '=' state and on the "realize_photo" routine for checking for legal values
*****************************************************************************************/

#include "head.h"

void checkdopscale( double parval, double parmin, double parmax, int mode,
                    unsigned char *baddopscale, double *baddopscale_logfactor);


void realize_dopscale( struct par_t *par, struct dat_t *dat, double dopscale_factor,
                       int dopscale_mode)
{
  int s_dopscale, type_dopscale, s;

  /*  Initialize the flag for illegal Doppler scaling factor values  */

  (*par).baddopscale = 0;
  (*par).baddopscale_logfactor = 0.0;

  /*
      If a dataset has a Doppler scaling factor with state = '=', go backwards
      in the datafile until we reach a delay-Doppler or Doppler dataset whose
      Doppler scaling factor has state 'f' or 'c' rather than '='.

      s_dopscale is the number of the dataset we find.
      type_dopscale tells whether that dataset is delay-Doppler or Doppler.
  */

  s_dopscale = -1;
  type_dopscale = -1;

  for (s=0; s<(*dat).nsets; s++) {
    if ((*dat).set[s].type == DELAY) {
        if ((*dat).set[s].desc.deldop.dopscale.state != '=') {
            s_dopscale = s;
            type_dopscale = DELAY;
            if ((*dat).set[s].desc.deldop.dopscale.state == 'f') {
              if (dopscale_mode != 0)
                (*dat).set[s].desc.deldop.dopscale.val =
                     (*dat).set[s].desc.deldop.dopscale_save * dopscale_factor;
              if (dopscale_mode != 1)
                (*dat).set[s].desc.deldop.dopscale_save =
                             (*dat).set[s].desc.deldop.dopscale.val;
            }
            checkdopscale( (*dat).set[s].desc.deldop.dopscale.val,
                           (*par).dopscale_min, (*par).dopscale_max, 3,
                           &(*par).baddopscale, &(*par).baddopscale_logfactor);
        } else if (s_dopscale < 0) {
            bailout("can't use \"=\" state for the first (delay-)Doppler dataset\n");
        } else if (type_dopscale == DELAY) {
            (*dat).set[s].desc.deldop.dopscale.val =
                          (*dat).set[s_dopscale].desc.deldop.dopscale.val;
        } else {
            (*dat).set[s].desc.deldop.dopscale.val =
                          (*dat).set[s_dopscale].desc.doppler.dopscale.val;
        }
    } else if ((*dat).set[s].type == DOPPLER) {
        if ((*dat).set[s].desc.doppler.dopscale.state != '=') {
            s_dopscale = s;
            type_dopscale = DOPPLER;
            if ((*dat).set[s].desc.doppler.dopscale.state == 'f') {
              if (dopscale_mode != 0)
                (*dat).set[s].desc.doppler.dopscale.val =
                     (*dat).set[s].desc.doppler.dopscale_save * dopscale_factor;
              if (dopscale_mode != 1)
                (*dat).set[s].desc.doppler.dopscale_save =
                             (*dat).set[s].desc.doppler.dopscale.val;
            }
            checkdopscale( (*dat).set[s].desc.doppler.dopscale.val,
                           (*par).dopscale_min, (*par).dopscale_max, 3,
                           &(*par).baddopscale, &(*par).baddopscale_logfactor);
        } else if (s_dopscale < 0) {
            bailout("can't use \"=\" state for the first (delay-)Doppler dataset\n");
        } else if (type_dopscale == DELAY) {
            (*dat).set[s].desc.doppler.dopscale.val =
                          (*dat).set[s_dopscale].desc.deldop.dopscale.val;
        } else {
            (*dat).set[s].desc.doppler.dopscale.val =
                          (*dat).set[s_dopscale].desc.doppler.dopscale.val;
        }
    }
  }
}


void checkdopscale( double parval, double parmin, double parmax, int mode,
                    unsigned char *baddopscale, double *baddopscale_logfactor)
{

  /*  Flag Doppler scaling factor as bad if
           mode = 0:  parval <  parmin  or  parval >  parmax
           mode = 1:  parval <= parmin  or  parval >  parmax
           mode = 2:  parval <  parmin  or  parval >= parmax
           mode = 3:  parval <= parmin  or  parval >= parmax  */

  if (mode < 0 || mode > 3)
    bailout("realize_dopscale.c: checkdopscale mode must be between 0 and 3\n");

  if (mode == 0 || mode == 2) {
      if (parval < parmin) {
        *baddopscale = 1;
        *baddopscale_logfactor += log(1 + parmin - parval);
      }
  } else {
      if (parval <= parmin) {
        *baddopscale = 1;
        *baddopscale_logfactor += log(1 + parmin - parval);
      }
  }

  if (mode == 0 || mode == 1) {
      if (parval > parmax) {
        *baddopscale = 1;
        *baddopscale_logfactor += log(1 + parval - parmax);
      }
  } else {
      if (parval >= parmax) {
        *baddopscale = 1;
        *baddopscale_logfactor += log(1 + parval - parmax);
      }
  }

}
