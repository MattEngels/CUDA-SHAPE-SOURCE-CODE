/*****************************************************************************************
                                                                           realize_spin.c
  
Takes the initial spin state described in mod, computes the spin state at the epoch of
each data frame, and produces the various coordinate transformation matrices needed in
dat.  Also computes the total apparent spin vector at the epoch of each data frame.

Modified 2015 June 3 by CM:
    Implement smearing for the "fit" and "write" actions

Modified 2011 August 15 by CM:
    Determine which spin impulses must be applied to each frame or
        lightcurve point
    Pass the "int_abstol" parameter to the inteuler routine

Modified 2006 June 18 by CM:
    Eliminate range datasets

Modified 2005 January 20 by CM:
    For POS and range datasets, save the intrisic spin vector and total
        (intrinsic plus orbital) spin vector

Modified 2004 March 22 by CM:
    For lightcurve points, save the intrisic spin vector and total
        (intrinsic plus orbital) spin vector

Modified 2004 Feb 5 by CM:
    Implement "=" state for angle and spin offsets by creating
    routines realize_angleoff and realize_omegaoff

Modified 2003 May 4 by CM:
    Apply angle offsets to Doppler datasets, not just delay-Doppler
*****************************************************************************************/

#include "head.h"

void realize_impulse( struct spin_t spin, double t,
                      double t_integrate[], double impulse[][3], int *n_integrate);


void realize_spin( struct par_t *par, struct mod_t *mod, struct dat_t *dat)
{
  double anglesave[3], omegasave[3];
  int s, f, i, j, k;
  struct deldop_t *deldop;
  struct doppler_t *doppler;
  struct poset_t *poset;
  struct lghtcrv_t *lghtcrv;

  /*  Get the three components of the angle and spin offsets
      for all datasets, with any "=" states taken into account  */

  realize_angleoff( dat);
  realize_omegaoff( dat);

  /*  Determine the model spin state for each dataset in turn  */

  for (s=0; s<(*dat).nsets; s++) {

    /*  Add this dataset's angle offsets to the model Euler angles.
        (Later we'll add the spin offsets for each frame separately,
        after updating the intrinsic spin vector to each epoch.)
        Save the original Euler angles to be restored later.          */

    for (j=0; j<=2; j++) {
      anglesave[j] = (*mod).spin.angle[j].val;
      omegasave[j] = (*mod).spin.omega[j].val;
      (*mod).spin.angle[j].val += (*dat).set[s].angleoff[j].val;
    }

    switch ((*dat).set[s].type) {
    case DOPPLER:
        doppler = &(*dat).set[s].desc.doppler;

        /*  Loop through every view for every (smeared) frame  */

        for (f=0; f<(*doppler).nframes; f++)
          for (k=0; k<(*doppler).nviews; k++) {

            /*  Create lists of epochs and impulses, starting at initial spin epoch t0 and
                ending at this view's epoch t, that will be "encountered" in evolving the
                spin state from t0 to t, with the impulses negated if we're evolving
                backwards in time.  These lists will be used by the inteuler routine to
                break up the evolution of the spin state) into integrations over several
                smaller time intervals, punctuated by spin impulses.                        */

            realize_impulse( (*mod).spin, (*doppler).frame[f].view[k].t,
                             (*doppler).frame[f].t_integrate,
                             (*doppler).frame[f].impulse, &(*doppler).frame[f].n_integrate);

            /*  Integrate Euler's equations to get the model's intrinsic spin vector
                at the (light-time corrected) epoch of each view.  (*par).int_method
                tells inteuler which integration method to use.  If (*mod).spin.pa == 1,
                Euler's equations aren't used (principal-axis rotator).

                Input (*mod).spin is the initial spin specification given in the mod file.
                Output is frame[f].view[k].ae, the transformation matrix from ecliptic to
                body coordinates at epoch frame[f].view[k].t, and frame[f].view[k].intspin,
                the intrinsic spin vector (in body-fixed coordinates) at this epoch.         */

            inteuler( (*mod).spin, (*doppler).frame[f].t_integrate,
                      (*doppler).frame[f].impulse, (*doppler).frame[f].n_integrate,
                      (*doppler).frame[f].view[k].intspin, (*doppler).frame[f].view[k].ae,
                      (*mod).spin.pa, (*par).int_method, (*par).int_abstol);

            /*  Apply this dataset's spin offsets (also in body coordinates)
                to the intrinsic spin vector of this view.                    */

            for (j=0; j<=2; j++)
              (*doppler).frame[f].view[k].intspin[j] += (*dat).set[s].omegaoff[j].val;

            /*  Transform this view's intrinsic spin vector from body to
                ecliptic coordinates.  Note that the "-1" passed as the last
                argument to routine cotrans means that we multiply intspin
                by the transpose of matrix ae rather than by ae itself; thus
                we go from body to ecliptic coordinates rather than the other
                way around.                                                    */

            cotrans( (*doppler).frame[f].view[k].intspin, (*doppler).frame[f].view[k].ae,
                     (*doppler).frame[f].view[k].intspin, -1);

            /*  Add the intrinsic spin vector to the contribution (also in
                ecliptic coordinates) from plane-of-sky motion, thus giving
                us the total apparent spin vector in ecliptic coordinates.   */

            for (j=0; j<=2; j++)
              (*doppler).frame[f].view[k].spin[j] = (*doppler).frame[f].view[k].orbspin[j] +
                                                    (*doppler).frame[f].view[k].intspin[j];
          }
        break;

    case DELAY:

        /*  See "case DOPPLER" above for more extensive comments,
            since the Doppler and delay-Doppler procedures are identical.  */

        deldop = &(*dat).set[s].desc.deldop;
        for (f=0; f<(*deldop).nframes; f++)
          for (k=0; k<(*deldop).nviews; k++) {

            /*  Deal with spin impulses  */

            realize_impulse( (*mod).spin, (*deldop).frame[f].view[k].t,
                             (*deldop).frame[f].t_integrate,
                             (*deldop).frame[f].impulse, &(*deldop).frame[f].n_integrate);

            /*  Get the model's intrinsic spin vector (in body coordinates)
                at the (light-time corrected) epoch of each view.            */

            inteuler( (*mod).spin,
            		(*deldop).frame[f].t_integrate,
            		(*deldop).frame[f].impulse,
            		(*deldop).frame[f].n_integrate,
            		(*deldop).frame[f].view[k].intspin,
            		(*deldop).frame[f].view[k].ae,
                    (*mod).spin.pa,
                    (*par).int_method,
                    (*par).int_abstol);

            /*  Apply this dataset's spin offsets (also in body coordinates)
                to the intrinsic spin vector of this view.                    */

            for (j=0; j<=2; j++)
              (*deldop).frame[f].view[k].intspin[j] += (*dat).set[s].omegaoff[j].val;

            /*  Transform the intrinsic spin vector to ecliptic coordinates  */

            cotrans( (*deldop).frame[f].view[k].intspin, (*deldop).frame[f].view[k].ae,
                     (*deldop).frame[f].view[k].intspin, -1);

            /*  Now get the total apparent spin vector in ecliptic coordinates  */

            for (j=0; j<=2; j++)
              (*deldop).frame[f].view[k].spin[j] = (*deldop).frame[f].view[k].orbspin[j] +
                                                   (*deldop).frame[f].view[k].intspin[j];

//            printf("TEST");
          }
        break;

    case POS:

        /*  See "case DOPPLER" above for more extensive comments, since
            the Doppler and POS procedures are identical.*/

        poset = &(*dat).set[s].desc.poset;
        for (f=0; f<(*poset).nframes; f++)
          for (k=0; k<(*poset).nviews; k++) {

            /*  Deal with spin impulses*/

            realize_impulse( (*mod).spin, (*poset).frame[f].view[k].t,
                             (*poset).frame[f].t_integrate,
                             (*poset).frame[f].impulse, &(*poset).frame[f].n_integrate);

             /* Get the model's intrinsic spin vector (in body coordinates)
                at the (light-time corrected) epoch of each view.*/

            inteuler( (*mod).spin, (*poset).frame[f].t_integrate,
                      (*poset).frame[f].impulse, (*poset).frame[f].n_integrate,
                      (*poset).frame[f].view[k].intspin, (*poset).frame[f].view[k].ae,
                      (*mod).spin.pa, (*par).int_method, (*par).int_abstol);

             /* Apply this dataset's spin offsets (also in body coordinates)
                to the intrinsic spin vector of this view.*/

            for (j=0; j<=2; j++)
              (*poset).frame[f].view[k].intspin[j] += (*dat).set[s].omegaoff[j].val;

             /* Transform the intrinsic spin vector to ecliptic coordinates*/

            cotrans( (*poset).frame[f].view[k].intspin, (*poset).frame[f].view[k].ae,
                     (*poset).frame[f].view[k].intspin, -1);

             /* Now get the total apparent spin vector in ecliptic coordinates*/

            for (j=0; j<=2; j++)
              (*poset).frame[f].view[k].spin[j] = (*poset).frame[f].view[k].orbspin[j] +
                                                  (*poset).frame[f].view[k].intspin[j];
          }
        break;

    case LGHTCRV:

        /*  See "case DOPPLER" above for more extensive comments, since
            the procedure for each Doppler frame is identical to the 
            procedure for each calculated lightcurve point (except that
            calculated lightcurve points don't have multiple "views").*/
        lghtcrv = &(*dat).set[s].desc.lghtcrv;
        for (i=1; i<=(*lghtcrv).ncalc; i++) {

          /*  Deal with spin impulses*/

          realize_impulse( (*mod).spin, (*lghtcrv).x[i], (*lghtcrv).rend[i].t_integrate,
                           (*lghtcrv).rend[i].impulse, &(*lghtcrv).rend[i].n_integrate);

          /*  Get the model's intrinsic spin vector (in body coordinates)
              at the (light-time corrected) epoch of this lightcurve point.*/

          inteuler( (*mod).spin, (*lghtcrv).rend[i].t_integrate, (*lghtcrv).rend[i].impulse,
                    (*lghtcrv).rend[i].n_integrate, (*lghtcrv).rend[i].intspin,
                    (*lghtcrv).rend[i].ae, (*mod).spin.pa, (*par).int_method, (*par).int_abstol);

          /*  Apply this dataset's spin offsets (also in body coordinates)
              to the intrinsic spin vector of this point.*/

          for (j=0; j<=2; j++)
            (*lghtcrv).rend[i].intspin[j] += (*dat).set[s].omegaoff[j].val;

          /*  Transform the intrinsic spin vector to ecliptic coordinates*/

          cotrans( (*lghtcrv).rend[i].intspin, (*lghtcrv).rend[i].ae,
                   (*lghtcrv).rend[i].intspin, -1);

          /*  Now get the total apparent spin vector in ecliptic coordinates*/

          for (j=0; j<=2; j++)
            (*lghtcrv).rend[i].spin[j] = (*lghtcrv).rend[i].orbspin[j] +
                                         (*lghtcrv).rend[i].intspin[j];

//          if (s==6 && i==5) {
//        	  for (j=0; j<=2; j++)
//        		  printf("set[%i].lghtcrv.rend[%i].spin[%i], %3.8g\n", s, i, j, dat->set[s].desc.lghtcrv.rend[i].spin[j]);
//
//        	  for (int w=0; w<3; w++)
//        		  for (int x=0; x<3; x++)
//        			  printf("ae[%i][%i], %3.8g\n", w, x, lghtcrv->rend[i].ae[w][x]);
//          }
        }
        break;
    default:
        bailout("realize_spin: can't handle this type yet\n");
    }

    for (j=0; j<=2; j++) {
      (*mod).spin.angle[j].val = anglesave[j];
      (*mod).spin.omega[j].val = omegasave[j];
    }
  }
}


/*  realize_angleoff implements the '=' state
    for each component of the angle offset     */

void realize_angleoff( struct dat_t *dat)
{
  int j, s_angleoff, s;

  for (j=0; j<=2; j++) {

    /*
        If a dataset has state = '=' for component j of the angle offset,
        go backwards in the datafile until we reach a dataset for which
        component j of the angle offset has state 'f' or 'c' rather than '='.

        s_angleoff is the number of the dataset we find.
    */

    s_angleoff = -1;

    for (s=0; s<(*dat).nsets; s++) {
      if ((*dat).set[s].angleoff[j].state != '=')
        s_angleoff = s;
      else if (s_angleoff < 0)
        bailout("can't use \"=\" state for the first dataset's angle offsets\n");
      else
        (*dat).set[s].angleoff[j].val = (*dat).set[s_angleoff].angleoff[j].val;
    }
  }
}


/*  realize_omegaoff implements the '=' state
    for each component of the spin offset      */

void realize_omegaoff( struct dat_t *dat)
{
  int j, s_omegaoff, s;

  for (j=0; j<=2; j++) {

    /*
        If a dataset has state = '=' for component j of the spin offset,
        go backwards in the datafile until we reach a dataset for which
        component j of the spin offset has state 'f' or 'c' rather than '='.

        s_omegaoff is the number of the dataset we find.
    */

    s_omegaoff = -1;

    for (s=0; s<(*dat).nsets; s++) {
      if ((*dat).set[s].omegaoff[j].state != '=')
        s_omegaoff = s;
      else if (s_omegaoff < 0)
        bailout("can't use \"=\" state for the first dataset's spin offsets\n");
      else
        (*dat).set[s].omegaoff[j].val = (*dat).set[s_omegaoff].omegaoff[j].val;
    }
  }
}


/*  Figure out which spin impulses will be "encountered" in evolving the spin state
    from the initial spin epoch t0 to the epoch t of a particular frame or lightcurve
    point; then create lists of epochs and impulses, starting at t0 and ending at t,
    with the impulses negated if we're evolving backwards in time.                     */

void realize_impulse( struct spin_t spin, double t,
                      double t_integrate[], double impulse[][3], int *n_integrate)
{
  int k, j, n;

  k = 0;
  t_integrate[k] = spin.t0;
  for (j=0; j<=2; j++)
    impulse[k][j] = 0.0;
  if (t >= spin.t0) {

      /*  We'll be integrating forward in time, so add the spin impulses  */

      for (n=0; n<spin.n_impulse; n++) {
        if (spin.t_impulse[n] > spin.t0 && spin.t_impulse[n] <= t) {
          k++;
          t_integrate[k] = spin.t_impulse[n];
          for (j=0; j<=2; j++)
            impulse[k][j] = spin.impulse[n][j].val;
        }
      }
      if (t_integrate[k] < t) {
        k++;
        t_integrate[k] = t;
        for (j=0; j<=2; j++)
          impulse[k][j] = 0.0;
      }

  } else {

      /*  We'll be integrating backwards in time, so subtract the spin impulses  */

      for (n=spin.n_impulse-1; n>=0; n--) {
        if (spin.t_impulse[n] < spin.t0 && spin.t_impulse[n] >= t) {
          k++;
          t_integrate[k] = spin.t_impulse[n];
          for (j=0; j<=2; j++)
            impulse[k][j] = -spin.impulse[n][j].val;
        }
      }
      if (t_integrate[k] > t) {
        k++;
        t_integrate[k] = t;
        for (j=0; j<=2; j++)
          impulse[k][j] = 0.0;
      }
  }
  *n_integrate = k + 1;
}
