/*****************************************************************************************
                                                                          realize_photo.c

Realize the model's radar and optical photometric scattering parameters, including those
set to the '=' state.

Setting radar and optical albedos is more complicated than setting other photometric
parameters, since the "vary_radalb" and "vary_optalb" fitting parameters, which permit
radar and optical albedos to be varied jointly with shape/spin parameters that are being
fit, might be turned on.  The "albedo_mode" parameter to realize_photo determines how
albedos are handled:

    albedo_mode = 0:
        Albedos are not being varied jointly with shape/spin parameters, or else we aren't
        fitting a shape/spin parameter right now; just update the "save" parameters (e.g.,
        R_save) by setting them equal to the corresponding albedo-related parameter values
        (e.g., R) in case joint variation is needed later in the fit

    albedo_mode = 1:
        Albedos are being varied jointly with shape/spin parameters, and we're in the
        process of fitting some shape/spin parameter p (i.e., we're realizing the model
        for a trial value of p); set each radar-albedo-related parameter equal to the
        corresponding "save" value multiplied by "radalb_factor" and set each
        optical-albedo-related parameter equal to the corresponding "save" value
        multiplied by "optalb_factor"

    albedo_mode = 2:
        Albedos are being varied jointly with shape/spin parameters, we've just obtained
        the best-fit value for shape/spin parameter p, and now we need to set the albedos
        to their best-fit values (i.e., to the values which "go with" the best-fit value
        of p); set each radar-albedo-related parameter equal to the corresponding "save"
        value multiplied by "radalb_factor" and set each optical-albedo-related parameter
        equal to the corresponding "save" value multiplied by "optalb_factor," then update
        the "save" parameters by setting them equal to these same products

Modified 2014 February 12 by CM:
    Implement multiple radar and optical scattering laws

Modified 2013 July 15 by CM:
    Bug fix: delete repeated statement in INHOCOSINE_DIFF block (dealing with R)

Modified 2011 September 7 by CM:
    Bug fix: don't vary any parameters with state = 'c'
    Add "harmlambert" and "inholambert" optical scattering laws

Modified 2010 April 27 by CM:
    Added "tabular" radar scattering law

Modified 2009 April 21 by CM:
    Fix a bug in the "checkphotopar" routine

Modified 2009 April 3 by CM:
    Initialize the "badphoto_logfactor" parameter and set its value if
        there are illegal values for photometric parameters
    Add the "checkphotopar" routine

Modified 2007 August 10 by CM:
    Initialize uninitialized variable

Modified 2006 October 1 by CM:
    Add "radalb_factor" "optalb_factor" and "albedo_mode" parameters in
        order to implement the "vary_radalb" and "vary_optalb" parameters

Modified 2006 June 18 by CM:
    Implement user-specified upper and lower limits to photometric
        parameters (rad_R_min, opt_w_max, etc.)

Modified 2005 September 8 by CM:
    Added "harmlommel" "harmhapke" and "harmkaas" optical scattering laws
    Added "harmcosine" radar scattering law

Modified 2005 August 8 by CM:
    Added "inhokaas" optical scattering law

Modified 2005 July 20 by CM:
    Added "gaussian" and "hagfors" and "cosine_qs" and "gauss+cosine" and
        "hagfors+cosine" and "cosine+cosine" and inhomogeneous "inhocosine"
        radar scattering laws

Modified 2005 July 4 by CM:
    Adjusted structure for the "inholommel" optical scattering law
    Enabled "inholommel" and "inhohapke" laws to be used for ellipsoid
        and harmonic model components, not just for vertex components

Modified 2005 February 28 by CM:
    Added checks for radar scattering law parameters
    Added NOLAW case for both optical and radar scattering laws
    Initialize the "badphoto" parameter (flag indicating illegal values
        for photometric parameters) to 0 here instead of in bestfit.c
        so that it can be used for actions other than "fit"
    Added a default case to switch statement, just to be safe

Modified 2004 May 8 by CM:
    Fixed treatment of inhomogeneous Lommel-Seeliger and Hapke laws

Modified 2004 April 29 by CM:
    For Kaasalainen scattering law, switch from weighting factor "c"
        (ranging from 0 to infinity) to "wt" (ranging from 0 to 1)

Modified 2004 March 25 by CM:
    Check for valid parameter values for Hapke optical scattering law.
        Indicate illegal values by setting the badphoto parameter to 1;
        this later will result in a doubled objective function for the
        model.

Modified 2004 February 26 by CM:
    Check for valid parameter values for Lambert, Lommel-Seeliger,
        geometric, and Kaasalainen "Lambert + Lommel-Seeliger"
        optical scattering laws.  Indicate illegal values by setting
        the badphoto parameter to 1; this later will result in a
        doubled objective function for the model.
    As a result, must now pass the model's parameter structure
        to realize_photo
*****************************************************************************************/

#include "head.h"

void checkphotopar( double parval, double parmin, double parmax, int mode,
                    unsigned char *badphoto, double *badphoto_logfactor);


void realize_photo( struct par_t *par, struct mod_t *mod,
                    double radalb_factor, double optalb_factor, int albedo_mode)
{
  int c, f, Lmax, L, l, m, i, n, ilaw;
  double **nlm, plm, **afactor, **bfactor=NULL, costheta, phi;

  /*  Initialize the flag for illegal photometric parameters  */

  par->badphoto = 0;
  par->badphoto_logfactor = 0.0;

  /*  Check that all optical scattering law parameters have legal values  */

  for (ilaw=0; ilaw<mod->photo.noptlaws; ilaw++) {
    switch (mod->photo.opttype[ilaw]) {
    case NOLAW:
        break;
    case GEOMETRICAL:
    case LAMBERTLAW:
    case LOMMEL:
        if (mod->photo.optical[ilaw].R.R.state == 'f') {
          if (albedo_mode != 0)
            mod->photo.optical[ilaw].R.R.val = mod->photo.optical[ilaw].R.R_save * optalb_factor;
          if (albedo_mode != 1)
            mod->photo.optical[ilaw].R.R_save = mod->photo.optical[ilaw].R.R.val;
        }
        checkphotopar( mod->photo.optical[ilaw].R.R.val,
                       par->opt_R_min, par->opt_R_max, 3,
                       &par->badphoto, &par->badphoto_logfactor);
        break;
    case HARMLAMBERT:
    case HARMLOMMEL:
        L = mod->photo.optical[ilaw].harmR.R.nhar;
        for (l=0; l<=L; l++) {
          if (mod->photo.optical[ilaw].harmR.R.a[l][0].state == 'f') {
            if (albedo_mode != 0)
              mod->photo.optical[ilaw].harmR.R.a[l][0].val
                       = mod->photo.optical[ilaw].harmR.R.a_save[l][0] * optalb_factor;
            if (albedo_mode != 1)
              mod->photo.optical[ilaw].harmR.R.a_save[l][0]
                       = mod->photo.optical[ilaw].harmR.R.a[l][0].val;
          }
          for (m=1; m<=l; m++) {
            if (mod->photo.optical[ilaw].harmR.R.a[l][m].state == 'f') {
              if (albedo_mode != 0)
                mod->photo.optical[ilaw].harmR.R.a[l][m].val
                         = mod->photo.optical[ilaw].harmR.R.a_save[l][m] * optalb_factor;
              if (albedo_mode != 1)
                mod->photo.optical[ilaw].harmR.R.a_save[l][m]
                         = mod->photo.optical[ilaw].harmR.R.a[l][m].val;
            }
            if (mod->photo.optical[ilaw].harmR.R.b[l][m].state == 'f') {
              if (albedo_mode != 0)
                mod->photo.optical[ilaw].harmR.R.b[l][m].val
                         = mod->photo.optical[ilaw].harmR.R.b_save[l][m] * optalb_factor;
              if (albedo_mode != 1)
                mod->photo.optical[ilaw].harmR.R.b_save[l][m]
                         = mod->photo.optical[ilaw].harmR.R.b[l][m].val;
            }
          }
        }

        nlm = matrix( 0, L, 0, L);
        for (l=0; l<=L; l++)
          for (m=0; m<=l; m++)
            nlm[l][m] = sqrt( (2*l+1) * exp(gammln(l-m+1.0) - gammln(l+m+1.0)) );
        afactor = matrix( 0, L, 0, L);
        if (L > 0)
          bfactor = matrix( 1, L, 1, L);

        for (c=0; c<mod->shape.ncomp; c++)
          for (f=0; f<mod->shape.comp[c].real.nf; f++) {
            costheta = cos( mod->shape.comp[c].real.f[f].theta);
            phi = mod->shape.comp[c].real.f[f].phi;
            for (l=0; l<=L; l++)
              for (m=0; m<=l; m++) {
                plm = nlm[l][m]*plgndr( l, m, costheta);
                afactor[l][m] = cos(m*phi)*plm;
                if (m > 0)
                  bfactor[l][m] = sin(m*phi)*plm;
              }

            L = mod->photo.optical[ilaw].harmR.R.nhar;
            mod->photo.optical[ilaw].harmR.local[c][f].R.val = 0.0;
            for (l=0; l<=L; l++) {
              mod->photo.optical[ilaw].harmR.local[c][f].R.val
                    += mod->photo.optical[ilaw].harmR.R.a[l][0].val
                       * afactor[l][0];
              for (m=1; m<=l; m++)
                mod->photo.optical[ilaw].harmR.local[c][f].R.val
                      += mod->photo.optical[ilaw].harmR.R.a[l][m].val
                         * afactor[l][m]
                         + mod->photo.optical[ilaw].harmR.R.b[l][m].val
                           * bfactor[l][m];
            }

            checkphotopar( mod->photo.optical[ilaw].harmR.local[c][f].R.val,
                           par->opt_R_min, par->opt_R_max, 3,
                           &par->badphoto, &par->badphoto_logfactor);
          }

        free_matrix( nlm, 0, L, 0, L);
        free_matrix( afactor, 0, L, 0, L);
        if (L > 0)
          free_matrix( bfactor, 1, L, 1, L);
        break;
    case INHOLAMBERT:
    case INHOLOMMEL:
        if (mod->photo.optical[ilaw].inhoR.global.R.state == 'f') {
          if (albedo_mode != 0)
            mod->photo.optical[ilaw].inhoR.global.R.val
                     = mod->photo.optical[ilaw].inhoR.global.R_save * optalb_factor;
          if (albedo_mode != 1)
            mod->photo.optical[ilaw].inhoR.global.R_save
                     = mod->photo.optical[ilaw].inhoR.global.R.val;
        }
        checkphotopar( mod->photo.optical[ilaw].inhoR.global.R.val,
                       par->opt_R_min, par->opt_R_max, 3,
                       &par->badphoto, &par->badphoto_logfactor);
        for (c=0; c<mod->shape.ncomp; c++)
          for (f=0; f<mod->shape.comp[c].real.nf; f++) {
            if (mod->photo.optical[ilaw].inhoR.local[c][f].R.state == '=') {
                mod->photo.optical[ilaw].inhoR.local[c][f].R.val
                      = mod->photo.optical[ilaw].inhoR.global.R.val;
            } else if (mod->photo.optical[ilaw].inhoR.local[c][f].R.state == 'f') {
                if (albedo_mode != 0)
                  mod->photo.optical[ilaw].inhoR.local[c][f].R.val
                           = mod->photo.optical[ilaw].inhoR.local[c][f].R_save
                             * optalb_factor;
                if (albedo_mode != 1)
                  mod->photo.optical[ilaw].inhoR.local[c][f].R_save
                           = mod->photo.optical[ilaw].inhoR.local[c][f].R.val;
            }
            checkphotopar( mod->photo.optical[ilaw].inhoR.local[c][f].R.val,
                           par->opt_R_min, par->opt_R_max, 3,
                           &par->badphoto, &par->badphoto_logfactor);
          }
        break;
    case HAPKE:
        if (mod->photo.optical[ilaw].hapke.w.state == 'f') {
          if (albedo_mode != 0)
            mod->photo.optical[ilaw].hapke.w.val = mod->photo.optical[ilaw].hapke.w_save
                                               * optalb_factor;
          if (albedo_mode != 1)
            mod->photo.optical[ilaw].hapke.w_save = mod->photo.optical[ilaw].hapke.w.val;
        }
        checkphotopar( mod->photo.optical[ilaw].hapke.w.val,
                       par->opt_w_min, par->opt_w_max, 3,
                       &par->badphoto, &par->badphoto_logfactor);
        checkphotopar( mod->photo.optical[ilaw].hapke.h.val,
                       par->opt_h_min, par->opt_h_max, 3,
                       &par->badphoto, &par->badphoto_logfactor);
        checkphotopar( mod->photo.optical[ilaw].hapke.B0.val,
                       par->opt_B0_min, par->opt_B0_max, 2,
                       &par->badphoto, &par->badphoto_logfactor);
        checkphotopar( mod->photo.optical[ilaw].hapke.g.val,
                       par->opt_g_min, par->opt_g_max, 3,
                       &par->badphoto, &par->badphoto_logfactor);
        checkphotopar( mod->photo.optical[ilaw].hapke.theta.val,
                       par->opt_theta_min, par->opt_theta_max, 2,
                       &par->badphoto, &par->badphoto_logfactor);
        break;
    case HARMHAPKE:
        L = mod->photo.optical[ilaw].harmhapke.w.nhar;
        for (l=0; l<=L; l++) {
          if (mod->photo.optical[ilaw].harmhapke.w.a[l][0].state == 'f') {
            if (albedo_mode != 0)
              mod->photo.optical[ilaw].harmhapke.w.a[l][0].val
                       = mod->photo.optical[ilaw].harmhapke.w.a_save[l][0] * optalb_factor;
            if (albedo_mode != 1)
              mod->photo.optical[ilaw].harmhapke.w.a_save[l][0]
                       = mod->photo.optical[ilaw].harmhapke.w.a[l][0].val;
          }
          for (m=1; m<=l; m++) {
            if (mod->photo.optical[ilaw].harmhapke.w.a[l][m].state == 'f') {
              if (albedo_mode != 0)
                mod->photo.optical[ilaw].harmhapke.w.a[l][m].val
                         = mod->photo.optical[ilaw].harmhapke.w.a_save[l][m] * optalb_factor;
              if (albedo_mode != 1)
                mod->photo.optical[ilaw].harmhapke.w.a_save[l][m]
                         = mod->photo.optical[ilaw].harmhapke.w.a[l][m].val;
            }
            if (mod->photo.optical[ilaw].harmhapke.w.b[l][m].state == 'f') {
              if (albedo_mode != 0)
                mod->photo.optical[ilaw].harmhapke.w.b[l][m].val
                         = mod->photo.optical[ilaw].harmhapke.w.b_save[l][m] * optalb_factor;
              if (albedo_mode != 1)
                mod->photo.optical[ilaw].harmhapke.w.b_save[l][m]
                         = mod->photo.optical[ilaw].harmhapke.w.b[l][m].val;
            }
          }
        }

        Lmax = MAX( mod->photo.optical[ilaw].harmhapke.w.nhar,
                    mod->photo.optical[ilaw].harmhapke.h.nhar);
        Lmax = MAX( Lmax, mod->photo.optical[ilaw].harmhapke.B0.nhar);
        Lmax = MAX( Lmax, mod->photo.optical[ilaw].harmhapke.g.nhar);
        Lmax = MAX( Lmax, mod->photo.optical[ilaw].harmhapke.theta.nhar);
        nlm = matrix( 0, Lmax, 0, Lmax);
        for (l=0; l<=Lmax; l++)
          for (m=0; m<=l; m++)
            nlm[l][m] = sqrt( (2*l+1) * exp(gammln(l-m+1.0) - gammln(l+m+1.0)) );
        afactor = matrix( 0, Lmax, 0, Lmax);
        if (Lmax > 0)
          bfactor = matrix( 1, Lmax, 1, Lmax);

        for (c=0; c<mod->shape.ncomp; c++)
          for (f=0; f<mod->shape.comp[c].real.nf; f++) {
            costheta = cos( mod->shape.comp[c].real.f[f].theta);
            phi = mod->shape.comp[c].real.f[f].phi;
            for (l=0; l<=Lmax; l++)
              for (m=0; m<=l; m++) {
                plm = nlm[l][m]*plgndr( l, m, costheta);
                afactor[l][m] = cos(m*phi)*plm;
                if (m > 0)
                  bfactor[l][m] = sin(m*phi)*plm;
              }

            L = mod->photo.optical[ilaw].harmhapke.w.nhar;
            mod->photo.optical[ilaw].harmhapke.local[c][f].w.val = 0.0;
            for (l=0; l<=L; l++) {
              mod->photo.optical[ilaw].harmhapke.local[c][f].w.val
                    += mod->photo.optical[ilaw].harmhapke.w.a[l][0].val
                       * afactor[l][0];
              for (m=1; m<=l; m++)
                mod->photo.optical[ilaw].harmhapke.local[c][f].w.val
                      += mod->photo.optical[ilaw].harmhapke.w.a[l][m].val
                         * afactor[l][m]
                         + mod->photo.optical[ilaw].harmhapke.w.b[l][m].val
                           * bfactor[l][m];
            }

            L = mod->photo.optical[ilaw].harmhapke.h.nhar;
            mod->photo.optical[ilaw].harmhapke.local[c][f].h.val = 0.0;
            for (l=0; l<=L; l++) {
              mod->photo.optical[ilaw].harmhapke.local[c][f].h.val
                    += mod->photo.optical[ilaw].harmhapke.h.a[l][0].val
                       * afactor[l][0];
              for (m=1; m<=l; m++)
                mod->photo.optical[ilaw].harmhapke.local[c][f].h.val
                      += mod->photo.optical[ilaw].harmhapke.h.a[l][m].val
                         * afactor[l][m]
                         + mod->photo.optical[ilaw].harmhapke.h.b[l][m].val
                           * bfactor[l][m];
            }

            L = mod->photo.optical[ilaw].harmhapke.B0.nhar;
            mod->photo.optical[ilaw].harmhapke.local[c][f].B0.val = 0.0;
            for (l=0; l<=L; l++) {
              mod->photo.optical[ilaw].harmhapke.local[c][f].B0.val
                    += mod->photo.optical[ilaw].harmhapke.B0.a[l][0].val
                       * afactor[l][0];
              for (m=1; m<=l; m++)
                mod->photo.optical[ilaw].harmhapke.local[c][f].B0.val
                      += mod->photo.optical[ilaw].harmhapke.B0.a[l][m].val
                         * afactor[l][m]
                         + mod->photo.optical[ilaw].harmhapke.B0.b[l][m].val
                           * bfactor[l][m];
            }

            L = mod->photo.optical[ilaw].harmhapke.g.nhar;
            mod->photo.optical[ilaw].harmhapke.local[c][f].g.val = 0.0;
            for (l=0; l<=L; l++) {
              mod->photo.optical[ilaw].harmhapke.local[c][f].g.val
                    += mod->photo.optical[ilaw].harmhapke.g.a[l][0].val
                       * afactor[l][0];
              for (m=1; m<=l; m++)
                mod->photo.optical[ilaw].harmhapke.local[c][f].g.val
                      += mod->photo.optical[ilaw].harmhapke.g.a[l][m].val
                         * afactor[l][m]
                         + mod->photo.optical[ilaw].harmhapke.g.b[l][m].val
                           * bfactor[l][m];
            }

            L = mod->photo.optical[ilaw].harmhapke.theta.nhar;
            mod->photo.optical[ilaw].harmhapke.local[c][f].theta.val = 0.0;
            for (l=0; l<=L; l++) {
              mod->photo.optical[ilaw].harmhapke.local[c][f].theta.val
                    += mod->photo.optical[ilaw].harmhapke.theta.a[l][0].val
                       * afactor[l][0];
              for (m=1; m<=l; m++)
                mod->photo.optical[ilaw].harmhapke.local[c][f].theta.val
                      += mod->photo.optical[ilaw].harmhapke.theta.a[l][m].val
                         * afactor[l][m]
                         + mod->photo.optical[ilaw].harmhapke.theta.b[l][m].val
                           * bfactor[l][m];
            }

            checkphotopar( mod->photo.optical[ilaw].harmhapke.local[c][f].w.val,
                           par->opt_w_min, par->opt_w_max, 3,
                           &par->badphoto, &par->badphoto_logfactor);
            checkphotopar( mod->photo.optical[ilaw].harmhapke.local[c][f].h.val,
                           par->opt_h_min, par->opt_h_max, 3,
                           &par->badphoto, &par->badphoto_logfactor);
            checkphotopar( mod->photo.optical[ilaw].harmhapke.local[c][f].B0.val,
                           par->opt_B0_min, par->opt_B0_max, 2,
                           &par->badphoto, &par->badphoto_logfactor);
            checkphotopar( mod->photo.optical[ilaw].harmhapke.local[c][f].g.val,
                           par->opt_g_min, par->opt_g_max, 3,
                           &par->badphoto, &par->badphoto_logfactor);
            checkphotopar( mod->photo.optical[ilaw].harmhapke.local[c][f].theta.val,
                           par->opt_theta_min, par->opt_theta_max, 2,
                           &par->badphoto, &par->badphoto_logfactor);
          }

        free_matrix( nlm, 0, Lmax, 0, Lmax);
        free_matrix( afactor, 0, Lmax, 0, Lmax);
        if (Lmax > 0)
          free_matrix( bfactor, 1, Lmax, 1, Lmax);
        break;
    case INHOHAPKE:
        if (mod->photo.optical[ilaw].inhohapke.global.w.state == 'f') {
          if (albedo_mode != 0)
            mod->photo.optical[ilaw].inhohapke.global.w.val
                     = mod->photo.optical[ilaw].inhohapke.global.w_save * optalb_factor;
          if (albedo_mode != 1)
            mod->photo.optical[ilaw].inhohapke.global.w_save
                     = mod->photo.optical[ilaw].inhohapke.global.w.val;
        }
        checkphotopar( mod->photo.optical[ilaw].inhohapke.global.w.val,
                       par->opt_w_min, par->opt_w_max, 3,
                       &par->badphoto, &par->badphoto_logfactor);
        checkphotopar( mod->photo.optical[ilaw].inhohapke.global.h.val,
                       par->opt_h_min, par->opt_h_max, 3,
                       &par->badphoto, &par->badphoto_logfactor);
        checkphotopar( mod->photo.optical[ilaw].inhohapke.global.B0.val,
                       par->opt_B0_min, par->opt_B0_max, 2,
                       &par->badphoto, &par->badphoto_logfactor);
        checkphotopar( mod->photo.optical[ilaw].inhohapke.global.g.val,
                       par->opt_g_min, par->opt_g_max, 3,
                       &par->badphoto, &par->badphoto_logfactor);
        checkphotopar( mod->photo.optical[ilaw].inhohapke.global.theta.val,
                       par->opt_theta_min, par->opt_theta_max, 2,
                       &par->badphoto, &par->badphoto_logfactor);
        for (c=0; c<mod->shape.ncomp; c++)
          for (f=0; f<mod->shape.comp[c].real.nf; f++) {
            if (mod->photo.optical[ilaw].inhohapke.local[c][f].w.state == '=') {
                mod->photo.optical[ilaw].inhohapke.local[c][f].w.val
                      = mod->photo.optical[ilaw].inhohapke.global.w.val;
            } else if (mod->photo.optical[ilaw].inhohapke.local[c][f].w.state == 'f') {
                if (albedo_mode != 0)
                  mod->photo.optical[ilaw].inhohapke.local[c][f].w.val
                           = mod->photo.optical[ilaw].inhohapke.local[c][f].w_save
                             * optalb_factor;
                if (albedo_mode != 1)
                  mod->photo.optical[ilaw].inhohapke.local[c][f].w_save
                           = mod->photo.optical[ilaw].inhohapke.local[c][f].w.val;
            }
            if (mod->photo.optical[ilaw].inhohapke.local[c][f].h.state == '=')
              mod->photo.optical[ilaw].inhohapke.local[c][f].h.val
                    = mod->photo.optical[ilaw].inhohapke.global.h.val;
            if (mod->photo.optical[ilaw].inhohapke.local[c][f].B0.state == '=')
              mod->photo.optical[ilaw].inhohapke.local[c][f].B0.val
                    = mod->photo.optical[ilaw].inhohapke.global.B0.val;
            if (mod->photo.optical[ilaw].inhohapke.local[c][f].g.state == '=')
              mod->photo.optical[ilaw].inhohapke.local[c][f].g.val
                    = mod->photo.optical[ilaw].inhohapke.global.g.val;
            if (mod->photo.optical[ilaw].inhohapke.local[c][f].theta.state == '=')
              mod->photo.optical[ilaw].inhohapke.local[c][f].theta.val
                    = mod->photo.optical[ilaw].inhohapke.global.theta.val;
            checkphotopar( mod->photo.optical[ilaw].inhohapke.local[c][f].w.val,
                           par->opt_w_min, par->opt_w_max, 3,
                           &par->badphoto, &par->badphoto_logfactor);
            checkphotopar( mod->photo.optical[ilaw].inhohapke.local[c][f].h.val,
                           par->opt_h_min, par->opt_h_max, 3,
                           &par->badphoto, &par->badphoto_logfactor);
            checkphotopar( mod->photo.optical[ilaw].inhohapke.local[c][f].B0.val,
                           par->opt_B0_min, par->opt_B0_max, 2,
                           &par->badphoto, &par->badphoto_logfactor);
            checkphotopar( mod->photo.optical[ilaw].inhohapke.local[c][f].g.val,
                           par->opt_g_min, par->opt_g_max, 3,
                           &par->badphoto, &par->badphoto_logfactor);
            checkphotopar( mod->photo.optical[ilaw].inhohapke.local[c][f].theta.val,
                           par->opt_theta_min, par->opt_theta_max, 2,
                           &par->badphoto, &par->badphoto_logfactor);
          }
        break;
    case KAASALAINEN:
        if (mod->photo.optical[ilaw].kaas.R.state == 'f') {
          if (albedo_mode != 0)
            mod->photo.optical[ilaw].kaas.R.val = mod->photo.optical[ilaw].kaas.R_save * optalb_factor;
          if (albedo_mode != 1)
            mod->photo.optical[ilaw].kaas.R_save = mod->photo.optical[ilaw].kaas.R.val;
        }
        checkphotopar( mod->photo.optical[ilaw].kaas.R.val,
                       par->opt_R_min, par->opt_R_max, 3,
                       &par->badphoto, &par->badphoto_logfactor);
        checkphotopar( mod->photo.optical[ilaw].kaas.wt.val,
                       par->opt_wt_min, par->opt_wt_max, 0,
                       &par->badphoto, &par->badphoto_logfactor);
        checkphotopar( mod->photo.optical[ilaw].kaas.A0.val,
                       par->opt_A0_min, par->opt_A0_max, 2,
                       &par->badphoto, &par->badphoto_logfactor);
        checkphotopar( mod->photo.optical[ilaw].kaas.D.val,
                       par->opt_D_min, par->opt_D_max, 3,
                       &par->badphoto, &par->badphoto_logfactor);
        checkphotopar( mod->photo.optical[ilaw].kaas.k.val,
                       par->opt_k_min, par->opt_k_max, 1,
                       &par->badphoto, &par->badphoto_logfactor);
        break;
    case HARMKAAS:
        L = mod->photo.optical[ilaw].harmkaas.R.nhar;
        for (l=0; l<=L; l++) {
          if (mod->photo.optical[ilaw].harmkaas.R.a[l][0].state == 'f') {
            if (albedo_mode != 0)
              mod->photo.optical[ilaw].harmkaas.R.a[l][0].val
                       = mod->photo.optical[ilaw].harmkaas.R.a_save[l][0] * optalb_factor;
            if (albedo_mode != 1)
              mod->photo.optical[ilaw].harmkaas.R.a_save[l][0]
                       = mod->photo.optical[ilaw].harmkaas.R.a[l][0].val;
          }
          for (m=1; m<=l; m++) {
            if (mod->photo.optical[ilaw].harmkaas.R.a[l][m].state == 'f') {
              if (albedo_mode != 0)
                mod->photo.optical[ilaw].harmkaas.R.a[l][m].val
                         = mod->photo.optical[ilaw].harmkaas.R.a_save[l][m] * optalb_factor;
              if (albedo_mode != 1)
                mod->photo.optical[ilaw].harmkaas.R.a_save[l][m]
                         = mod->photo.optical[ilaw].harmkaas.R.a[l][m].val;
            }
            if (mod->photo.optical[ilaw].harmkaas.R.b[l][m].state == 'f') {
              if (albedo_mode != 0)
                mod->photo.optical[ilaw].harmkaas.R.b[l][m].val
                         = mod->photo.optical[ilaw].harmkaas.R.b_save[l][m] * optalb_factor;
              if (albedo_mode != 1)
                mod->photo.optical[ilaw].harmkaas.R.b_save[l][m]
                         = mod->photo.optical[ilaw].harmkaas.R.b[l][m].val;
            }
          }
        }

        Lmax = MAX( mod->photo.optical[ilaw].harmkaas.R.nhar,
                    mod->photo.optical[ilaw].harmkaas.wt.nhar);
        Lmax = MAX( Lmax, mod->photo.optical[ilaw].harmkaas.A0.nhar);
        Lmax = MAX( Lmax, mod->photo.optical[ilaw].harmkaas.D.nhar);
        Lmax = MAX( Lmax, mod->photo.optical[ilaw].harmkaas.k.nhar);
        nlm = matrix( 0, Lmax, 0, Lmax);
        for (l=0; l<=Lmax; l++)
          for (m=0; m<=l; m++)
            nlm[l][m] = sqrt( (2*l+1) * exp(gammln(l-m+1.0) - gammln(l+m+1.0)) );
        afactor = matrix( 0, Lmax, 0, Lmax);
        if (Lmax > 0)
          bfactor = matrix( 1, Lmax, 1, Lmax);

        for (c=0; c<mod->shape.ncomp; c++)
          for (f=0; f<mod->shape.comp[c].real.nf; f++) {
            costheta = cos( mod->shape.comp[c].real.f[f].theta);
            phi = mod->shape.comp[c].real.f[f].phi;
            for (l=0; l<=Lmax; l++)
              for (m=0; m<=l; m++) {
                plm = nlm[l][m]*plgndr( l, m, costheta);
                afactor[l][m] = cos(m*phi)*plm;
                if (m > 0)
                  bfactor[l][m] = sin(m*phi)*plm;
              }

            L = mod->photo.optical[ilaw].harmkaas.R.nhar;
            mod->photo.optical[ilaw].harmkaas.local[c][f].R.val = 0.0;
            for (l=0; l<=L; l++) {
              mod->photo.optical[ilaw].harmkaas.local[c][f].R.val
                    += mod->photo.optical[ilaw].harmkaas.R.a[l][0].val
                       * afactor[l][0];
              for (m=1; m<=l; m++)
                mod->photo.optical[ilaw].harmkaas.local[c][f].R.val
                      += mod->photo.optical[ilaw].harmkaas.R.a[l][m].val
                         * afactor[l][m]
                         + mod->photo.optical[ilaw].harmkaas.R.b[l][m].val
                           * bfactor[l][m];
            }

            L = mod->photo.optical[ilaw].harmkaas.wt.nhar;
            mod->photo.optical[ilaw].harmkaas.local[c][f].wt.val = 0.0;
            for (l=0; l<=L; l++) {
              mod->photo.optical[ilaw].harmkaas.local[c][f].wt.val
                    += mod->photo.optical[ilaw].harmkaas.wt.a[l][0].val
                       * afactor[l][0];
              for (m=1; m<=l; m++)
                mod->photo.optical[ilaw].harmkaas.local[c][f].wt.val
                      += mod->photo.optical[ilaw].harmkaas.wt.a[l][m].val
                         * afactor[l][m]
                         + mod->photo.optical[ilaw].harmkaas.wt.b[l][m].val
                           * bfactor[l][m];
            }

            L = mod->photo.optical[ilaw].harmkaas.A0.nhar;
            mod->photo.optical[ilaw].harmkaas.local[c][f].A0.val = 0.0;
            for (l=0; l<=L; l++) {
              mod->photo.optical[ilaw].harmkaas.local[c][f].A0.val
                    += mod->photo.optical[ilaw].harmkaas.A0.a[l][0].val
                       * afactor[l][0];
              for (m=1; m<=l; m++)
                mod->photo.optical[ilaw].harmkaas.local[c][f].A0.val
                      += mod->photo.optical[ilaw].harmkaas.A0.a[l][m].val
                         * afactor[l][m]
                         + mod->photo.optical[ilaw].harmkaas.A0.b[l][m].val
                           * bfactor[l][m];
            }

            L = mod->photo.optical[ilaw].harmkaas.D.nhar;
            mod->photo.optical[ilaw].harmkaas.local[c][f].D.val = 0.0;
            for (l=0; l<=L; l++) {
              mod->photo.optical[ilaw].harmkaas.local[c][f].D.val
                    += mod->photo.optical[ilaw].harmkaas.D.a[l][0].val
                       * afactor[l][0];
              for (m=1; m<=l; m++)
                mod->photo.optical[ilaw].harmkaas.local[c][f].D.val
                      += mod->photo.optical[ilaw].harmkaas.D.a[l][m].val
                         * afactor[l][m]
                         + mod->photo.optical[ilaw].harmkaas.D.b[l][m].val
                           * bfactor[l][m];
            }

            L = mod->photo.optical[ilaw].harmkaas.k.nhar;
            mod->photo.optical[ilaw].harmkaas.local[c][f].k.val = 0.0;
            for (l=0; l<=L; l++) {
              mod->photo.optical[ilaw].harmkaas.local[c][f].k.val
                    += mod->photo.optical[ilaw].harmkaas.k.a[l][0].val
                       * afactor[l][0];
              for (m=1; m<=l; m++)
                mod->photo.optical[ilaw].harmkaas.local[c][f].k.val
                      += mod->photo.optical[ilaw].harmkaas.k.a[l][m].val
                         * afactor[l][m]
                         + mod->photo.optical[ilaw].harmkaas.k.b[l][m].val
                           * bfactor[l][m];
            }

            checkphotopar( mod->photo.optical[ilaw].harmkaas.local[c][f].R.val,
                           par->opt_R_min, par->opt_R_max, 3,
                           &par->badphoto, &par->badphoto_logfactor);
            checkphotopar( mod->photo.optical[ilaw].harmkaas.local[c][f].wt.val,
                           par->opt_wt_min, par->opt_wt_max, 0,
                           &par->badphoto, &par->badphoto_logfactor);
            checkphotopar( mod->photo.optical[ilaw].harmkaas.local[c][f].A0.val,
                           par->opt_A0_min, par->opt_A0_max, 2,
                           &par->badphoto, &par->badphoto_logfactor);
            checkphotopar( mod->photo.optical[ilaw].harmkaas.local[c][f].D.val,
                           par->opt_D_min, par->opt_D_max, 3,
                           &par->badphoto, &par->badphoto_logfactor);
            checkphotopar( mod->photo.optical[ilaw].harmkaas.local[c][f].k.val,
                           par->opt_k_min, par->opt_k_max, 1,
                           &par->badphoto, &par->badphoto_logfactor);
          }

        free_matrix( nlm, 0, Lmax, 0, Lmax);
        free_matrix( afactor, 0, Lmax, 0, Lmax);
        if (Lmax > 0)
          free_matrix( bfactor, 1, Lmax, 1, Lmax);
        break;
    case INHOKAAS:
        if (mod->photo.optical[ilaw].inhokaas.global.R.state == 'f') {
          if (albedo_mode != 0)
            mod->photo.optical[ilaw].inhokaas.global.R.val
                     = mod->photo.optical[ilaw].inhokaas.global.R_save * optalb_factor;
          if (albedo_mode != 1)
            mod->photo.optical[ilaw].inhokaas.global.R_save
                     = mod->photo.optical[ilaw].inhokaas.global.R.val;
        }
        checkphotopar( mod->photo.optical[ilaw].inhokaas.global.R.val,
                       par->opt_R_min, par->opt_R_max, 3,
                       &par->badphoto, &par->badphoto_logfactor);
        checkphotopar( mod->photo.optical[ilaw].inhokaas.global.wt.val,
                       par->opt_wt_min, par->opt_wt_max, 0,
                       &par->badphoto, &par->badphoto_logfactor);
        checkphotopar( mod->photo.optical[ilaw].inhokaas.global.A0.val,
                       par->opt_A0_min, par->opt_A0_max, 2,
                       &par->badphoto, &par->badphoto_logfactor);
        checkphotopar( mod->photo.optical[ilaw].inhokaas.global.D.val,
                       par->opt_D_min, par->opt_D_max, 3,
                       &par->badphoto, &par->badphoto_logfactor);
        checkphotopar( mod->photo.optical[ilaw].inhokaas.global.k.val,
                       par->opt_k_min, par->opt_k_max, 1,
                       &par->badphoto, &par->badphoto_logfactor);
        for (c=0; c<mod->shape.ncomp; c++)
          for (f=0; f<mod->shape.comp[c].real.nf; f++) {
            if (mod->photo.optical[ilaw].inhokaas.local[c][f].R.state == '=') {
                mod->photo.optical[ilaw].inhokaas.local[c][f].R.val
                      = mod->photo.optical[ilaw].inhokaas.global.R.val;
            } else if (mod->photo.optical[ilaw].inhokaas.local[c][f].R.state == 'f') {
                if (albedo_mode != 0)
                  mod->photo.optical[ilaw].inhokaas.local[c][f].R.val
                           = mod->photo.optical[ilaw].inhokaas.local[c][f].R_save
                             * optalb_factor;
                if (albedo_mode != 1)
                  mod->photo.optical[ilaw].inhokaas.local[c][f].R_save
                           = mod->photo.optical[ilaw].inhokaas.local[c][f].R.val;
            }
            if (mod->photo.optical[ilaw].inhokaas.local[c][f].wt.state == '=')
              mod->photo.optical[ilaw].inhokaas.local[c][f].wt.val
                    = mod->photo.optical[ilaw].inhokaas.global.wt.val;
            if (mod->photo.optical[ilaw].inhokaas.local[c][f].A0.state == '=')
              mod->photo.optical[ilaw].inhokaas.local[c][f].A0.val
                    = mod->photo.optical[ilaw].inhokaas.global.A0.val;
            if (mod->photo.optical[ilaw].inhokaas.local[c][f].D.state == '=')
              mod->photo.optical[ilaw].inhokaas.local[c][f].D.val
                    = mod->photo.optical[ilaw].inhokaas.global.D.val;
            if (mod->photo.optical[ilaw].inhokaas.local[c][f].k.state == '=')
              mod->photo.optical[ilaw].inhokaas.local[c][f].k.val
                    = mod->photo.optical[ilaw].inhokaas.global.k.val;
            checkphotopar( mod->photo.optical[ilaw].inhokaas.local[c][f].R.val,
                           par->opt_R_min, par->opt_R_max, 3,
                           &par->badphoto, &par->badphoto_logfactor);
            checkphotopar( mod->photo.optical[ilaw].inhokaas.local[c][f].wt.val,
                           par->opt_wt_min, par->opt_wt_max, 0,
                           &par->badphoto, &par->badphoto_logfactor);
            checkphotopar( mod->photo.optical[ilaw].inhokaas.local[c][f].A0.val,
                           par->opt_A0_min, par->opt_A0_max, 2,
                           &par->badphoto, &par->badphoto_logfactor);
            checkphotopar( mod->photo.optical[ilaw].inhokaas.local[c][f].D.val,
                           par->opt_D_min, par->opt_D_max, 3,
                           &par->badphoto, &par->badphoto_logfactor);
            checkphotopar( mod->photo.optical[ilaw].inhokaas.local[c][f].k.val,
                           par->opt_k_min, par->opt_k_max, 1,
                           &par->badphoto, &par->badphoto_logfactor);
          }
        break;
    default:
        bailout("realize_photo: can't handle this optical law yet\n");
    }
  }

  /*  Check that all radar scattering law parameters have legal values  */

  for (ilaw=0; ilaw<mod->photo.nradlaws; ilaw++) {
    switch (mod->photo.radtype[ilaw]) {
    case NOLAW:
        break;
    case COSINELAW_DIFF:
        if (mod->photo.radar[ilaw].RC.R.state == 'f') {
          if (albedo_mode != 0)
            mod->photo.radar[ilaw].RC.R.val = mod->photo.radar[ilaw].RC.R_save * radalb_factor;
          if (albedo_mode != 1)
            mod->photo.radar[ilaw].RC.R_save = mod->photo.radar[ilaw].RC.R.val;
        }
        checkphotopar( mod->photo.radar[ilaw].RC.R.val,
                       par->rad_R_min, par->rad_R_max, 3,
                       &par->badphoto, &par->badphoto_logfactor);
        checkphotopar( mod->photo.radar[ilaw].RC.C.val,
                       par->rad_C_min, par->rad_C_max, 3,
                       &par->badphoto, &par->badphoto_logfactor);
        break;
    case TABULARLAW:
        n = mod->photo.radar[ilaw].tabular.n;
        for (i=0; i<n; i++) {
          if (mod->photo.radar[ilaw].tabular.rho[i].state == '=') {
              mod->photo.radar[ilaw].tabular.rho[i].val = mod->photo.radar[ilaw].tabular.rho[i-1].val;
          } else if (mod->photo.radar[ilaw].tabular.rho[i].state == 'f') {
              if (albedo_mode != 0)
                mod->photo.radar[ilaw].tabular.rho[i].val = mod->photo.radar[ilaw].tabular.rho_save[i]
                                                        * radalb_factor;
              if (albedo_mode != 1)
                mod->photo.radar[ilaw].tabular.rho_save[i] = mod->photo.radar[ilaw].tabular.rho[i].val;
          }
          checkphotopar( mod->photo.radar[ilaw].tabular.rho[i].val,
                         par->rad_rho_min, par->rad_rho_max, 3,
                         &par->badphoto, &par->badphoto_logfactor);
        }
        break;
    case GAUSSIANLAW :
    case HAGFORSLAW  :
    case COSINELAW_QS:
        if (mod->photo.radar[ilaw].quasispec.R.state == 'f') {
          if (albedo_mode != 0)
            mod->photo.radar[ilaw].quasispec.R.val = mod->photo.radar[ilaw].quasispec.R_save
                                                 * radalb_factor;
          if (albedo_mode != 1)
            mod->photo.radar[ilaw].quasispec.R_save = mod->photo.radar[ilaw].quasispec.R.val;
        }
        checkphotopar( mod->photo.radar[ilaw].quasispec.R.val,
                       par->rad_R_min, par->rad_R_max, 3,
                       &par->badphoto, &par->badphoto_logfactor);
        checkphotopar( mod->photo.radar[ilaw].quasispec.C.val,
                       par->rad_C_min, par->rad_C_max, 3,
                       &par->badphoto, &par->badphoto_logfactor);
        break;
    case GAUSSIAN_COSINE:
    case HAGFORS_COSINE :
    case COSINE_COSINE  :
        if (mod->photo.radar[ilaw].hybrid.qs.R.state == 'f') {
          if (albedo_mode != 0)
            mod->photo.radar[ilaw].hybrid.qs.R.val = mod->photo.radar[ilaw].hybrid.qs.R_save
                                                 * radalb_factor;
          if (albedo_mode != 1)
            mod->photo.radar[ilaw].hybrid.qs.R_save = mod->photo.radar[ilaw].hybrid.qs.R.val;
        }
        if (mod->photo.radar[ilaw].hybrid.diff.R.state == 'f') {
          if (albedo_mode != 0)
            mod->photo.radar[ilaw].hybrid.diff.R.val = mod->photo.radar[ilaw].hybrid.diff.R_save
                                                   * radalb_factor;
          if (albedo_mode != 1)
            mod->photo.radar[ilaw].hybrid.diff.R_save = mod->photo.radar[ilaw].hybrid.diff.R.val;
        }
        checkphotopar( mod->photo.radar[ilaw].hybrid.qs.R.val,
                       par->rad_R_min, par->rad_R_max, 3,
                       &par->badphoto, &par->badphoto_logfactor);
        checkphotopar( mod->photo.radar[ilaw].hybrid.qs.C.val,
                       par->rad_C_min, par->rad_C_max, 3,
                       &par->badphoto, &par->badphoto_logfactor);
        checkphotopar( mod->photo.radar[ilaw].hybrid.diff.R.val,
                       par->rad_R_min, par->rad_R_max, 3,
                       &par->badphoto, &par->badphoto_logfactor);
        checkphotopar( mod->photo.radar[ilaw].hybrid.diff.C.val,
                       par->rad_C_min, par->rad_C_max, 3,
                       &par->badphoto, &par->badphoto_logfactor);
        break;
    case HARMCOSINE_DIFF:
        L = mod->photo.radar[ilaw].harmcosine.R.nhar;
        for (l=0; l<=L; l++) {
          if (mod->photo.radar[ilaw].harmcosine.R.a[l][0].state == 'f') {
            if (albedo_mode != 0)
              mod->photo.radar[ilaw].harmcosine.R.a[l][0].val
                       = mod->photo.radar[ilaw].harmcosine.R.a_save[l][0] * radalb_factor;
            if (albedo_mode != 1)
              mod->photo.radar[ilaw].harmcosine.R.a_save[l][0]
                       = mod->photo.radar[ilaw].harmcosine.R.a[l][0].val;
          }
          for (m=1; m<=l; m++) {
            if (mod->photo.radar[ilaw].harmcosine.R.a[l][m].state == 'f') {
              if (albedo_mode != 0)
                mod->photo.radar[ilaw].harmcosine.R.a[l][m].val
                         = mod->photo.radar[ilaw].harmcosine.R.a_save[l][m] * radalb_factor;
              if (albedo_mode != 1)
                mod->photo.radar[ilaw].harmcosine.R.a_save[l][m]
                         = mod->photo.radar[ilaw].harmcosine.R.a[l][m].val;
            }
            if (mod->photo.radar[ilaw].harmcosine.R.b[l][m].state == 'f') {
              if (albedo_mode != 0)
                mod->photo.radar[ilaw].harmcosine.R.b[l][m].val
                         = mod->photo.radar[ilaw].harmcosine.R.b_save[l][m] * radalb_factor;
              if (albedo_mode != 1)
                mod->photo.radar[ilaw].harmcosine.R.b_save[l][m]
                         = mod->photo.radar[ilaw].harmcosine.R.b[l][m].val;
            }
          }
        }

        Lmax = MAX( mod->photo.radar[ilaw].harmcosine.R.nhar,
                    mod->photo.radar[ilaw].harmcosine.C.nhar);
        nlm = matrix( 0, Lmax, 0, Lmax);
        for (l=0; l<=Lmax; l++)
          for (m=0; m<=l; m++)
            nlm[l][m] = sqrt( (2*l+1) * exp(gammln(l-m+1.0) - gammln(l+m+1.0)) );
        afactor = matrix( 0, Lmax, 0, Lmax);
        if (Lmax > 0)
          bfactor = matrix( 1, Lmax, 1, Lmax);

        for (c=0; c<mod->shape.ncomp; c++)
          for (f=0; f<mod->shape.comp[c].real.nf; f++) {
            costheta = cos( mod->shape.comp[c].real.f[f].theta);
            phi = mod->shape.comp[c].real.f[f].phi;
            for (l=0; l<=Lmax; l++)
              for (m=0; m<=l; m++) {
                plm = nlm[l][m]*plgndr( l, m, costheta);
                afactor[l][m] = cos(m*phi)*plm;
                if (m > 0)
                  bfactor[l][m] = sin(m*phi)*plm;
              }

            L = mod->photo.radar[ilaw].harmcosine.R.nhar;
            mod->photo.radar[ilaw].harmcosine.local[c][f].R.val = 0.0;
            for (l=0; l<=L; l++) {
              mod->photo.radar[ilaw].harmcosine.local[c][f].R.val
                    += mod->photo.radar[ilaw].harmcosine.R.a[l][0].val
                       * afactor[l][0];
              for (m=1; m<=l; m++)
                mod->photo.radar[ilaw].harmcosine.local[c][f].R.val
                      += mod->photo.radar[ilaw].harmcosine.R.a[l][m].val
                         * afactor[l][m]
                         + mod->photo.radar[ilaw].harmcosine.R.b[l][m].val
                           * bfactor[l][m];
            }

            L = mod->photo.radar[ilaw].harmcosine.C.nhar;
            mod->photo.radar[ilaw].harmcosine.local[c][f].C.val = 0.0;
            for (l=0; l<=L; l++) {
              mod->photo.radar[ilaw].harmcosine.local[c][f].C.val
                    += mod->photo.radar[ilaw].harmcosine.C.a[l][0].val
                       * afactor[l][0];
              for (m=1; m<=l; m++)
                mod->photo.radar[ilaw].harmcosine.local[c][f].C.val
                      += mod->photo.radar[ilaw].harmcosine.C.a[l][m].val
                         * afactor[l][m]
                         + mod->photo.radar[ilaw].harmcosine.C.b[l][m].val
                           * bfactor[l][m];
            }

            checkphotopar( mod->photo.radar[ilaw].harmcosine.local[c][f].R.val,
                           par->rad_R_min, par->rad_R_max, 3,
                           &par->badphoto, &par->badphoto_logfactor);
            checkphotopar( mod->photo.radar[ilaw].harmcosine.local[c][f].C.val,
                           par->rad_C_min, par->rad_C_max, 3,
                           &par->badphoto, &par->badphoto_logfactor);
          }

        free_matrix( nlm, 0, Lmax, 0, Lmax);
        free_matrix( afactor, 0, Lmax, 0, Lmax);
        if (Lmax > 0)
          free_matrix( bfactor, 1, Lmax, 1, Lmax);
        break;
    case INHOCOSINE_DIFF:
        if (mod->photo.radar[ilaw].inhocosine.global.R.state == 'f') {
          if (albedo_mode != 0)
            mod->photo.radar[ilaw].inhocosine.global.R.val
                     = mod->photo.radar[ilaw].inhocosine.global.R_save * radalb_factor;
          if (albedo_mode != 1)
            mod->photo.radar[ilaw].inhocosine.global.R_save
                     = mod->photo.radar[ilaw].inhocosine.global.R.val;
        }
        checkphotopar( mod->photo.radar[ilaw].inhocosine.global.R.val,
                       par->rad_R_min, par->rad_R_max, 3,
                       &par->badphoto, &par->badphoto_logfactor);
        checkphotopar( mod->photo.radar[ilaw].inhocosine.global.C.val,
                       par->rad_C_min, par->rad_C_max, 3,
                       &par->badphoto, &par->badphoto_logfactor);
        for (c=0; c<mod->shape.ncomp; c++)
          for (f=0; f<mod->shape.comp[c].real.nf; f++) {
            if (mod->photo.radar[ilaw].inhocosine.local[c][f].R.state == '=') {
                mod->photo.radar[ilaw].inhocosine.local[c][f].R.val
                      = mod->photo.radar[ilaw].inhocosine.global.R.val;
            } else if (mod->photo.radar[ilaw].inhocosine.local[c][f].R.state == 'f') {
                if (albedo_mode != 0)
                  mod->photo.radar[ilaw].inhocosine.local[c][f].R.val
                           = mod->photo.radar[ilaw].inhocosine.local[c][f].R_save
                             * radalb_factor;
                if (albedo_mode != 1)
                  mod->photo.radar[ilaw].inhocosine.local[c][f].R_save
                           = mod->photo.radar[ilaw].inhocosine.local[c][f].R.val;
            }
            if (mod->photo.radar[ilaw].inhocosine.local[c][f].C.state == '=')
              mod->photo.radar[ilaw].inhocosine.local[c][f].C.val
                    = mod->photo.radar[ilaw].inhocosine.global.C.val;
            checkphotopar( mod->photo.radar[ilaw].inhocosine.local[c][f].R.val,
                           par->rad_R_min, par->rad_R_max, 3,
                           &par->badphoto, &par->badphoto_logfactor);
            checkphotopar( mod->photo.radar[ilaw].inhocosine.local[c][f].C.val,
                           par->rad_C_min, par->rad_C_max, 3,
                           &par->badphoto, &par->badphoto_logfactor);
          }
        break;
    default:
        bailout("realize_photo: can't handle this radar law yet\n");
    }
  }
}


void checkphotopar( double parval, double parmin, double parmax, int mode,
                    unsigned char *badphoto, double *badphoto_logfactor)
{

  /*  Flag photometric parameter as bad if
           mode = 0:  parval <  parmin  or  parval >  parmax
           mode = 1:  parval <= parmin  or  parval >  parmax
           mode = 2:  parval <  parmin  or  parval >= parmax
           mode = 3:  parval <= parmin  or  parval >= parmax  */

  if (mode < 0 || mode > 3)
    bailout("realize_photo.c: checkphotopar mode must be between 0 and 3\n");

  if (mode == 0 || mode == 2) {
      if (parval < parmin) {
        *badphoto = 1;
        *badphoto_logfactor += log(1 + parmin - parval);
      }
  } else {
      if (parval <= parmin) {
        *badphoto = 1;
        *badphoto_logfactor += log(1 + parmin - parval);
      }
  }

  if (mode == 0 || mode == 1) {
      if (parval > parmax) {
        *badphoto = 1;
        *badphoto_logfactor += log(1 + parval - parmax);
      }
  } else {
      if (parval >= parmax) {
        *badphoto = 1;
        *badphoto_logfactor += log(1 + parval - parmax);
      }
  }

}
