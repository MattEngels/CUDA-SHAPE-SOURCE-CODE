/*****************************************************************************************
                                                                              photoharm.c

Convert radar and/or optical scattering laws to inhomogeneous scattering laws (harmcosine,
harmlommel, harmhapke, harmkaas) for which the various photometric parameters are given as
spherical harmonic expansions

Modified 2014 February 15 by CM:
    Implement multiple radar and optical scattering laws

Modified 2012 July 4 by CM:
    Initialize "scalefactor" elements to dummy values

Modified 2011 September 2 by CM:
    Add "harmlambert" and "inholambert" optical scattering laws

Modified 2007 August 10 by CM:
    Initialize uninitialized variables

Modified 2007 January 13 by CM:
    Fixed bug introduced when "vary_radalb" and "vary_optalb" parameters
        were added: must allocate memory for and initialize various
        "a_save" and "b_save" matrices (just as is done in read_mod.c)

Modified 2006 October 1 by CM:
    Added three new arguments to realize_photo

Written 2005 September 11 by CM
*****************************************************************************************/

#include "head.h"

void photoharm( struct par_t *par, struct mod_t *mod)
{
  char state;
  int Lmax, L, l, m, nc, nf, c, f, ilaw, faceted_inputlaw, v0, v1, v2, j, R_flag,
     hapke_flag, kaas_flag;
  double twopi, fourpi, u_fac[3], r_fac, area_fac, normconst, assoc_legendre,
         solidanglesum, corrfactor;
  double *solidangle, *costheta, *phi, ***afactor=NULL, ***bfactor=NULL;
  struct harmcosine_t *harmcosine;
  struct harmR_t *harmR;
  struct harmhapke_t *harmhapke;
  struct harmkaas_t *harmkaas;

  /*  Initialize variables to avoid compilation warnings  */

  Lmax = 0;
  harmcosine = NULL;
  harmR = NULL;
  harmhapke = NULL;
  harmkaas = NULL;

  /*  Check that this is a one-component model  */

  nc = mod->shape.ncomp;
  if (nc > 1)
    bailout("write_mod.c: 'photoharm' action is only for one-component models\n");
  c = 0;
  nf = mod->shape.comp[c].real.nf;

  twopi = 8*atan(1.0);
  fourpi = 2*twopi;

  /*  For faceted inhomogeneous input scattering laws, compute the quantities
      needed for integrating over the sphere to get the harmonic coefficients  */

  faceted_inputlaw = 0;
  if (par->radharm)
    for (ilaw=0; ilaw<mod->photo.nradlaws; ilaw++)
      if (mod->photo.radtype[ilaw] == INHOCOSINE_DIFF)
        faceted_inputlaw = 1;
  if (par->optharm)
    for (ilaw=0; ilaw<mod->photo.noptlaws; ilaw++)
      if (mod->photo.opttype[ilaw] == INHOLOMMEL
                 || mod->photo.opttype[ilaw] == INHOHAPKE
                 || mod->photo.opttype[ilaw] == INHOKAAS)
        faceted_inputlaw = 1;

  if (faceted_inputlaw) {
    Lmax = -1;
    if (par->radharm)
      for (ilaw=0; ilaw<mod->photo.nradlaws; ilaw++)
        if (mod->photo.radtype[ilaw] == INHOCOSINE_DIFF) {
          Lmax = MAX( Lmax, par->rad_R_nharm);
          Lmax = MAX( Lmax, par->rad_C_nharm);
        }
    if (par->optharm)
      for (ilaw=0; ilaw<mod->photo.noptlaws; ilaw++)
        if (mod->photo.opttype[ilaw] == INHOLOMMEL) {
            Lmax = MAX( Lmax, par->opt_R_nharm);
        } else if (mod->photo.opttype[ilaw] == INHOHAPKE) {
            Lmax = MAX( Lmax, par->opt_w_nharm);
            Lmax = MAX( Lmax, par->opt_h_nharm);
            Lmax = MAX( Lmax, par->opt_B0_nharm);
            Lmax = MAX( Lmax, par->opt_g_nharm);
            Lmax = MAX( Lmax, par->opt_theta_nharm);
        } else if (mod->photo.opttype[ilaw] == INHOKAAS) {
            Lmax = MAX( Lmax, par->opt_R_nharm);
            Lmax = MAX( Lmax, par->opt_wt_nharm);
            Lmax = MAX( Lmax, par->opt_A0_nharm);
            Lmax = MAX( Lmax, par->opt_D_nharm);
            Lmax = MAX( Lmax, par->opt_k_nharm);
        }

    solidangle = vector( 0, nf-1);
    costheta = vector( 0, nf-1);
    phi = vector( 0, nf-1);
    afactor = d3tensor( 0, Lmax, 0, Lmax, 0, nf-1);
    if (Lmax > 0)
      bfactor = d3tensor( 1, Lmax, 1, Lmax, 0, nf-1);

    solidanglesum = 0.0;
    for (f=0; f<nf; f++) {
      v0 = mod->shape.comp[c].real.f[f].v[0];
      v1 = mod->shape.comp[c].real.f[f].v[1];
      v2 = mod->shape.comp[c].real.f[f].v[2];
      for (j=0; j<=2; j++)
        u_fac[j] = (mod->shape.comp[c].real.v[v0].x[j] +
                    mod->shape.comp[c].real.v[v1].x[j] +
                    mod->shape.comp[c].real.v[v2].x[j]   ) / 3;
      r_fac = normalize( u_fac);
      area_fac = facnrm( mod->shape.comp[c].real, f);
      solidangle[f] = dot( u_fac, mod->shape.comp[c].real.f[f].n) * area_fac
                      / (r_fac*r_fac);
      solidanglesum += solidangle[f];
      costheta[f] = u_fac[2];
      if (u_fac[0] == 0.0 && u_fac[1] == 0.0)
        phi[f] = 0.0;
      else
        phi[f] = atan2( u_fac[1], u_fac[0]);
    }

    /*  Correct for the fact that the facet solid angles are
        approximate and hence don't quite sum to 4 pi steradians  */

    corrfactor = fourpi/solidanglesum;

    for (l=0; l<=Lmax; l++) {
      normconst = corrfactor*sqrt(2*l+1)/fourpi;
      for (f=0; f<nf; f++) {
        assoc_legendre = plgndr( l, 0, costheta[f]);
        afactor[l][0][f] = normconst*assoc_legendre*solidangle[f];
      }
      for (m=1; m<=l; m++) {
        normconst = corrfactor*sqrt( (2*l+1) * exp(gammln(l-m+1.0) - gammln(l+m+1.0)) )
                    / twopi;
        for (f=0; f<nf; f++) {
          assoc_legendre = plgndr( l, m, costheta[f]);
          afactor[l][m][f] = normconst*assoc_legendre*cos(m*phi[f])*solidangle[f];
          bfactor[l][m][f] = normconst*assoc_legendre*sin(m*phi[f])*solidangle[f];
        }
      }
    }

    free_vector( solidangle, 0, nf-1);
    free_vector( costheta, 0, nf-1);
    free_vector( phi, 0, nf-1);
  }

  /*  Convert the radar scattering law into a spherical-harmonic inhomogeneous law  */

  if (par->radharm) {
    harmcosine = (struct harmcosine_t *) calloc( mod->photo.nradlaws,
                                                 sizeof( struct harmcosine_t));
    for (ilaw=0; ilaw<mod->photo.nradlaws; ilaw++) {
      switch (mod->photo.radtype[ilaw]) {
      case COSINELAW_DIFF:
          mod->photo.radtype[ilaw] = HARMCOSINE_DIFF;

          L = par->rad_R_nharm;
          harmcosine[ilaw].R.nhar = L;
          harmcosine[ilaw].R.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmcosine[ilaw].R.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmcosine[ilaw].R.a_save = (double **) calloc( L+1, sizeof( struct param_t *));
          harmcosine[ilaw].R.b_save = (double **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmcosine[ilaw].R.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmcosine[ilaw].R.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmcosine[ilaw].R.a_save[l] = (double *) calloc( l+1, sizeof( struct param_t));
            harmcosine[ilaw].R.b_save[l] = (double *) calloc( l+1, sizeof( struct param_t));
          }
          harmcosine[ilaw].R.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmcosine[ilaw].R.scalefactor[j].val = -9.99;  /* dummy values */
            harmcosine[ilaw].R.scalefactor[j].state = 'c';
          }
          harmcosine[ilaw].R.a[0][0].val = mod->photo.radar[ilaw].RC.R.val;
          harmcosine[ilaw].R.a[0][0].state = mod->photo.radar[ilaw].RC.R.state;
          harmcosine[ilaw].R.a_save[0][0] = harmcosine[ilaw].R.a[0][0].val;
          for (l=1; l<=L; l++)
            for (m=0; m<=l; m++) {
              harmcosine[ilaw].R.a[l][m].val = 0.0;
              harmcosine[ilaw].R.a[l][m].state = mod->photo.radar[ilaw].RC.R.state;
              harmcosine[ilaw].R.a_save[l][m] = harmcosine[ilaw].R.a[l][m].val;
              if (m > 0) {
                harmcosine[ilaw].R.b[l][m].val = 0.0;
                harmcosine[ilaw].R.b[l][m].state = mod->photo.radar[ilaw].RC.R.state;
                harmcosine[ilaw].R.b_save[l][m] = harmcosine[ilaw].R.b[l][m].val;
              }
            }

          L = par->rad_C_nharm;
          harmcosine[ilaw].C.nhar = L;
          harmcosine[ilaw].C.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmcosine[ilaw].C.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmcosine[ilaw].C.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmcosine[ilaw].C.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
          }
          harmcosine[ilaw].C.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmcosine[ilaw].C.scalefactor[j].val = -9.99;  /* dummy values */
            harmcosine[ilaw].C.scalefactor[j].state = 'c';
          }
          harmcosine[ilaw].C.a[0][0].val = mod->photo.radar[ilaw].RC.C.val;
          harmcosine[ilaw].C.a[0][0].state = mod->photo.radar[ilaw].RC.C.state;
          for (l=1; l<=L; l++)
            for (m=0; m<=l; m++) {
              harmcosine[ilaw].C.a[l][m].val = 0.0;
              harmcosine[ilaw].C.a[l][m].state = mod->photo.radar[ilaw].RC.C.state;
              if (m > 0) {
                harmcosine[ilaw].C.b[l][m].val = 0.0;
                harmcosine[ilaw].C.b[l][m].state = mod->photo.radar[ilaw].RC.C.state;
              }
            }

          harmcosine[ilaw].local = (struct RC_t **) calloc( nc, sizeof( struct RC_t *));
          harmcosine[ilaw].local[c] = (struct RC_t *) calloc( nf, sizeof( struct RC_t));
  
          mod->photo.radar[ilaw].harmcosine = harmcosine[ilaw];
          break;
      case HARMCOSINE_DIFF:
          L = par->rad_R_nharm;
          harmcosine[ilaw].R.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmcosine[ilaw].R.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmcosine[ilaw].R.a_save = (double **) calloc( L+1, sizeof( struct param_t *));
          harmcosine[ilaw].R.b_save = (double **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmcosine[ilaw].R.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmcosine[ilaw].R.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmcosine[ilaw].R.a_save[l] = (double *) calloc( l+1, sizeof( struct param_t));
            harmcosine[ilaw].R.b_save[l] = (double *) calloc( l+1, sizeof( struct param_t));
          }
          for (l=0; l<=L; l++)
            for (m=0; m<=l; m++) {
              if (l <= mod->photo.radar[ilaw].harmcosine.R.nhar) {
                  harmcosine[ilaw].R.a[l][m].val = mod->photo.radar[ilaw].harmcosine.R.a[l][m].val;
                  harmcosine[ilaw].R.a[l][m].state = mod->photo.radar[ilaw].harmcosine.R.a[l][m].state;
                  if (m > 0) {
                    harmcosine[ilaw].R.b[l][m].val = mod->photo.radar[ilaw].harmcosine.R.b[l][m].val;
                    harmcosine[ilaw].R.b[l][m].state = mod->photo.radar[ilaw].harmcosine.R.b[l][m].state;
                  }
              } else {
                  harmcosine[ilaw].R.a[l][m].val = 0.0;
                  harmcosine[ilaw].R.a[l][m].state = 'f';
                  if (m > 0) {
                    harmcosine[ilaw].R.b[l][m].val = 0.0;
                    harmcosine[ilaw].R.b[l][m].state = 'f';
                  }
              }
              harmcosine[ilaw].R.a_save[l][m] = harmcosine[ilaw].R.a[l][m].val;
              if (m > 0)
                harmcosine[ilaw].R.b_save[l][m] = harmcosine[ilaw].R.b[l][m].val;
            }
          harmcosine[ilaw].R.nhar = L;
          harmcosine[ilaw].R.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmcosine[ilaw].R.scalefactor[j].val = -9.99;  /* dummy values */
            harmcosine[ilaw].R.scalefactor[j].state = 'c';
          }

          L = par->rad_C_nharm;
          harmcosine[ilaw].C.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmcosine[ilaw].C.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmcosine[ilaw].C.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmcosine[ilaw].C.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
          }
          for (l=0; l<=L; l++)
            for (m=0; m<=l; m++) {
              if (l <= mod->photo.radar[ilaw].harmcosine.C.nhar) {
                  harmcosine[ilaw].C.a[l][m].val = mod->photo.radar[ilaw].harmcosine.C.a[l][m].val;
                  harmcosine[ilaw].C.a[l][m].state = mod->photo.radar[ilaw].harmcosine.C.a[l][m].state;
                  if (m > 0) {
                    harmcosine[ilaw].C.b[l][m].val = mod->photo.radar[ilaw].harmcosine.C.b[l][m].val;
                    harmcosine[ilaw].C.b[l][m].state = mod->photo.radar[ilaw].harmcosine.C.b[l][m].state;
                  }
              } else {
                  harmcosine[ilaw].C.a[l][m].val = 0.0;
                  harmcosine[ilaw].C.a[l][m].state = 'f';
                  if (m > 0) {
                    harmcosine[ilaw].C.b[l][m].val = 0.0;
                    harmcosine[ilaw].C.b[l][m].state = 'f';
                  }
              }
            }
          harmcosine[ilaw].C.nhar = L;
          harmcosine[ilaw].C.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmcosine[ilaw].C.scalefactor[j].val = -9.99;  /* dummy values */
            harmcosine[ilaw].C.scalefactor[j].state = 'c';
          }

          harmcosine[ilaw].local = (struct RC_t **) calloc( nc, sizeof( struct RC_t *));
          harmcosine[ilaw].local[c] = (struct RC_t *) calloc( nf, sizeof( struct RC_t));

          mod->photo.radar[ilaw].harmcosine = harmcosine[ilaw];
          break;
      case INHOCOSINE_DIFF:
          mod->photo.radtype[ilaw] = HARMCOSINE_DIFF;

          L = par->rad_R_nharm;
          harmcosine[ilaw].R.nhar = L;
          harmcosine[ilaw].R.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmcosine[ilaw].R.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmcosine[ilaw].R.a_save = (double **) calloc( L+1, sizeof( struct param_t *));
          harmcosine[ilaw].R.b_save = (double **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmcosine[ilaw].R.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmcosine[ilaw].R.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmcosine[ilaw].R.a_save[l] = (double *) calloc( l+1, sizeof( struct param_t));
            harmcosine[ilaw].R.b_save[l] = (double *) calloc( l+1, sizeof( struct param_t));
          }
          harmcosine[ilaw].R.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmcosine[ilaw].R.scalefactor[j].val = -9.99;  /* dummy values */
            harmcosine[ilaw].R.scalefactor[j].state = 'c';
          }
          state = mod->photo.radar[ilaw].inhocosine.global.R.state;
          for (f=0; f<nf; f++)
            if (mod->photo.radar[ilaw].inhocosine.local[c][f].R.state == 'f')
              state = 'f';
          for (l=0; l<=L; l++)
            for (m=0; m<=l; m++) {
              harmcosine[ilaw].R.a[l][m].val = 0.0;
              if (m > 0)
                harmcosine[ilaw].R.b[l][m].val = 0.0;
              for (f=0; f<nf; f++) {
                harmcosine[ilaw].R.a[l][m].val
                       += afactor[l][m][f]*mod->photo.radar[ilaw].inhocosine.local[c][f].R.val;
                if (m > 0)
                  harmcosine[ilaw].R.b[l][m].val
                         += bfactor[l][m][f]*mod->photo.radar[ilaw].inhocosine.local[c][f].R.val;
              }
              harmcosine[ilaw].R.a[l][m].state = state;
              harmcosine[ilaw].R.a_save[l][m] = harmcosine[ilaw].R.a[l][m].val;
              if (m > 0) {
                harmcosine[ilaw].R.b[l][m].state = state;
                harmcosine[ilaw].R.b_save[l][m] = harmcosine[ilaw].R.b[l][m].val;
              }
            }

          L = par->rad_C_nharm;
          harmcosine[ilaw].C.nhar = L;
          harmcosine[ilaw].C.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmcosine[ilaw].C.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmcosine[ilaw].C.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmcosine[ilaw].C.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
          }
          harmcosine[ilaw].C.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmcosine[ilaw].C.scalefactor[j].val = -9.99;  /* dummy values */
            harmcosine[ilaw].C.scalefactor[j].state = 'c';
          }
          state = mod->photo.radar[ilaw].inhocosine.global.C.state;
          for (f=0; f<nf; f++)
            if (mod->photo.radar[ilaw].inhocosine.local[c][f].C.state == 'f')
              state = 'f';
          for (l=0; l<=L; l++)
            for (m=0; m<=l; m++) {
              harmcosine[ilaw].C.a[l][m].val = 0.0;
              if (m > 0)
                harmcosine[ilaw].C.b[l][m].val = 0.0;
              for (f=0; f<nf; f++) {
                harmcosine[ilaw].C.a[l][m].val
                       += afactor[l][m][f]*mod->photo.radar[ilaw].inhocosine.local[c][f].C.val;
                if (m > 0)
                  harmcosine[ilaw].C.b[l][m].val
                         += bfactor[l][m][f]*mod->photo.radar[ilaw].inhocosine.local[c][f].C.val;
              }
              harmcosine[ilaw].C.a[l][m].state = state;
              if (m > 0)
                harmcosine[ilaw].C.b[l][m].state = state;
            }

          harmcosine[ilaw].local = (struct RC_t **) calloc( nc, sizeof( struct RC_t *));
          harmcosine[ilaw].local[c] = (struct RC_t *) calloc( nf, sizeof( struct RC_t));

          mod->photo.radar[ilaw].harmcosine = harmcosine[ilaw];
          break;
      default:
          bailout("photoharm: can't handle this radar law yet\n");
      }
    }
  }

  /*  Convert the optical scattering law into a spherical-harmonic inhomogeneous law  */

  if (par->optharm) {

    /*  Allocate as much memory for vectors of harmonic structures as might be needed  */

    R_flag = hapke_flag = kaas_flag = 0;
    for (ilaw=0; ilaw<mod->photo.noptlaws; ilaw++) {
      if (mod->photo.opttype[ilaw] == LAMBERTLAW || mod->photo.opttype[ilaw] == LOMMEL
                                                   || mod->photo.opttype[ilaw] == HARMLAMBERT
                                                   || mod->photo.opttype[ilaw] == HARMLOMMEL
                                                   || mod->photo.opttype[ilaw] == INHOLAMBERT
                                                   || mod->photo.opttype[ilaw] == INHOLOMMEL)
        R_flag = 1;
      if (mod->photo.opttype[ilaw] == HAPKE || mod->photo.opttype[ilaw] == HARMHAPKE
                                              || mod->photo.opttype[ilaw] == INHOHAPKE)
        hapke_flag = 1;
      if (mod->photo.opttype[ilaw] == KAASALAINEN || mod->photo.opttype[ilaw] == HARMKAAS
                                                    || mod->photo.opttype[ilaw] == INHOKAAS)
        kaas_flag = 1;
    }
    if (R_flag)
      harmR = (struct harmR_t *) calloc( mod->photo.noptlaws, sizeof( struct harmR_t));
    if (hapke_flag)
      harmhapke = (struct harmhapke_t *) calloc( mod->photo.noptlaws,
                                                  sizeof( struct harmhapke_t));
    if (kaas_flag)
      harmkaas = (struct harmkaas_t *) calloc( mod->photo.noptlaws,
                                                sizeof( struct harmkaas_t));

    /*  Create the new harmonic structure(s)  */

    for (ilaw=0; ilaw<mod->photo.noptlaws; ilaw++) {
      switch (mod->photo.opttype[ilaw]) {
      case LAMBERTLAW:
      case LOMMEL:
          if (mod->photo.opttype[ilaw] == LAMBERTLAW)
            mod->photo.opttype[ilaw] = HARMLAMBERT;
          else
            mod->photo.opttype[ilaw] = HARMLOMMEL;
          L = par->opt_R_nharm;
          harmR[ilaw].R.nhar = L;
          harmR[ilaw].R.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmR[ilaw].R.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmR[ilaw].R.a_save = (double **) calloc( L+1, sizeof( struct param_t *));
          harmR[ilaw].R.b_save = (double **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmR[ilaw].R.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmR[ilaw].R.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmR[ilaw].R.a_save[l] = (double *) calloc( l+1, sizeof( struct param_t));
            harmR[ilaw].R.b_save[l] = (double *) calloc( l+1, sizeof( struct param_t));
          }
          harmR[ilaw].R.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmR[ilaw].R.scalefactor[j].val = -9.99;  /* dummy values */
            harmR[ilaw].R.scalefactor[j].state = 'c';
          }
          harmR[ilaw].R.a[0][0].val = mod->photo.optical[ilaw].R.R.val;
          harmR[ilaw].R.a[0][0].state = mod->photo.optical[ilaw].R.R.state;
          harmR[ilaw].R.a_save[0][0] = harmR[ilaw].R.a[0][0].val;
          for (l=1; l<=L; l++)
            for (m=0; m<=l; m++) {
              harmR[ilaw].R.a[l][m].val = 0.0;
              harmR[ilaw].R.a[l][m].state = mod->photo.optical[ilaw].R.R.state;
              harmR[ilaw].R.a_save[l][m] = harmR[ilaw].R.a[l][m].val;
              if (m > 0) {
                harmR[ilaw].R.b[l][m].val = 0.0;
                harmR[ilaw].R.b[l][m].state = mod->photo.optical[ilaw].R.R.state;
                harmR[ilaw].R.b_save[l][m] = harmR[ilaw].R.b[l][m].val;
              }
            }
          harmR[ilaw].local = (struct R_t **) calloc( nc, sizeof( struct R_t *));
          harmR[ilaw].local[c] = (struct R_t *) calloc( nf, sizeof( struct R_t));
          mod->photo.optical[ilaw].harmR = harmR[ilaw];
          break;
      case HARMLAMBERT:
      case HARMLOMMEL:
          L = par->opt_R_nharm;
          harmR[ilaw].R.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmR[ilaw].R.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmR[ilaw].R.a_save = (double **) calloc( L+1, sizeof( struct param_t *));
          harmR[ilaw].R.b_save = (double **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmR[ilaw].R.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmR[ilaw].R.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmR[ilaw].R.a_save[l] = (double *) calloc( l+1, sizeof( struct param_t));
            harmR[ilaw].R.b_save[l] = (double *) calloc( l+1, sizeof( struct param_t));
          }
          for (l=0; l<=L; l++)
            for (m=0; m<=l; m++) {
              if (l <= mod->photo.optical[ilaw].harmR.R.nhar) {
                  harmR[ilaw].R.a[l][m].val = mod->photo.optical[ilaw].harmR.R.a[l][m].val;
                  harmR[ilaw].R.a[l][m].state = mod->photo.optical[ilaw].harmR.R.a[l][m].state;
                  if (m > 0) {
                    harmR[ilaw].R.b[l][m].val = mod->photo.optical[ilaw].harmR.R.b[l][m].val;
                    harmR[ilaw].R.b[l][m].state = mod->photo.optical[ilaw].harmR.R.b[l][m].state;
                  }
              } else {
                  harmR[ilaw].R.a[l][m].val = 0.0;
                  harmR[ilaw].R.a[l][m].state = 'f';
                  if (m > 0) {
                    harmR[ilaw].R.b[l][m].val = 0.0;
                    harmR[ilaw].R.b[l][m].state = 'f';
                  }
              }
              harmR[ilaw].R.a_save[l][m] = harmR[ilaw].R.a[l][m].val;
              if (m > 0)
                harmR[ilaw].R.b_save[l][m] = harmR[ilaw].R.b[l][m].val;
            }
          harmR[ilaw].R.nhar = L;
          harmR[ilaw].R.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmR[ilaw].R.scalefactor[j].val = -9.99;  /* dummy values */
            harmR[ilaw].R.scalefactor[j].state = 'c';
          }
          harmR[ilaw].local = (struct R_t **) calloc( nc, sizeof( struct R_t *));
          harmR[ilaw].local[c] = (struct R_t *) calloc( nf, sizeof( struct R_t));
          mod->photo.optical[ilaw].harmR = harmR[ilaw];
          break;
      case INHOLAMBERT:
      case INHOLOMMEL:
          if (mod->photo.opttype[ilaw] == INHOLAMBERT)
            mod->photo.opttype[ilaw] = HARMLAMBERT;
          else
            mod->photo.opttype[ilaw] = HARMLOMMEL;
          L = par->opt_R_nharm;
          harmR[ilaw].R.nhar = L;
          harmR[ilaw].R.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmR[ilaw].R.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmR[ilaw].R.a_save = (double **) calloc( L+1, sizeof( struct param_t *));
          harmR[ilaw].R.b_save = (double **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmR[ilaw].R.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmR[ilaw].R.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmR[ilaw].R.a_save[l] = (double *) calloc( l+1, sizeof( struct param_t));
            harmR[ilaw].R.b_save[l] = (double *) calloc( l+1, sizeof( struct param_t));
          }
          harmR[ilaw].R.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmR[ilaw].R.scalefactor[j].val = -9.99;  /* dummy values */
            harmR[ilaw].R.scalefactor[j].state = 'c';
          }
          state = mod->photo.optical[ilaw].inhoR.global.R.state;
          for (f=0; f<nf; f++)
            if (mod->photo.optical[ilaw].inhoR.local[c][f].R.state == 'f')
              state = 'f';
          for (l=0; l<=L; l++)
            for (m=0; m<=l; m++) {
              harmR[ilaw].R.a[l][m].val = 0.0;
              if (m > 0)
                harmR[ilaw].R.b[l][m].val = 0.0;
              for (f=0; f<nf; f++) {
                harmR[ilaw].R.a[l][m].val
                       += afactor[l][m][f]*mod->photo.optical[ilaw].inhoR.local[c][f].R.val;
                if (m > 0)
                  harmR[ilaw].R.b[l][m].val
                         += bfactor[l][m][f]*mod->photo.optical[ilaw].inhoR.local[c][f].R.val;
              }
              harmR[ilaw].R.a[l][m].state = state;
              harmR[ilaw].R.a_save[l][m] = harmR[ilaw].R.a[l][m].val;
              if (m > 0) {
                harmR[ilaw].R.b[l][m].state = state;
                harmR[ilaw].R.b_save[l][m] = harmR[ilaw].R.b[l][m].val;
              }
            }
          harmR[ilaw].local = (struct R_t **) calloc( nc, sizeof( struct R_t *));
          harmR[ilaw].local[c] = (struct R_t *) calloc( nf, sizeof( struct R_t));
          mod->photo.optical[ilaw].harmR = harmR[ilaw];
          break;
      case HAPKE:
          mod->photo.opttype[ilaw] = HARMHAPKE;
  
          L = par->opt_w_nharm;
          harmhapke[ilaw].w.nhar = L;
          harmhapke[ilaw].w.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmhapke[ilaw].w.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmhapke[ilaw].w.a_save = (double **) calloc( L+1, sizeof( struct param_t *));
          harmhapke[ilaw].w.b_save = (double **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmhapke[ilaw].w.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmhapke[ilaw].w.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmhapke[ilaw].w.a_save[l] = (double *) calloc( l+1, sizeof( struct param_t));
            harmhapke[ilaw].w.b_save[l] = (double *) calloc( l+1, sizeof( struct param_t));
          }
          harmhapke[ilaw].w.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmhapke[ilaw].w.scalefactor[j].val = -9.99;  /* dummy values */
            harmhapke[ilaw].w.scalefactor[j].state = 'c';
          }
          harmhapke[ilaw].w.a[0][0].val = mod->photo.optical[ilaw].hapke.w.val;
          harmhapke[ilaw].w.a[0][0].state = mod->photo.optical[ilaw].hapke.w.state;
          harmhapke[ilaw].w.a_save[0][0] = harmhapke[ilaw].w.a[0][0].val;
          for (l=1; l<=L; l++)
            for (m=0; m<=l; m++) {
              harmhapke[ilaw].w.a[l][m].val = 0.0;
              harmhapke[ilaw].w.a[l][m].state = mod->photo.optical[ilaw].hapke.w.state;
              harmhapke[ilaw].w.a_save[l][m] = harmhapke[ilaw].w.a[l][m].val;
              if (m > 0) {
                harmhapke[ilaw].w.b[l][m].val = 0.0;
                harmhapke[ilaw].w.b[l][m].state = mod->photo.optical[ilaw].hapke.w.state;
                harmhapke[ilaw].w.b_save[l][m] = harmhapke[ilaw].w.b[l][m].val;
              }
            }
  
          L = par->opt_h_nharm;
          harmhapke[ilaw].h.nhar = L;
          harmhapke[ilaw].h.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmhapke[ilaw].h.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmhapke[ilaw].h.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmhapke[ilaw].h.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
          }
          harmhapke[ilaw].h.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmhapke[ilaw].h.scalefactor[j].val = -9.99;  /* dummy values */
            harmhapke[ilaw].h.scalefactor[j].state = 'c';
          }
          harmhapke[ilaw].h.a[0][0].val = mod->photo.optical[ilaw].hapke.h.val;
          harmhapke[ilaw].h.a[0][0].state = mod->photo.optical[ilaw].hapke.h.state;
          for (l=1; l<=L; l++)
            for (m=0; m<=l; m++) {
              harmhapke[ilaw].h.a[l][m].val = 0.0;
              harmhapke[ilaw].h.a[l][m].state = mod->photo.optical[ilaw].hapke.h.state;
              if (m > 0) {
                harmhapke[ilaw].h.b[l][m].val = 0.0;
                harmhapke[ilaw].h.b[l][m].state = mod->photo.optical[ilaw].hapke.h.state;
              }
            }
  
          L = par->opt_B0_nharm;
          harmhapke[ilaw].B0.nhar = L;
          harmhapke[ilaw].B0.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmhapke[ilaw].B0.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmhapke[ilaw].B0.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmhapke[ilaw].B0.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
          }
          harmhapke[ilaw].B0.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmhapke[ilaw].B0.scalefactor[j].val = -9.99;  /* dummy values */
            harmhapke[ilaw].B0.scalefactor[j].state = 'c';
          }
          harmhapke[ilaw].B0.a[0][0].val = mod->photo.optical[ilaw].hapke.B0.val;
          harmhapke[ilaw].B0.a[0][0].state = mod->photo.optical[ilaw].hapke.B0.state;
          for (l=1; l<=L; l++)
            for (m=0; m<=l; m++) {
              harmhapke[ilaw].B0.a[l][m].val = 0.0;
              harmhapke[ilaw].B0.a[l][m].state = mod->photo.optical[ilaw].hapke.B0.state;
              if (m > 0) {
                harmhapke[ilaw].B0.b[l][m].val = 0.0;
                harmhapke[ilaw].B0.b[l][m].state = mod->photo.optical[ilaw].hapke.B0.state;
              }
            }
  
          L = par->opt_g_nharm;
          harmhapke[ilaw].g.nhar = L;
          harmhapke[ilaw].g.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmhapke[ilaw].g.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmhapke[ilaw].g.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmhapke[ilaw].g.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
          }
          harmhapke[ilaw].g.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmhapke[ilaw].g.scalefactor[j].val = -9.99;  /* dummy values */
            harmhapke[ilaw].g.scalefactor[j].state = 'c';
          }
          harmhapke[ilaw].g.a[0][0].val = mod->photo.optical[ilaw].hapke.g.val;
          harmhapke[ilaw].g.a[0][0].state = mod->photo.optical[ilaw].hapke.g.state;
          for (l=1; l<=L; l++)
            for (m=0; m<=l; m++) {
              harmhapke[ilaw].g.a[l][m].val = 0.0;
              harmhapke[ilaw].g.a[l][m].state = mod->photo.optical[ilaw].hapke.g.state;
              if (m > 0) {
                harmhapke[ilaw].g.b[l][m].val = 0.0;
                harmhapke[ilaw].g.b[l][m].state = mod->photo.optical[ilaw].hapke.g.state;
              }
            }
  
          L = par->opt_theta_nharm;
          harmhapke[ilaw].theta.nhar = L;
          harmhapke[ilaw].theta.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmhapke[ilaw].theta.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmhapke[ilaw].theta.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmhapke[ilaw].theta.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
          }
          harmhapke[ilaw].theta.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmhapke[ilaw].theta.scalefactor[j].val = -9.99;  /* dummy values */
            harmhapke[ilaw].theta.scalefactor[j].state = 'c';
          }
          harmhapke[ilaw].theta.a[0][0].val = mod->photo.optical[ilaw].hapke.theta.val;
          harmhapke[ilaw].theta.a[0][0].state = mod->photo.optical[ilaw].hapke.theta.state;
          for (l=1; l<=L; l++)
            for (m=0; m<=l; m++) {
              harmhapke[ilaw].theta.a[l][m].val = 0.0;
              harmhapke[ilaw].theta.a[l][m].state = mod->photo.optical[ilaw].hapke.theta.state;
              if (m > 0) {
                harmhapke[ilaw].theta.b[l][m].val = 0.0;
                harmhapke[ilaw].theta.b[l][m].state = mod->photo.optical[ilaw].hapke.theta.state;
              }
            }
  
          harmhapke[ilaw].local = (struct hapke_t **) calloc( nc, sizeof( struct hapke_t *));
          harmhapke[ilaw].local[c] = (struct hapke_t *) calloc( nf, sizeof( struct hapke_t));
  
          mod->photo.optical[ilaw].harmhapke = harmhapke[ilaw];
          break;
      case HARMHAPKE:
          L = par->opt_w_nharm;
          harmhapke[ilaw].w.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmhapke[ilaw].w.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmhapke[ilaw].w.a_save = (double **) calloc( L+1, sizeof( struct param_t *));
          harmhapke[ilaw].w.b_save = (double **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmhapke[ilaw].w.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmhapke[ilaw].w.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmhapke[ilaw].w.a_save[l] = (double *) calloc( l+1, sizeof( struct param_t));
            harmhapke[ilaw].w.b_save[l] = (double *) calloc( l+1, sizeof( struct param_t));
          }
          for (l=0; l<=L; l++)
            for (m=0; m<=l; m++) {
              if (l <= mod->photo.optical[ilaw].harmhapke.w.nhar) {
                  harmhapke[ilaw].w.a[l][m].val = mod->photo.optical[ilaw].harmhapke.w.a[l][m].val;
                  harmhapke[ilaw].w.a[l][m].state = mod->photo.optical[ilaw].harmhapke.w.a[l][m].state;
                  if (m > 0) {
                    harmhapke[ilaw].w.b[l][m].val = mod->photo.optical[ilaw].harmhapke.w.b[l][m].val;
                    harmhapke[ilaw].w.b[l][m].state = mod->photo.optical[ilaw].harmhapke.w.b[l][m].state;
                  }
              } else {
                  harmhapke[ilaw].w.a[l][m].val = 0.0;
                  harmhapke[ilaw].w.a[l][m].state = 'f';
                  if (m > 0) {
                    harmhapke[ilaw].w.b[l][m].val = 0.0;
                    harmhapke[ilaw].w.b[l][m].state = 'f';
                  }
              }
              harmhapke[ilaw].w.a_save[l][m] = harmhapke[ilaw].w.a[l][m].val;
              if (m > 0)
                harmhapke[ilaw].w.b_save[l][m] = harmhapke[ilaw].w.b[l][m].val;
            }
          harmhapke[ilaw].w.nhar = L;
          harmhapke[ilaw].w.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmhapke[ilaw].w.scalefactor[j].val = -9.99;  /* dummy values */
            harmhapke[ilaw].w.scalefactor[j].state = 'c';
          }
  
          L = par->opt_h_nharm;
          harmhapke[ilaw].h.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmhapke[ilaw].h.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmhapke[ilaw].h.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmhapke[ilaw].h.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
          }
          for (l=0; l<=L; l++)
            for (m=0; m<=l; m++) {
              if (l <= mod->photo.optical[ilaw].harmhapke.h.nhar) {
                  harmhapke[ilaw].h.a[l][m].val = mod->photo.optical[ilaw].harmhapke.h.a[l][m].val;
                  harmhapke[ilaw].h.a[l][m].state = mod->photo.optical[ilaw].harmhapke.h.a[l][m].state;
                  if (m > 0) {
                    harmhapke[ilaw].h.b[l][m].val = mod->photo.optical[ilaw].harmhapke.h.b[l][m].val;
                    harmhapke[ilaw].h.b[l][m].state = mod->photo.optical[ilaw].harmhapke.h.b[l][m].state;
                  }
              } else {
                  harmhapke[ilaw].h.a[l][m].val = 0.0;
                  harmhapke[ilaw].h.a[l][m].state = 'f';
                  if (m > 0) {
                    harmhapke[ilaw].h.b[l][m].val = 0.0;
                    harmhapke[ilaw].h.b[l][m].state = 'f';
                  }
              }
            }
          harmhapke[ilaw].h.nhar = L;
          harmhapke[ilaw].h.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmhapke[ilaw].h.scalefactor[j].val = -9.99;  /* dummy values */
            harmhapke[ilaw].h.scalefactor[j].state = 'c';
          }
  
          L = par->opt_B0_nharm;
          harmhapke[ilaw].B0.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmhapke[ilaw].B0.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmhapke[ilaw].B0.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmhapke[ilaw].B0.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
          }
          for (l=0; l<=L; l++)
            for (m=0; m<=l; m++) {
              if (l <= mod->photo.optical[ilaw].harmhapke.B0.nhar) {
                  harmhapke[ilaw].B0.a[l][m].val = mod->photo.optical[ilaw].harmhapke.B0.a[l][m].val;
                  harmhapke[ilaw].B0.a[l][m].state = mod->photo.optical[ilaw].harmhapke.B0.a[l][m].state;
                  if (m > 0) {
                    harmhapke[ilaw].B0.b[l][m].val = mod->photo.optical[ilaw].harmhapke.B0.b[l][m].val;
                    harmhapke[ilaw].B0.b[l][m].state = mod->photo.optical[ilaw].harmhapke.B0.b[l][m].state;
                  }
              } else {
                  harmhapke[ilaw].B0.a[l][m].val = 0.0;
                  harmhapke[ilaw].B0.a[l][m].state = 'f';
                  if (m > 0) {
                    harmhapke[ilaw].B0.b[l][m].val = 0.0;
                    harmhapke[ilaw].B0.b[l][m].state = 'f';
                  }
              }
            }
          harmhapke[ilaw].B0.nhar = L;
          harmhapke[ilaw].B0.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmhapke[ilaw].B0.scalefactor[j].val = -9.99;  /* dummy values */
            harmhapke[ilaw].B0.scalefactor[j].state = 'c';
          }
  
          L = par->opt_g_nharm;
          harmhapke[ilaw].g.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmhapke[ilaw].g.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmhapke[ilaw].g.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmhapke[ilaw].g.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
          }
          for (l=0; l<=L; l++)
            for (m=0; m<=l; m++) {
              if (l <= mod->photo.optical[ilaw].harmhapke.g.nhar) {
                  harmhapke[ilaw].g.a[l][m].val = mod->photo.optical[ilaw].harmhapke.g.a[l][m].val;
                  harmhapke[ilaw].g.a[l][m].state = mod->photo.optical[ilaw].harmhapke.g.a[l][m].state;
                  if (m > 0) {
                    harmhapke[ilaw].g.b[l][m].val = mod->photo.optical[ilaw].harmhapke.g.b[l][m].val;
                    harmhapke[ilaw].g.b[l][m].state = mod->photo.optical[ilaw].harmhapke.g.b[l][m].state;
                  }
              } else {
                  harmhapke[ilaw].g.a[l][m].val = 0.0;
                  harmhapke[ilaw].g.a[l][m].state = 'f';
                  if (m > 0) {
                    harmhapke[ilaw].g.b[l][m].val = 0.0;
                    harmhapke[ilaw].g.b[l][m].state = 'f';
                  }
              }
            }
          harmhapke[ilaw].g.nhar = L;
          harmhapke[ilaw].g.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmhapke[ilaw].g.scalefactor[j].val = -9.99;  /* dummy values */
            harmhapke[ilaw].g.scalefactor[j].state = 'c';
          }
  
          L = par->opt_theta_nharm;
          harmhapke[ilaw].theta.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmhapke[ilaw].theta.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmhapke[ilaw].theta.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmhapke[ilaw].theta.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
          }
          for (l=0; l<=L; l++)
            for (m=0; m<=l; m++) {
              if (l <= mod->photo.optical[ilaw].harmhapke.theta.nhar) {
                  harmhapke[ilaw].theta.a[l][m].val = mod->photo.optical[ilaw].harmhapke.theta.a[l][m].val;
                  harmhapke[ilaw].theta.a[l][m].state = mod->photo.optical[ilaw].harmhapke.theta.a[l][m].state;
                  if (m > 0) {
                    harmhapke[ilaw].theta.b[l][m].val = mod->photo.optical[ilaw].harmhapke.theta.b[l][m].val;
                    harmhapke[ilaw].theta.b[l][m].state = mod->photo.optical[ilaw].harmhapke.theta.b[l][m].state;
                  }
              } else {
                  harmhapke[ilaw].theta.a[l][m].val = 0.0;
                  harmhapke[ilaw].theta.a[l][m].state = 'f';
                  if (m > 0) {
                    harmhapke[ilaw].theta.b[l][m].val = 0.0;
                    harmhapke[ilaw].theta.b[l][m].state = 'f';
                  }
              }
            }
          harmhapke[ilaw].theta.nhar = L;
          harmhapke[ilaw].theta.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmhapke[ilaw].theta.scalefactor[j].val = -9.99;  /* dummy values */
            harmhapke[ilaw].theta.scalefactor[j].state = 'c';
          }
  
          harmhapke[ilaw].local = (struct hapke_t **) calloc( nc, sizeof( struct hapke_t *));
          harmhapke[ilaw].local[c] = (struct hapke_t *) calloc( nf, sizeof( struct hapke_t));
  
          mod->photo.optical[ilaw].harmhapke = harmhapke[ilaw];
          break;
      case INHOHAPKE:
          mod->photo.opttype[ilaw] = HARMHAPKE;
  
          L = par->opt_w_nharm;
          harmhapke[ilaw].w.nhar = L;
          harmhapke[ilaw].w.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmhapke[ilaw].w.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmhapke[ilaw].w.a_save = (double **) calloc( L+1, sizeof( struct param_t *));
          harmhapke[ilaw].w.b_save = (double **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmhapke[ilaw].w.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmhapke[ilaw].w.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmhapke[ilaw].w.a_save[l] = (double *) calloc( l+1, sizeof( struct param_t));
            harmhapke[ilaw].w.b_save[l] = (double *) calloc( l+1, sizeof( struct param_t));
          }
          harmhapke[ilaw].w.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmhapke[ilaw].w.scalefactor[j].val = -9.99;  /* dummy values */
            harmhapke[ilaw].w.scalefactor[j].state = 'c';
          }
          state = mod->photo.optical[ilaw].inhohapke.global.w.state;
          for (f=0; f<nf; f++)
            if (mod->photo.optical[ilaw].inhohapke.local[c][f].w.state == 'f')
              state = 'f';
          for (l=0; l<=L; l++)
            for (m=0; m<=l; m++) {
              harmhapke[ilaw].w.a[l][m].val = 0.0;
              if (m > 0)
                harmhapke[ilaw].w.b[l][m].val = 0.0;
              for (f=0; f<nf; f++) {
                harmhapke[ilaw].w.a[l][m].val
                       += afactor[l][m][f]*mod->photo.optical[ilaw].inhohapke.local[c][f].w.val;
                if (m > 0)
                  harmhapke[ilaw].w.b[l][m].val
                         += bfactor[l][m][f]*mod->photo.optical[ilaw].inhohapke.local[c][f].w.val;
              }
              harmhapke[ilaw].w.a[l][m].state = state;
              harmhapke[ilaw].w.a_save[l][m] = harmhapke[ilaw].w.a[l][m].val;
              if (m > 0) {
                harmhapke[ilaw].w.b[l][m].state = state;
                harmhapke[ilaw].w.b_save[l][m] = harmhapke[ilaw].w.b[l][m].val;
              }
            }
  
          L = par->opt_h_nharm;
          harmhapke[ilaw].h.nhar = L;
          harmhapke[ilaw].h.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmhapke[ilaw].h.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmhapke[ilaw].h.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmhapke[ilaw].h.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
          }
          harmhapke[ilaw].h.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmhapke[ilaw].h.scalefactor[j].val = -9.99;  /* dummy values */
            harmhapke[ilaw].h.scalefactor[j].state = 'c';
          }
          state = mod->photo.optical[ilaw].inhohapke.global.h.state;
          for (f=0; f<nf; f++)
            if (mod->photo.optical[ilaw].inhohapke.local[c][f].h.state == 'f')
              state = 'f';
          for (l=0; l<=L; l++)
            for (m=0; m<=l; m++) {
              harmhapke[ilaw].h.a[l][m].val = 0.0;
              if (m > 0)
                harmhapke[ilaw].h.b[l][m].val = 0.0;
              for (f=0; f<nf; f++) {
                harmhapke[ilaw].h.a[l][m].val
                       += afactor[l][m][f]*mod->photo.optical[ilaw].inhohapke.local[c][f].h.val;
                if (m > 0)
                  harmhapke[ilaw].h.b[l][m].val
                         += bfactor[l][m][f]*mod->photo.optical[ilaw].inhohapke.local[c][f].h.val;
              }
              harmhapke[ilaw].h.a[l][m].state = state;
              if (m > 0)
                harmhapke[ilaw].h.b[l][m].state = state;
            }
  
          L = par->opt_B0_nharm;
          harmhapke[ilaw].B0.nhar = L;
          harmhapke[ilaw].B0.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmhapke[ilaw].B0.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmhapke[ilaw].B0.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmhapke[ilaw].B0.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
          }
          harmhapke[ilaw].B0.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmhapke[ilaw].B0.scalefactor[j].val = -9.99;  /* dummy values */
            harmhapke[ilaw].B0.scalefactor[j].state = 'c';
          }
          state = mod->photo.optical[ilaw].inhohapke.global.B0.state;
          for (f=0; f<nf; f++)
            if (mod->photo.optical[ilaw].inhohapke.local[c][f].B0.state == 'f')
              state = 'f';
          for (l=0; l<=L; l++)
            for (m=0; m<=l; m++) {
              harmhapke[ilaw].B0.a[l][m].val = 0.0;
              if (m > 0)
                harmhapke[ilaw].B0.b[l][m].val = 0.0;
              for (f=0; f<nf; f++) {
                harmhapke[ilaw].B0.a[l][m].val
                       += afactor[l][m][f]*mod->photo.optical[ilaw].inhohapke.local[c][f].B0.val;
                if (m > 0)
                  harmhapke[ilaw].B0.b[l][m].val
                         += bfactor[l][m][f]*mod->photo.optical[ilaw].inhohapke.local[c][f].B0.val;
              }
              harmhapke[ilaw].B0.a[l][m].state = state;
              if (m > 0)
                harmhapke[ilaw].B0.b[l][m].state = state;
            }
  
          L = par->opt_g_nharm;
          harmhapke[ilaw].g.nhar = L;
          harmhapke[ilaw].g.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmhapke[ilaw].g.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmhapke[ilaw].g.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmhapke[ilaw].g.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
          }
          harmhapke[ilaw].g.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmhapke[ilaw].g.scalefactor[j].val = -9.99;  /* dummy values */
            harmhapke[ilaw].g.scalefactor[j].state = 'c';
          }
          state = mod->photo.optical[ilaw].inhohapke.global.g.state;
          for (f=0; f<nf; f++)
            if (mod->photo.optical[ilaw].inhohapke.local[c][f].g.state == 'f')
              state = 'f';
          for (l=0; l<=L; l++)
            for (m=0; m<=l; m++) {
              harmhapke[ilaw].g.a[l][m].val = 0.0;
              if (m > 0)
                harmhapke[ilaw].g.b[l][m].val = 0.0;
              for (f=0; f<nf; f++) {
                harmhapke[ilaw].g.a[l][m].val
                       += afactor[l][m][f]*mod->photo.optical[ilaw].inhohapke.local[c][f].g.val;
                if (m > 0)
                  harmhapke[ilaw].g.b[l][m].val
                         += bfactor[l][m][f]*mod->photo.optical[ilaw].inhohapke.local[c][f].g.val;
              }
              harmhapke[ilaw].g.a[l][m].state = state;
              if (m > 0)
                harmhapke[ilaw].g.b[l][m].state = state;
            }
  
          L = par->opt_theta_nharm;
          harmhapke[ilaw].theta.nhar = L;
          harmhapke[ilaw].theta.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmhapke[ilaw].theta.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmhapke[ilaw].theta.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmhapke[ilaw].theta.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
          }
          harmhapke[ilaw].theta.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmhapke[ilaw].theta.scalefactor[j].val = -9.99;  /* dummy values */
            harmhapke[ilaw].theta.scalefactor[j].state = 'c';
          }
          state = mod->photo.optical[ilaw].inhohapke.global.theta.state;
          for (f=0; f<nf; f++)
            if (mod->photo.optical[ilaw].inhohapke.local[c][f].theta.state == 'f')
              state = 'f';
          for (l=0; l<=L; l++)
            for (m=0; m<=l; m++) {
              harmhapke[ilaw].theta.a[l][m].val = 0.0;
              if (m > 0)
                harmhapke[ilaw].theta.b[l][m].val = 0.0;
              for (f=0; f<nf; f++) {
                harmhapke[ilaw].theta.a[l][m].val
                       += afactor[l][m][f]*mod->photo.optical[ilaw].inhohapke.local[c][f].theta.val;
                if (m > 0)
                  harmhapke[ilaw].theta.b[l][m].val
                         += bfactor[l][m][f]*mod->photo.optical[ilaw].inhohapke.local[c][f].theta.val;
              }
              harmhapke[ilaw].theta.a[l][m].state = state;
              if (m > 0)
                harmhapke[ilaw].theta.b[l][m].state = state;
            }
  
          harmhapke[ilaw].local = (struct hapke_t **) calloc( nc, sizeof( struct hapke_t *));
          harmhapke[ilaw].local[c] = (struct hapke_t *) calloc( nf, sizeof( struct hapke_t));
  
          mod->photo.optical[ilaw].harmhapke = harmhapke[ilaw];
          break;
      case KAASALAINEN:
          mod->photo.opttype[ilaw] = HARMKAAS;
  
          L = par->opt_R_nharm;
          harmkaas[ilaw].R.nhar = L;
          harmkaas[ilaw].R.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmkaas[ilaw].R.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmkaas[ilaw].R.a_save = (double **) calloc( L+1, sizeof( struct param_t *));
          harmkaas[ilaw].R.b_save = (double **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmkaas[ilaw].R.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmkaas[ilaw].R.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmkaas[ilaw].R.a_save[l] = (double *) calloc( l+1, sizeof( struct param_t));
            harmkaas[ilaw].R.b_save[l] = (double *) calloc( l+1, sizeof( struct param_t));
          }
          harmkaas[ilaw].R.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmkaas[ilaw].R.scalefactor[j].val = -9.99;  /* dummy values */
            harmkaas[ilaw].R.scalefactor[j].state = 'c';
          }
          harmkaas[ilaw].R.a[0][0].val = mod->photo.optical[ilaw].kaas.R.val;
          harmkaas[ilaw].R.a[0][0].state = mod->photo.optical[ilaw].kaas.R.state;
          harmkaas[ilaw].R.a_save[0][0] = harmkaas[ilaw].R.a[0][0].val;
          for (l=1; l<=L; l++)
            for (m=0; m<=l; m++) {
              harmkaas[ilaw].R.a[l][m].val = 0.0;
              harmkaas[ilaw].R.a[l][m].state = mod->photo.optical[ilaw].kaas.R.state;
              harmkaas[ilaw].R.a_save[l][m] = harmkaas[ilaw].R.a[l][m].val;
              if (m > 0) {
                harmkaas[ilaw].R.b[l][m].val = 0.0;
                harmkaas[ilaw].R.b[l][m].state = mod->photo.optical[ilaw].kaas.R.state;
                harmkaas[ilaw].R.b_save[l][m] = harmkaas[ilaw].R.b[l][m].val;
              }
            }
  
          L = par->opt_wt_nharm;
          harmkaas[ilaw].wt.nhar = L;
          harmkaas[ilaw].wt.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmkaas[ilaw].wt.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmkaas[ilaw].wt.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmkaas[ilaw].wt.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
          }
          harmkaas[ilaw].wt.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmkaas[ilaw].wt.scalefactor[j].val = -9.99;  /* dummy values */
            harmkaas[ilaw].wt.scalefactor[j].state = 'c';
          }
          harmkaas[ilaw].wt.a[0][0].val = mod->photo.optical[ilaw].kaas.wt.val;
          harmkaas[ilaw].wt.a[0][0].state = mod->photo.optical[ilaw].kaas.wt.state;
          for (l=1; l<=L; l++)
            for (m=0; m<=l; m++) {
              harmkaas[ilaw].wt.a[l][m].val = 0.0;
              harmkaas[ilaw].wt.a[l][m].state = mod->photo.optical[ilaw].kaas.wt.state;
              if (m > 0) {
                harmkaas[ilaw].wt.b[l][m].val = 0.0;
                harmkaas[ilaw].wt.b[l][m].state = mod->photo.optical[ilaw].kaas.wt.state;
              }
            }
  
          L = par->opt_A0_nharm;
          harmkaas[ilaw].A0.nhar = L;
          harmkaas[ilaw].A0.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmkaas[ilaw].A0.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmkaas[ilaw].A0.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmkaas[ilaw].A0.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
          }
          harmkaas[ilaw].A0.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmkaas[ilaw].A0.scalefactor[j].val = -9.99;  /* dummy values */
            harmkaas[ilaw].A0.scalefactor[j].state = 'c';
          }
          harmkaas[ilaw].A0.a[0][0].val = mod->photo.optical[ilaw].kaas.A0.val;
          harmkaas[ilaw].A0.a[0][0].state = mod->photo.optical[ilaw].kaas.A0.state;
          for (l=1; l<=L; l++)
            for (m=0; m<=l; m++) {
              harmkaas[ilaw].A0.a[l][m].val = 0.0;
              harmkaas[ilaw].A0.a[l][m].state = mod->photo.optical[ilaw].kaas.A0.state;
              if (m > 0) {
                harmkaas[ilaw].A0.b[l][m].val = 0.0;
                harmkaas[ilaw].A0.b[l][m].state = mod->photo.optical[ilaw].kaas.A0.state;
              }
            }
  
          L = par->opt_D_nharm;
          harmkaas[ilaw].D.nhar = L;
          harmkaas[ilaw].D.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmkaas[ilaw].D.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmkaas[ilaw].D.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmkaas[ilaw].D.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
          }
          harmkaas[ilaw].D.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmkaas[ilaw].D.scalefactor[j].val = -9.99;  /* dummy values */
            harmkaas[ilaw].D.scalefactor[j].state = 'c';
          }
          harmkaas[ilaw].D.a[0][0].val = mod->photo.optical[ilaw].kaas.D.val;
          harmkaas[ilaw].D.a[0][0].state = mod->photo.optical[ilaw].kaas.D.state;
          for (l=1; l<=L; l++)
            for (m=0; m<=l; m++) {
              harmkaas[ilaw].D.a[l][m].val = 0.0;
              harmkaas[ilaw].D.a[l][m].state = mod->photo.optical[ilaw].kaas.D.state;
              if (m > 0) {
                harmkaas[ilaw].D.b[l][m].val = 0.0;
                harmkaas[ilaw].D.b[l][m].state = mod->photo.optical[ilaw].kaas.D.state;
              }
            }
  
          L = par->opt_k_nharm;
          harmkaas[ilaw].k.nhar = L;
          harmkaas[ilaw].k.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmkaas[ilaw].k.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmkaas[ilaw].k.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmkaas[ilaw].k.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
          }
          harmkaas[ilaw].k.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmkaas[ilaw].k.scalefactor[j].val = -9.99;  /* dummy values */
            harmkaas[ilaw].k.scalefactor[j].state = 'c';
          }
          harmkaas[ilaw].k.a[0][0].val = mod->photo.optical[ilaw].kaas.k.val;
          harmkaas[ilaw].k.a[0][0].state = mod->photo.optical[ilaw].kaas.k.state;
          for (l=1; l<=L; l++)
            for (m=0; m<=l; m++) {
              harmkaas[ilaw].k.a[l][m].val = 0.0;
              harmkaas[ilaw].k.a[l][m].state = mod->photo.optical[ilaw].kaas.k.state;
              if (m > 0) {
                harmkaas[ilaw].k.b[l][m].val = 0.0;
                harmkaas[ilaw].k.b[l][m].state = mod->photo.optical[ilaw].kaas.k.state;
              }
            }
  
          harmkaas[ilaw].local = (struct kaas_t **) calloc( nc, sizeof( struct kaas_t *));
          harmkaas[ilaw].local[c] = (struct kaas_t *) calloc( nf, sizeof( struct kaas_t));
  
          mod->photo.optical[ilaw].harmkaas = harmkaas[ilaw];
          break;
      case HARMKAAS:
          L = par->opt_R_nharm;
          harmkaas[ilaw].R.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmkaas[ilaw].R.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmkaas[ilaw].R.a_save = (double **) calloc( L+1, sizeof( struct param_t *));
          harmkaas[ilaw].R.b_save = (double **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmkaas[ilaw].R.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmkaas[ilaw].R.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmkaas[ilaw].R.a_save[l] = (double *) calloc( l+1, sizeof( struct param_t));
            harmkaas[ilaw].R.b_save[l] = (double *) calloc( l+1, sizeof( struct param_t));
          }
          for (l=0; l<=L; l++)
            for (m=0; m<=l; m++) {
              if (l <= mod->photo.optical[ilaw].harmkaas.R.nhar) {
                  harmkaas[ilaw].R.a[l][m].val = mod->photo.optical[ilaw].harmkaas.R.a[l][m].val;
                  harmkaas[ilaw].R.a[l][m].state = mod->photo.optical[ilaw].harmkaas.R.a[l][m].state;
                  if (m > 0) {
                    harmkaas[ilaw].R.b[l][m].val = mod->photo.optical[ilaw].harmkaas.R.b[l][m].val;
                    harmkaas[ilaw].R.b[l][m].state = mod->photo.optical[ilaw].harmkaas.R.b[l][m].state;
                  }
              } else {
                  harmkaas[ilaw].R.a[l][m].val = 0.0;
                  harmkaas[ilaw].R.a[l][m].state = 'f';
                  if (m > 0) {
                    harmkaas[ilaw].R.b[l][m].val = 0.0;
                    harmkaas[ilaw].R.b[l][m].state = 'f';
                  }
              }
              harmkaas[ilaw].R.a_save[l][m] = harmkaas[ilaw].R.a[l][m].val;
              if (m > 0)
                harmkaas[ilaw].R.b_save[l][m] = harmkaas[ilaw].R.b[l][m].val;
            }
          harmkaas[ilaw].R.nhar = L;
          harmkaas[ilaw].R.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmkaas[ilaw].R.scalefactor[j].val = -9.99;  /* dummy values */
            harmkaas[ilaw].R.scalefactor[j].state = 'c';
          }
  
          L = par->opt_wt_nharm;
          harmkaas[ilaw].wt.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmkaas[ilaw].wt.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmkaas[ilaw].wt.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmkaas[ilaw].wt.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
          }
          for (l=0; l<=L; l++)
            for (m=0; m<=l; m++) {
              if (l <= mod->photo.optical[ilaw].harmkaas.wt.nhar) {
                  harmkaas[ilaw].wt.a[l][m].val = mod->photo.optical[ilaw].harmkaas.wt.a[l][m].val;
                  harmkaas[ilaw].wt.a[l][m].state = mod->photo.optical[ilaw].harmkaas.wt.a[l][m].state;
                  if (m > 0) {
                    harmkaas[ilaw].wt.b[l][m].val = mod->photo.optical[ilaw].harmkaas.wt.b[l][m].val;
                    harmkaas[ilaw].wt.b[l][m].state = mod->photo.optical[ilaw].harmkaas.wt.b[l][m].state;
                  }
              } else {
                  harmkaas[ilaw].wt.a[l][m].val = 0.0;
                  harmkaas[ilaw].wt.a[l][m].state = 'f';
                  if (m > 0) {
                    harmkaas[ilaw].wt.b[l][m].val = 0.0;
                    harmkaas[ilaw].wt.b[l][m].state = 'f';
                  }
              }
            }
          harmkaas[ilaw].wt.nhar = L;
          harmkaas[ilaw].wt.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmkaas[ilaw].wt.scalefactor[j].val = -9.99;  /* dummy values */
            harmkaas[ilaw].wt.scalefactor[j].state = 'c';
          }
  
          L = par->opt_A0_nharm;
          harmkaas[ilaw].A0.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmkaas[ilaw].A0.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmkaas[ilaw].A0.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmkaas[ilaw].A0.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
          }
          for (l=0; l<=L; l++)
            for (m=0; m<=l; m++) {
              if (l <= mod->photo.optical[ilaw].harmkaas.A0.nhar) {
                  harmkaas[ilaw].A0.a[l][m].val = mod->photo.optical[ilaw].harmkaas.A0.a[l][m].val;
                  harmkaas[ilaw].A0.a[l][m].state = mod->photo.optical[ilaw].harmkaas.A0.a[l][m].state;
                  if (m > 0) {
                    harmkaas[ilaw].A0.b[l][m].val = mod->photo.optical[ilaw].harmkaas.A0.b[l][m].val;
                    harmkaas[ilaw].A0.b[l][m].state = mod->photo.optical[ilaw].harmkaas.A0.b[l][m].state;
                  }
              } else {
                  harmkaas[ilaw].A0.a[l][m].val = 0.0;
                  harmkaas[ilaw].A0.a[l][m].state = 'f';
                  if (m > 0) {
                    harmkaas[ilaw].A0.b[l][m].val = 0.0;
                    harmkaas[ilaw].A0.b[l][m].state = 'f';
                  }
              }
            }
          harmkaas[ilaw].A0.nhar = L;
          harmkaas[ilaw].A0.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmkaas[ilaw].A0.scalefactor[j].val = -9.99;  /* dummy values */
            harmkaas[ilaw].A0.scalefactor[j].state = 'c';
          }
  
          L = par->opt_D_nharm;
          harmkaas[ilaw].D.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmkaas[ilaw].D.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmkaas[ilaw].D.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmkaas[ilaw].D.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
          }
          for (l=0; l<=L; l++)
            for (m=0; m<=l; m++) {
              if (l <= mod->photo.optical[ilaw].harmkaas.D.nhar) {
                  harmkaas[ilaw].D.a[l][m].val = mod->photo.optical[ilaw].harmkaas.D.a[l][m].val;
                  harmkaas[ilaw].D.a[l][m].state = mod->photo.optical[ilaw].harmkaas.D.a[l][m].state;
                  if (m > 0) {
                    harmkaas[ilaw].D.b[l][m].val = mod->photo.optical[ilaw].harmkaas.D.b[l][m].val;
                    harmkaas[ilaw].D.b[l][m].state = mod->photo.optical[ilaw].harmkaas.D.b[l][m].state;
                  }
              } else {
                  harmkaas[ilaw].D.a[l][m].val = 0.0;
                  harmkaas[ilaw].D.a[l][m].state = 'f';
                  if (m > 0) {
                    harmkaas[ilaw].D.b[l][m].val = 0.0;
                    harmkaas[ilaw].D.b[l][m].state = 'f';
                  }
              }
            }
          harmkaas[ilaw].D.nhar = L;
          harmkaas[ilaw].D.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmkaas[ilaw].D.scalefactor[j].val = -9.99;  /* dummy values */
            harmkaas[ilaw].D.scalefactor[j].state = 'c';
          }
  
          L = par->opt_k_nharm;
          harmkaas[ilaw].k.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmkaas[ilaw].k.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmkaas[ilaw].k.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmkaas[ilaw].k.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
          }
          for (l=0; l<=L; l++)
            for (m=0; m<=l; m++) {
              if (l <= mod->photo.optical[ilaw].harmkaas.k.nhar) {
                  harmkaas[ilaw].k.a[l][m].val = mod->photo.optical[ilaw].harmkaas.k.a[l][m].val;
                  harmkaas[ilaw].k.a[l][m].state = mod->photo.optical[ilaw].harmkaas.k.a[l][m].state;
                  if (m > 0) {
                    harmkaas[ilaw].k.b[l][m].val = mod->photo.optical[ilaw].harmkaas.k.b[l][m].val;
                    harmkaas[ilaw].k.b[l][m].state = mod->photo.optical[ilaw].harmkaas.k.b[l][m].state;
                  }
              } else {
                  harmkaas[ilaw].k.a[l][m].val = 0.0;
                  harmkaas[ilaw].k.a[l][m].state = 'f';
                  if (m > 0) {
                    harmkaas[ilaw].k.b[l][m].val = 0.0;
                    harmkaas[ilaw].k.b[l][m].state = 'f';
                  }
              }
            }
          harmkaas[ilaw].k.nhar = L;
          harmkaas[ilaw].k.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmkaas[ilaw].k.scalefactor[j].val = -9.99;  /* dummy values */
            harmkaas[ilaw].k.scalefactor[j].state = 'c';
          }
  
          harmkaas[ilaw].local = (struct kaas_t **) calloc( nc, sizeof( struct kaas_t *));
          harmkaas[ilaw].local[c] = (struct kaas_t *) calloc( nf, sizeof( struct kaas_t));
  
          mod->photo.optical[ilaw].harmkaas = harmkaas[ilaw];
          break;
      case INHOKAAS:
          mod->photo.opttype[ilaw] = HARMKAAS;
  
          L = par->opt_R_nharm;
          harmkaas[ilaw].R.nhar = L;
          harmkaas[ilaw].R.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmkaas[ilaw].R.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmkaas[ilaw].R.a_save = (double **) calloc( L+1, sizeof( struct param_t *));
          harmkaas[ilaw].R.b_save = (double **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmkaas[ilaw].R.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmkaas[ilaw].R.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmkaas[ilaw].R.a_save[l] = (double *) calloc( l+1, sizeof( struct param_t));
            harmkaas[ilaw].R.b_save[l] = (double *) calloc( l+1, sizeof( struct param_t));
          }
          harmkaas[ilaw].R.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmkaas[ilaw].R.scalefactor[j].val = -9.99;  /* dummy values */
            harmkaas[ilaw].R.scalefactor[j].state = 'c';
          }
          state = mod->photo.optical[ilaw].inhokaas.global.R.state;
          for (f=0; f<nf; f++)
            if (mod->photo.optical[ilaw].inhokaas.local[c][f].R.state == 'f')
              state = 'f';
          for (l=0; l<=L; l++)
            for (m=0; m<=l; m++) {
              harmkaas[ilaw].R.a[l][m].val = 0.0;
              if (m > 0)
                harmkaas[ilaw].R.b[l][m].val = 0.0;
              for (f=0; f<nf; f++) {
                harmkaas[ilaw].R.a[l][m].val
                       += afactor[l][m][f]*mod->photo.optical[ilaw].inhokaas.local[c][f].R.val;
                if (m > 0)
                  harmkaas[ilaw].R.b[l][m].val
                         += bfactor[l][m][f]*mod->photo.optical[ilaw].inhokaas.local[c][f].R.val;
              }
              harmkaas[ilaw].R.a[l][m].state = state;
              harmkaas[ilaw].R.a_save[l][m] = harmkaas[ilaw].R.a[l][m].val;
              if (m > 0) {
                harmkaas[ilaw].R.b[l][m].state = state;
                harmkaas[ilaw].R.b_save[l][m] = harmkaas[ilaw].R.b[l][m].val;
              }
            }
  
          L = par->opt_wt_nharm;
          harmkaas[ilaw].wt.nhar = L;
          harmkaas[ilaw].wt.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmkaas[ilaw].wt.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmkaas[ilaw].wt.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmkaas[ilaw].wt.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
          }
          harmkaas[ilaw].wt.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmkaas[ilaw].wt.scalefactor[j].val = -9.99;  /* dummy values */
            harmkaas[ilaw].wt.scalefactor[j].state = 'c';
          }
          state = mod->photo.optical[ilaw].inhokaas.global.wt.state;
          for (f=0; f<nf; f++)
            if (mod->photo.optical[ilaw].inhokaas.local[c][f].wt.state == 'f')
              state = 'f';
          for (l=0; l<=L; l++)
            for (m=0; m<=l; m++) {
              harmkaas[ilaw].wt.a[l][m].val = 0.0;
              if (m > 0)
                harmkaas[ilaw].wt.b[l][m].val = 0.0;
              for (f=0; f<nf; f++) {
                harmkaas[ilaw].wt.a[l][m].val
                       += afactor[l][m][f]*mod->photo.optical[ilaw].inhokaas.local[c][f].wt.val;
                if (m > 0)
                  harmkaas[ilaw].wt.b[l][m].val
                         += bfactor[l][m][f]*mod->photo.optical[ilaw].inhokaas.local[c][f].wt.val;
              }
              harmkaas[ilaw].wt.a[l][m].state = state;
              if (m > 0)
                harmkaas[ilaw].wt.b[l][m].state = state;
            }
  
          L = par->opt_A0_nharm;
          harmkaas[ilaw].A0.nhar = L;
          harmkaas[ilaw].A0.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmkaas[ilaw].A0.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmkaas[ilaw].A0.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmkaas[ilaw].A0.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
          }
          harmkaas[ilaw].A0.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmkaas[ilaw].A0.scalefactor[j].val = -9.99;  /* dummy values */
            harmkaas[ilaw].A0.scalefactor[j].state = 'c';
          }
          state = mod->photo.optical[ilaw].inhokaas.global.A0.state;
          for (f=0; f<nf; f++)
            if (mod->photo.optical[ilaw].inhokaas.local[c][f].A0.state == 'f')
              state = 'f';
          for (l=0; l<=L; l++)
            for (m=0; m<=l; m++) {
              harmkaas[ilaw].A0.a[l][m].val = 0.0;
              if (m > 0)
                harmkaas[ilaw].A0.b[l][m].val = 0.0;
              for (f=0; f<nf; f++) {
                harmkaas[ilaw].A0.a[l][m].val
                       += afactor[l][m][f]*mod->photo.optical[ilaw].inhokaas.local[c][f].A0.val;
                if (m > 0)
                  harmkaas[ilaw].A0.b[l][m].val
                         += bfactor[l][m][f]*mod->photo.optical[ilaw].inhokaas.local[c][f].A0.val;
              }
              harmkaas[ilaw].A0.a[l][m].state = state;
              if (m > 0)
                harmkaas[ilaw].A0.b[l][m].state = state;
            }
  
          L = par->opt_D_nharm;
          harmkaas[ilaw].D.nhar = L;
          harmkaas[ilaw].D.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmkaas[ilaw].D.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmkaas[ilaw].D.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmkaas[ilaw].D.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
          }
          harmkaas[ilaw].D.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmkaas[ilaw].D.scalefactor[j].val = -9.99;  /* dummy values */
            harmkaas[ilaw].D.scalefactor[j].state = 'c';
          }
          state = mod->photo.optical[ilaw].inhokaas.global.D.state;
          for (f=0; f<nf; f++)
            if (mod->photo.optical[ilaw].inhokaas.local[c][f].D.state == 'f')
              state = 'f';
          for (l=0; l<=L; l++)
            for (m=0; m<=l; m++) {
              harmkaas[ilaw].D.a[l][m].val = 0.0;
              if (m > 0)
                harmkaas[ilaw].D.b[l][m].val = 0.0;
              for (f=0; f<nf; f++) {
                harmkaas[ilaw].D.a[l][m].val
                       += afactor[l][m][f]*mod->photo.optical[ilaw].inhokaas.local[c][f].D.val;
                if (m > 0)
                  harmkaas[ilaw].D.b[l][m].val
                         += bfactor[l][m][f]*mod->photo.optical[ilaw].inhokaas.local[c][f].D.val;
              }
              harmkaas[ilaw].D.a[l][m].state = state;
              if (m > 0)
                harmkaas[ilaw].D.b[l][m].state = state;
            }
  
          L = par->opt_k_nharm;
          harmkaas[ilaw].k.nhar = L;
          harmkaas[ilaw].k.a = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          harmkaas[ilaw].k.b = (struct param_t **) calloc( L+1, sizeof( struct param_t *));
          for (l=0; l<=L; l++) {
            harmkaas[ilaw].k.a[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
            harmkaas[ilaw].k.b[l] = (struct param_t *) calloc( l+1, sizeof( struct param_t));
          }
          harmkaas[ilaw].k.ntheta = -1;  /* dummy value */
          for (j=0; j<=2; j++) {
            harmkaas[ilaw].k.scalefactor[j].val = -9.99;  /* dummy values */
            harmkaas[ilaw].k.scalefactor[j].state = 'c';
          }
          state = mod->photo.optical[ilaw].inhokaas.global.k.state;
          for (f=0; f<nf; f++)
            if (mod->photo.optical[ilaw].inhokaas.local[c][f].k.state == 'f')
              state = 'f';
          for (l=0; l<=L; l++)
            for (m=0; m<=l; m++) {
              harmkaas[ilaw].k.a[l][m].val = 0.0;
              if (m > 0)
                harmkaas[ilaw].k.b[l][m].val = 0.0;
              for (f=0; f<nf; f++) {
                harmkaas[ilaw].k.a[l][m].val
                       += afactor[l][m][f]*mod->photo.optical[ilaw].inhokaas.local[c][f].k.val;
                if (m > 0)
                  harmkaas[ilaw].k.b[l][m].val
                         += bfactor[l][m][f]*mod->photo.optical[ilaw].inhokaas.local[c][f].k.val;
              }
              harmkaas[ilaw].k.a[l][m].state = state;
              if (m > 0)
                harmkaas[ilaw].k.b[l][m].state = state;
            }
  
          harmkaas[ilaw].local = (struct kaas_t **) calloc( nc, sizeof( struct kaas_t *));
          harmkaas[ilaw].local[c] = (struct kaas_t *) calloc( nf, sizeof( struct kaas_t));
  
          mod->photo.optical[ilaw].harmkaas = harmkaas[ilaw];
          break;
      default:
          bailout("photoharm: can't handle this optical law yet\n");
      }
    }
  }

  /*  Compute "local" (facet) photometric parameters  */

  realize_photo( par, mod, 1.0, 1.0, 0);

  /*  Clean up storage space  */

  if (faceted_inputlaw) {
    free_d3tensor( afactor, 0, Lmax, 0, Lmax, 0, nf-1);
    if (Lmax > 0)
      free_d3tensor( bfactor, 1, Lmax, 1, Lmax, 0, nf-1);
  }
}
