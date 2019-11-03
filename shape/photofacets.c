/*****************************************************************************************
                                                                            photofacets.c

Convert radar and/or optical scattering laws to inhomogeneous scattering laws (inhocosine,
inholommel, inhohapke, inhokaas) for which the various photometric parameters are
explicitly specified facet by facet

Modified 2014 February 15 by CM:
    Implement multiple radar and optical scattering laws

Modified 2011 September 2 by CM:
    Add "harmlambert" and "inholambert" optical scattering laws

Modified 2007 January 13 by CM:
    Fix bug introduced when "vary_radalb" and "vary_optalb" parameters
        were added: initialize various "R_save" and "w_save" variables
        (just as is done in read_mod.c)

Modified 2006 October 1 by CM:
    Added three new arguments to realize_photo

Written 2005 September 8 by CM
*****************************************************************************************/

#include "head.h"

void photofacets( struct par_t *par, struct mod_t *mod)
{
  int ilaw, nc, nf, c, f;

  /* Convert radar/optical scattering laws into faceted inhomogeneous laws  */

  if (par->radfacets) {
    for (ilaw=0; ilaw<mod->photo.nradlaws; ilaw++) {
      switch (mod->photo.radtype[ilaw]) {
      case COSINELAW_DIFF:
          mod->photo.radtype[ilaw] = INHOCOSINE_DIFF;
          mod->photo.radar[ilaw].inhocosine.global.R.val
                 = mod->photo.radar[ilaw].RC.R.val;
          mod->photo.radar[ilaw].inhocosine.global.R.state
                 = par->rad_global_R_state;
          mod->photo.radar[ilaw].inhocosine.global.R_save
                 = mod->photo.radar[ilaw].inhocosine.global.R.val;
          mod->photo.radar[ilaw].inhocosine.global.C.val
                 = mod->photo.radar[ilaw].RC.C.val;
          mod->photo.radar[ilaw].inhocosine.global.C.state
                 = par->rad_global_C_state;
          nc = mod->shape.ncomp;
          mod->photo.radar[ilaw].inhocosine.local
                  = (struct RC_t **) calloc( nc, sizeof( struct RC_t *));
          for (c=0; c<nc; c++) {
            nf = mod->shape.comp[c].real.nf;
            mod->photo.radar[ilaw].inhocosine.local[c]
                    = (struct RC_t *) calloc( nf, sizeof( struct RC_t));
            for (f=0; f<nf; f++) {
              mod->photo.radar[ilaw].inhocosine.local[c][f].R.val
                     = mod->photo.radar[ilaw].RC.R.val;
              mod->photo.radar[ilaw].inhocosine.local[c][f].R.state
                     = par->rad_local_R_state;
              mod->photo.radar[ilaw].inhocosine.local[c][f].R_save
                     = mod->photo.radar[ilaw].inhocosine.local[c][f].R.val;
              mod->photo.radar[ilaw].inhocosine.local[c][f].C.val
                     = mod->photo.radar[ilaw].RC.C.val;
              mod->photo.radar[ilaw].inhocosine.local[c][f].C.state
                     = par->rad_local_C_state;
            }
          }
          break;
      case HARMCOSINE_DIFF:
          mod->photo.radtype[ilaw] = INHOCOSINE_DIFF;
          mod->photo.radar[ilaw].inhocosine.global.R.val
                 = mod->photo.radar[ilaw].harmcosine.R.a[0][0].val;
          mod->photo.radar[ilaw].inhocosine.global.R.state
                 = par->rad_global_R_state;
          mod->photo.radar[ilaw].inhocosine.global.R_save
                 = mod->photo.radar[ilaw].inhocosine.global.R.val;
          mod->photo.radar[ilaw].inhocosine.global.C.val
                 = mod->photo.radar[ilaw].harmcosine.C.a[0][0].val;
          mod->photo.radar[ilaw].inhocosine.global.C.state
                 = par->rad_global_C_state;
          nc = mod->shape.ncomp;
          mod->photo.radar[ilaw].inhocosine.local
                  = (struct RC_t **) calloc( nc, sizeof( struct RC_t *));
          for (c=0; c<nc; c++) {
            nf = mod->shape.comp[c].real.nf;
            mod->photo.radar[ilaw].inhocosine.local[c]
                    = (struct RC_t *) calloc( nf, sizeof( struct RC_t));
            for (f=0; f<nf; f++) {
              mod->photo.radar[ilaw].inhocosine.local[c][f].R.val
                     = mod->photo.radar[ilaw].harmcosine.local[c][f].R.val;
              mod->photo.radar[ilaw].inhocosine.local[c][f].R.state
                     = par->rad_local_R_state;
              mod->photo.radar[ilaw].inhocosine.local[c][f].R_save
                     = mod->photo.radar[ilaw].inhocosine.local[c][f].R.val;
              mod->photo.radar[ilaw].inhocosine.local[c][f].C.val
                     = mod->photo.radar[ilaw].harmcosine.local[c][f].C.val;
              mod->photo.radar[ilaw].inhocosine.local[c][f].C.state
                     = par->rad_local_C_state;
            }
          }
          break;
      case INHOCOSINE_DIFF:
          mod->photo.radar[ilaw].inhocosine.global.R.state
                 = par->rad_global_R_state;
          mod->photo.radar[ilaw].inhocosine.global.C.state
                 = par->rad_global_C_state;
          nc = mod->shape.ncomp;
          for (c=0; c<nc; c++) {
            nf = mod->shape.comp[c].real.nf;
            for (f=0; f<nf; f++) {
              mod->photo.radar[ilaw].inhocosine.local[c][f].R.state
                     = par->rad_local_R_state;
              mod->photo.radar[ilaw].inhocosine.local[c][f].C.state
                     = par->rad_local_C_state;
            }
          }
          break;
      default:
          bailout("photofacets: can't handle this radar law yet\n");
      }
    }
  }

  if (par->optfacets) {
    for (ilaw=0; ilaw<mod->photo.noptlaws; ilaw++) {
      switch (mod->photo.opttype[ilaw]) {
      case LAMBERTLAW:
      case LOMMEL:
          if (mod->photo.opttype[ilaw] == LAMBERTLAW)
            mod->photo.opttype[ilaw] = INHOLAMBERT;
          else
            mod->photo.opttype[ilaw] = INHOLOMMEL;
          mod->photo.optical[ilaw].inhoR.global.R.val
                 = mod->photo.optical[ilaw].R.R.val;
          mod->photo.optical[ilaw].inhoR.global.R_save
                 = mod->photo.optical[ilaw].inhoR.global.R.val;
          mod->photo.optical[ilaw].inhoR.global.R.state
                 = par->opt_global_R_state;
          nc = mod->shape.ncomp;
          mod->photo.optical[ilaw].inhoR.local
                  = (struct R_t **) calloc( nc, sizeof( struct R_t *));
          for (c=0; c<nc; c++) {
            nf = mod->shape.comp[c].real.nf;
            mod->photo.optical[ilaw].inhoR.local[c]
                    = (struct R_t *) calloc( nf, sizeof( struct R_t));
            for (f=0; f<nf; f++) {
              mod->photo.optical[ilaw].inhoR.local[c][f].R.val
                     = mod->photo.optical[ilaw].R.R.val;
              mod->photo.optical[ilaw].inhoR.local[c][f].R.state
                     = par->opt_local_R_state;
              mod->photo.optical[ilaw].inhoR.local[c][f].R_save
                     = mod->photo.optical[ilaw].inhoR.local[c][f].R.val;
            }
          }
          break;
      case HARMLAMBERT:
      case HARMLOMMEL:
          if (mod->photo.opttype[ilaw] == HARMLAMBERT)
            mod->photo.opttype[ilaw] = INHOLAMBERT;
          else
            mod->photo.opttype[ilaw] = INHOLOMMEL;
          mod->photo.optical[ilaw].inhoR.global.R.val
                 = mod->photo.optical[ilaw].harmR.R.a[0][0].val;
          mod->photo.optical[ilaw].inhoR.global.R.state
                 = par->opt_global_R_state;
          mod->photo.optical[ilaw].inhoR.global.R_save
                 = mod->photo.optical[ilaw].inhoR.global.R.val;
          nc = mod->shape.ncomp;
          mod->photo.optical[ilaw].inhoR.local
                 = (struct R_t **) calloc( nc, sizeof( struct R_t *));
          for (c=0; c<nc; c++) {
            nf = mod->shape.comp[c].real.nf;
            mod->photo.optical[ilaw].inhoR.local[c]
                    = (struct R_t *) calloc( nf, sizeof( struct R_t));
            for (f=0; f<nf; f++) {
              mod->photo.optical[ilaw].inhoR.local[c][f].R.val
                     = mod->photo.optical[ilaw].harmR.local[c][f].R.val;
              mod->photo.optical[ilaw].inhoR.local[c][f].R.state
                     = par->opt_local_R_state;
              mod->photo.optical[ilaw].inhoR.local[c][f].R_save
                     = mod->photo.optical[ilaw].inhoR.local[c][f].R.val;
            }
          }
          break;
      case INHOLAMBERT:
      case INHOLOMMEL:
          mod->photo.optical[ilaw].inhoR.global.R.state
                 = par->opt_global_R_state;
          nc = mod->shape.ncomp;
          for (c=0; c<nc; c++) {
            nf = mod->shape.comp[c].real.nf;
            for (f=0; f<nf; f++) {
              mod->photo.optical[ilaw].inhoR.local[c][f].R.state
                     = par->opt_local_R_state;
            }
          }
          break;
      case HAPKE:
          mod->photo.opttype[ilaw] = INHOHAPKE;
          mod->photo.optical[ilaw].inhohapke.global.w.val
                 = mod->photo.optical[ilaw].hapke.w.val;
          mod->photo.optical[ilaw].inhohapke.global.w.state
                 = par->opt_global_w_state;
          mod->photo.optical[ilaw].inhohapke.global.w_save
                 = mod->photo.optical[ilaw].inhohapke.global.w.val;
          mod->photo.optical[ilaw].inhohapke.global.h.val
                 = mod->photo.optical[ilaw].hapke.h.val;
          mod->photo.optical[ilaw].inhohapke.global.h.state
                 = par->opt_global_h_state;
          mod->photo.optical[ilaw].inhohapke.global.B0.val
                 = mod->photo.optical[ilaw].hapke.B0.val;
          mod->photo.optical[ilaw].inhohapke.global.B0.state
                 = par->opt_global_B0_state;
          mod->photo.optical[ilaw].inhohapke.global.g.val
                 = mod->photo.optical[ilaw].hapke.g.val;
          mod->photo.optical[ilaw].inhohapke.global.g.state
                 = par->opt_global_g_state;
          mod->photo.optical[ilaw].inhohapke.global.theta.val
                 = mod->photo.optical[ilaw].hapke.theta.val;
          mod->photo.optical[ilaw].inhohapke.global.theta.state
                 = par->opt_global_theta_state;
          nc = mod->shape.ncomp;
          mod->photo.optical[ilaw].inhohapke.local
                  = (struct hapke_t **) calloc( nc, sizeof( struct hapke_t *));
          for (c=0; c<nc; c++) {
            nf = mod->shape.comp[c].real.nf;
            mod->photo.optical[ilaw].inhohapke.local[c]
                    = (struct hapke_t *) calloc( nf, sizeof( struct hapke_t));
            for (f=0; f<nf; f++) {
              mod->photo.optical[ilaw].inhohapke.local[c][f].w.val
                     = mod->photo.optical[ilaw].hapke.w.val;
              mod->photo.optical[ilaw].inhohapke.local[c][f].w.state
                     = par->opt_local_w_state;
              mod->photo.optical[ilaw].inhohapke.local[c][f].w_save
                     = mod->photo.optical[ilaw].inhohapke.local[c][f].w.val;
              mod->photo.optical[ilaw].inhohapke.local[c][f].h.val
                     = mod->photo.optical[ilaw].hapke.h.val;
              mod->photo.optical[ilaw].inhohapke.local[c][f].h.state
                     = par->opt_local_h_state;
              mod->photo.optical[ilaw].inhohapke.local[c][f].B0.val
                     = mod->photo.optical[ilaw].hapke.B0.val;
              mod->photo.optical[ilaw].inhohapke.local[c][f].B0.state
                     = par->opt_local_B0_state;
              mod->photo.optical[ilaw].inhohapke.local[c][f].g.val
                     = mod->photo.optical[ilaw].hapke.g.val;
              mod->photo.optical[ilaw].inhohapke.local[c][f].g.state
                     = par->opt_local_g_state;
              mod->photo.optical[ilaw].inhohapke.local[c][f].theta.val
                     = mod->photo.optical[ilaw].hapke.theta.val;
              mod->photo.optical[ilaw].inhohapke.local[c][f].theta.state
                     = par->opt_local_theta_state;
            }
          }
          break;
      case HARMHAPKE:
          mod->photo.opttype[ilaw] = INHOHAPKE;
          mod->photo.optical[ilaw].inhohapke.global.w.val
                 = mod->photo.optical[ilaw].harmhapke.w.a[0][0].val;
          mod->photo.optical[ilaw].inhohapke.global.w.state
                 = par->opt_global_w_state;
          mod->photo.optical[ilaw].inhohapke.global.w_save
                 = mod->photo.optical[ilaw].inhohapke.global.w.val;
          mod->photo.optical[ilaw].inhohapke.global.h.val
                 = mod->photo.optical[ilaw].harmhapke.h.a[0][0].val;
          mod->photo.optical[ilaw].inhohapke.global.h.state
                 = par->opt_global_h_state;
          mod->photo.optical[ilaw].inhohapke.global.B0.val
                 = mod->photo.optical[ilaw].harmhapke.B0.a[0][0].val;
          mod->photo.optical[ilaw].inhohapke.global.B0.state
                 = par->opt_global_B0_state;
          mod->photo.optical[ilaw].inhohapke.global.g.val
                 = mod->photo.optical[ilaw].harmhapke.g.a[0][0].val;
          mod->photo.optical[ilaw].inhohapke.global.g.state
                 = par->opt_global_g_state;
          mod->photo.optical[ilaw].inhohapke.global.theta.val
                 = mod->photo.optical[ilaw].harmhapke.theta.a[0][0].val;
          mod->photo.optical[ilaw].inhohapke.global.theta.state
                 = par->opt_global_theta_state;
          nc = mod->shape.ncomp;
          mod->photo.optical[ilaw].inhohapke.local
                  = (struct hapke_t **) calloc( nc, sizeof( struct hapke_t *));
          for (c=0; c<nc; c++) {
            nf = mod->shape.comp[c].real.nf;
            mod->photo.optical[ilaw].inhohapke.local[c]
                    = (struct hapke_t *) calloc( nf, sizeof( struct hapke_t));
            for (f=0; f<nf; f++) {
              mod->photo.optical[ilaw].inhohapke.local[c][f].w.val
                     = mod->photo.optical[ilaw].harmhapke.local[c][f].w.val;
              mod->photo.optical[ilaw].inhohapke.local[c][f].w.state
                     = par->opt_local_w_state;
              mod->photo.optical[ilaw].inhohapke.local[c][f].w_save
                     = mod->photo.optical[ilaw].inhohapke.local[c][f].w.val;
              mod->photo.optical[ilaw].inhohapke.local[c][f].h.val
                     = mod->photo.optical[ilaw].harmhapke.local[c][f].h.val;
              mod->photo.optical[ilaw].inhohapke.local[c][f].h.state
                     = par->opt_local_h_state;
              mod->photo.optical[ilaw].inhohapke.local[c][f].B0.val
                     = mod->photo.optical[ilaw].harmhapke.local[c][f].B0.val;
              mod->photo.optical[ilaw].inhohapke.local[c][f].B0.state
                     = par->opt_local_B0_state;
              mod->photo.optical[ilaw].inhohapke.local[c][f].g.val
                     = mod->photo.optical[ilaw].harmhapke.local[c][f].g.val;
              mod->photo.optical[ilaw].inhohapke.local[c][f].g.state
                     = par->opt_local_g_state;
              mod->photo.optical[ilaw].inhohapke.local[c][f].theta.val
                     = mod->photo.optical[ilaw].harmhapke.local[c][f].theta.val;
              mod->photo.optical[ilaw].inhohapke.local[c][f].theta.state
                     = par->opt_local_theta_state;
            }
          }
          break;
      case INHOHAPKE:
          mod->photo.optical[ilaw].inhohapke.global.w.state
                 = par->opt_global_w_state;
          mod->photo.optical[ilaw].inhohapke.global.h.state
                 = par->opt_global_h_state;
          mod->photo.optical[ilaw].inhohapke.global.B0.state
                 = par->opt_global_B0_state;
          mod->photo.optical[ilaw].inhohapke.global.g.state
                 = par->opt_global_g_state;
          mod->photo.optical[ilaw].inhohapke.global.theta.state
                 = par->opt_global_theta_state;
          nc = mod->shape.ncomp;
          for (c=0; c<nc; c++) {
            nf = mod->shape.comp[c].real.nf;
            for (f=0; f<nf; f++) {
              mod->photo.optical[ilaw].inhohapke.local[c][f].w.state
                     = par->opt_local_w_state;
              mod->photo.optical[ilaw].inhohapke.local[c][f].h.state
                     = par->opt_local_h_state;
              mod->photo.optical[ilaw].inhohapke.local[c][f].B0.state
                     = par->opt_local_B0_state;
              mod->photo.optical[ilaw].inhohapke.local[c][f].g.state
                     = par->opt_local_g_state;
              mod->photo.optical[ilaw].inhohapke.local[c][f].theta.state
                     = par->opt_local_theta_state;
            }
          }
          break;
      case KAASALAINEN:
          mod->photo.opttype[ilaw] = INHOKAAS;
          mod->photo.optical[ilaw].inhokaas.global.R.val
                 = mod->photo.optical[ilaw].kaas.R.val;
          mod->photo.optical[ilaw].inhokaas.global.R.state
                 = par->opt_global_R_state;
          mod->photo.optical[ilaw].inhokaas.global.R_save
                 = mod->photo.optical[ilaw].inhokaas.global.R.val;
          mod->photo.optical[ilaw].inhokaas.global.wt.val
                 = mod->photo.optical[ilaw].kaas.wt.val;
          mod->photo.optical[ilaw].inhokaas.global.wt.state
                 = par->opt_global_wt_state;
          mod->photo.optical[ilaw].inhokaas.global.A0.val
                 = mod->photo.optical[ilaw].kaas.A0.val;
          mod->photo.optical[ilaw].inhokaas.global.A0.state
                 = par->opt_global_A0_state;
          mod->photo.optical[ilaw].inhokaas.global.D.val
                 = mod->photo.optical[ilaw].kaas.D.val;
          mod->photo.optical[ilaw].inhokaas.global.D.state
                 = par->opt_global_D_state;
          mod->photo.optical[ilaw].inhokaas.global.k.val
                 = mod->photo.optical[ilaw].kaas.k.val;
          mod->photo.optical[ilaw].inhokaas.global.k.state
                 = par->opt_global_k_state;
          nc = mod->shape.ncomp;
          mod->photo.optical[ilaw].inhokaas.local
                  = (struct kaas_t **) calloc( nc, sizeof( struct kaas_t *));
          for (c=0; c<nc; c++) {
            nf = mod->shape.comp[c].real.nf;
            mod->photo.optical[ilaw].inhokaas.local[c]
                    = (struct kaas_t *) calloc( nf, sizeof( struct kaas_t));
            for (f=0; f<nf; f++) {
              mod->photo.optical[ilaw].inhokaas.local[c][f].R.val
                     = mod->photo.optical[ilaw].kaas.R.val;
              mod->photo.optical[ilaw].inhokaas.local[c][f].R.state
                     = par->opt_local_R_state;
              mod->photo.optical[ilaw].inhokaas.local[c][f].R_save
                     = mod->photo.optical[ilaw].inhokaas.local[c][f].R.val;
              mod->photo.optical[ilaw].inhokaas.local[c][f].wt.val
                     = mod->photo.optical[ilaw].kaas.wt.val;
              mod->photo.optical[ilaw].inhokaas.local[c][f].wt.state
                     = par->opt_local_wt_state;
              mod->photo.optical[ilaw].inhokaas.local[c][f].A0.val
                     = mod->photo.optical[ilaw].kaas.A0.val;
              mod->photo.optical[ilaw].inhokaas.local[c][f].A0.state
                     = par->opt_local_A0_state;
              mod->photo.optical[ilaw].inhokaas.local[c][f].D.val
                     = mod->photo.optical[ilaw].kaas.D.val;
              mod->photo.optical[ilaw].inhokaas.local[c][f].D.state
                     = par->opt_local_D_state;
              mod->photo.optical[ilaw].inhokaas.local[c][f].k.val
                     = mod->photo.optical[ilaw].kaas.k.val;
              mod->photo.optical[ilaw].inhokaas.local[c][f].k.state
                     = par->opt_local_k_state;
            }
          }
          break;
      case HARMKAAS:
          mod->photo.opttype[ilaw] = INHOKAAS;
          mod->photo.optical[ilaw].inhokaas.global.R.val
                 = mod->photo.optical[ilaw].harmkaas.R.a[0][0].val;
          mod->photo.optical[ilaw].inhokaas.global.R.state
                 = par->opt_global_R_state;
          mod->photo.optical[ilaw].inhokaas.global.R_save
                 = mod->photo.optical[ilaw].inhokaas.global.R.val;
          mod->photo.optical[ilaw].inhokaas.global.wt.val
                 = mod->photo.optical[ilaw].harmkaas.wt.a[0][0].val;
          mod->photo.optical[ilaw].inhokaas.global.wt.state
                 = par->opt_global_wt_state;
          mod->photo.optical[ilaw].inhokaas.global.A0.val
                 = mod->photo.optical[ilaw].harmkaas.A0.a[0][0].val;
          mod->photo.optical[ilaw].inhokaas.global.A0.state
                 = par->opt_global_A0_state;
          mod->photo.optical[ilaw].inhokaas.global.D.val
                 = mod->photo.optical[ilaw].harmkaas.D.a[0][0].val;
          mod->photo.optical[ilaw].inhokaas.global.D.state
                 = par->opt_global_D_state;
          mod->photo.optical[ilaw].inhokaas.global.k.val
                 = mod->photo.optical[ilaw].harmkaas.k.a[0][0].val;
          mod->photo.optical[ilaw].inhokaas.global.k.state
                 = par->opt_global_k_state;
          nc = mod->shape.ncomp;
          mod->photo.optical[ilaw].inhokaas.local
                  = (struct kaas_t **) calloc( nc, sizeof( struct kaas_t *));
          for (c=0; c<nc; c++) {
            nf = mod->shape.comp[c].real.nf;
            mod->photo.optical[ilaw].inhokaas.local[c]
                    = (struct kaas_t *) calloc( nf, sizeof( struct kaas_t));
            for (f=0; f<nf; f++) {
              mod->photo.optical[ilaw].inhokaas.local[c][f].R.val
                     = mod->photo.optical[ilaw].harmkaas.local[c][f].R.val;
              mod->photo.optical[ilaw].inhokaas.local[c][f].R.state
                     = par->opt_local_R_state;
              mod->photo.optical[ilaw].inhokaas.local[c][f].R_save
                     = mod->photo.optical[ilaw].inhokaas.local[c][f].R.val;
              mod->photo.optical[ilaw].inhokaas.local[c][f].wt.val
                     = mod->photo.optical[ilaw].harmkaas.local[c][f].wt.val;
              mod->photo.optical[ilaw].inhokaas.local[c][f].wt.state
                     = par->opt_local_wt_state;
              mod->photo.optical[ilaw].inhokaas.local[c][f].A0.val
                     = mod->photo.optical[ilaw].harmkaas.local[c][f].A0.val;
              mod->photo.optical[ilaw].inhokaas.local[c][f].A0.state
                     = par->opt_local_A0_state;
              mod->photo.optical[ilaw].inhokaas.local[c][f].D.val
                     = mod->photo.optical[ilaw].harmkaas.local[c][f].D.val;
              mod->photo.optical[ilaw].inhokaas.local[c][f].D.state
                     = par->opt_local_D_state;
              mod->photo.optical[ilaw].inhokaas.local[c][f].k.val
                     = mod->photo.optical[ilaw].harmkaas.local[c][f].k.val;
              mod->photo.optical[ilaw].inhokaas.local[c][f].k.state
                     = par->opt_local_k_state;
            }
          }
          break;
      case INHOKAAS:
          mod->photo.optical[ilaw].inhokaas.global.R.state
                 = par->opt_global_R_state;
          mod->photo.optical[ilaw].inhokaas.global.wt.state
                 = par->opt_global_wt_state;
          mod->photo.optical[ilaw].inhokaas.global.A0.state
                 = par->opt_global_A0_state;
          mod->photo.optical[ilaw].inhokaas.global.D.state
                 = par->opt_global_D_state;
          mod->photo.optical[ilaw].inhokaas.global.k.state
                 = par->opt_global_k_state;
          nc = mod->shape.ncomp;
          for (c=0; c<nc; c++) {
            nf = mod->shape.comp[c].real.nf;
            for (f=0; f<nf; f++) {
              mod->photo.optical[ilaw].inhokaas.local[c][f].R.state
                     = par->opt_local_R_state;
              mod->photo.optical[ilaw].inhokaas.local[c][f].wt.state
                     = par->opt_local_wt_state;
              mod->photo.optical[ilaw].inhokaas.local[c][f].A0.state
                     = par->opt_local_A0_state;
              mod->photo.optical[ilaw].inhokaas.local[c][f].D.state
                     = par->opt_local_D_state;
              mod->photo.optical[ilaw].inhokaas.local[c][f].k.state
                     = par->opt_local_k_state;
            }
          }
          break;
      default:
          bailout("photofacets: can't handle this optical law yet\n");
      }
    }
  }

  /*  Implement '=' state for photometric parameters  */

  realize_photo( par, mod, 1.0, 1.0, 0);
}
