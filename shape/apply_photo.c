/*****************************************************************************************
                                                                            apply_photo.c

For each plane-of-sky pixel, compute the model's scattered optical power per unit
projected (POS) area per unit solid angle per unit incident flux, and then sum these
values over the entire POS.  (The POS pixel area is multiplied in elsewhere.)

The expressions given here differ from the bidirectional reflectance functions defined by,
say, Hapke 1993: bidirectional reflectance includes an extra factor of
cos(scattering angle), since it is defined per unit surface area rather than per unit
projected area.

Modified 2014 February 12 by CM:
    Implement multiple optical scatering laws

Modified 2011 September 2 by CM:
    Add the "harmlambert" and "inholambert" optical scattering laws

Modified 2007 August 4 by CM:
    Add body parameter for use with the "orbit" action: it denotes which
        orbiting body's optical power contributions are being computed
        on this call to the routine
    Don't zero out blank-sky and shadowed POS pixels in the sky rendering
        (the pos->b matrix): do it instead in the calling routine by
        having it call the posclr routine.  This way apply_photo can be
        called twice for the "orbit" action, once for each orbiting body.
    Add comp matrix for POS frames

Modified 2006 October 1 by CM:
    Add "intensityfactor" parameter: account for POS pixel area,
        1 AU Sun-target distance, and solar apparent magnitude here
        rather than after calling the routine

Modified 2006 September 1 by CM and MCN:
    For inhomogeneous laws, add check that facet number pos->f[i][j]
        is nonnegative

Modified 2005 September 7 by CM:
    Implement the "harmlommel" "harmhapke" and "harmkaas" optical
        scattering laws

Modified 2005 August 8 by CM:
    Implement the "inhokaas" optical scattering law
    Add some (cosi > 0) checks
    Move "sum == 0" check to the end

Modified 2005 July 4 by CM:
    Changed structure name for the INHOLOMMEL optical scattering law

Modified 2005 March 1 by CM:
    Add NOLAW case

Modified 2005 January 25 by CM:
    Eliminate unused variables

Modified 2004 April 29 by CM:
    Modify Kaasalainen scattering law to use "wt" as the relative
        weighting factor (0 = pure Lommel-Seeliger, 1 = pure Lambert)
        rather than "c" (which ranged from 0 to infinity)

Modified 2004 March 25 by CM:
    hapke routine now takes phase rather than cos(phase) as argument

Modified 2004 February 29 by CM:
    Added comments
    Added Kaasalainen "Lommel-Seeliger + Lambert" scattering law
    Eliminated "type" argument, since this routine was only being
       used to handle optical scattering.  (Radar scattering is
       instead handled by the "radlaw" routine.)
    Added "phase" argument (solar phase angle) so that we can compute
       the phase just once per calculated lightcurve point (in read_dat)
       rather than computing it every time we call apply_photo
*****************************************************************************************/
#include "head.h"

#define TINY 1.0e-40

double apply_photo( struct mod_t *mod, int ilaw, double phase, double intensityfactor,
                    struct pos_t *pos, int body, int s, int frm)
{
  int i, j, c, f;
  double sum=0.0, scale, phasefunc, scale_lommsee, scale_lambert;
//  int dbg_occ = 0;
  switch (mod->photo.opttype[ilaw]) {
  case LAMBERTLAW:
      scale = mod->photo.optical[ilaw].R.R.val/PIE;
      for (i=pos->xlim[0]; i<=pos->xlim[1]; i++)
        for (j=pos->ylim[0]; j<=pos->ylim[1]; j++) {
          if (pos->cose[i][j] > 0.0 && pos->cosi[i][j] > 0.0
                                      && pos->body[i][j] == body) {
            pos->b[i][j] = intensityfactor * scale * pos->cosi[i][j];
            sum += pos->b[i][j];
          }
        }
      break;
  case HARMLAMBERT:
      for (i=pos->xlim[0]; i<=pos->xlim[1]; i++)
        for (j=pos->ylim[0]; j<=pos->ylim[1]; j++) {
          if (pos->cose[i][j] > 0.0 && pos->cosi[i][j] > 0.0
                                      && pos->body[i][j] == body && pos->f[i][j] >= 0) {
            c = pos->comp[i][j];
            f = pos->f[i][j];
            scale = mod->photo.optical[ilaw].harmR.local[c][f].R.val/PIE;
            pos->b[i][j] = intensityfactor * scale * pos->cosi[i][j];
            sum += pos->b[i][j];
          }
        }
      break;
  case INHOLAMBERT:
      for (i=pos->xlim[0]; i<=pos->xlim[1]; i++)
        for (j=pos->ylim[0]; j<=pos->ylim[1]; j++) {
          if (pos->cose[i][j] > 0.0 && pos->cosi[i][j] > 0.0
                                      && pos->body[i][j] == body && pos->f[i][j] >= 0) {
            c = pos->comp[i][j];
            f = pos->f[i][j];
            scale = mod->photo.optical[ilaw].inhoR.local[c][f].R.val/PIE;
            pos->b[i][j] = intensityfactor * scale * pos->cosi[i][j];
            sum += pos->b[i][j];
          }
        }
      break;
  case LOMMEL:
      scale = mod->photo.optical[ilaw].R.R.val/(4*PIE);
      for (i=pos->xlim[0]; i<=pos->xlim[1]; i++)
        for (j=pos->ylim[0]; j<=pos->ylim[1]; j++) {
          if (pos->cose[i][j] > 0.0 && pos->cosi[i][j] > 0.0
                                      && pos->body[i][j] == body) {
            pos->b[i][j] = intensityfactor * scale * pos->cosi[i][j]
                                   / (pos->cosi[i][j] + pos->cose[i][j]);
            sum += pos->b[i][j];
          }
        }
      break;
  case HARMLOMMEL:
      for (i=pos->xlim[0]; i<=pos->xlim[1]; i++)
        for (j=pos->ylim[0]; j<=pos->ylim[1]; j++) {
          if (pos->cose[i][j] > 0.0 && pos->cosi[i][j] > 0.0
                                      && pos->body[i][j] == body && pos->f[i][j] >= 0) {
            c = pos->comp[i][j];
            f = pos->f[i][j];
            scale = mod->photo.optical[ilaw].harmR.local[c][f].R.val/(4*PIE);
            pos->b[i][j] = intensityfactor * scale * pos->cosi[i][j]
                                      / (pos->cosi[i][j] + pos->cose[i][j]);
            sum += pos->b[i][j];
          }
        }
      break;
  case INHOLOMMEL:
      for (i=pos->xlim[0]; i<=pos->xlim[1]; i++)
        for (j=pos->ylim[0]; j<=pos->ylim[1]; j++) {
          if (pos->cose[i][j] > 0.0 && pos->cosi[i][j] > 0.0
                                      && pos->body[i][j] == body && pos->f[i][j] >= 0) {
            c = pos->comp[i][j];
            f = pos->f[i][j];
            scale = mod->photo.optical[ilaw].inhoR.local[c][f].R.val/(4*PIE);
            pos->b[i][j] = intensityfactor * scale * pos->cosi[i][j]
                                      / (pos->cosi[i][j] + pos->cose[i][j]);
            sum += pos->b[i][j];
          }
        }
      break;
  case GEOMETRICAL:
      for (i=pos->xlim[0]; i<=pos->xlim[1]; i++)
        for (j=pos->ylim[0]; j<=pos->ylim[1]; j++) {
          if (pos->cose[i][j] > 0.0 && pos->cosi[i][j] > 0.0
                                      && pos->body[i][j] == body) {
            pos->b[i][j] = intensityfactor * mod->photo.optical[ilaw].R.R.val;
            sum += pos->b[i][j];
          }
        }
      break;
  case HAPKE:
      for (i=pos->xlim[0]; i<=pos->xlim[1]; i++)
        for (j=pos->ylim[0]; j<=pos->ylim[1]; j++)
          if (pos->cose[i][j] > 0.0 && pos->cosi[i][j] > 0.0
                                      && pos->body[i][j] == body) {
            pos->b[i][j] = intensityfactor
                             * hapke( pos->cosi[i][j], pos->cose[i][j],
                                      phase, 
                                      mod->photo.optical[ilaw].hapke.w.val,
                                      mod->photo.optical[ilaw].hapke.h.val,
                                      mod->photo.optical[ilaw].hapke.B0.val,
                                      mod->photo.optical[ilaw].hapke.g.val,
                                      mod->photo.optical[ilaw].hapke.theta.val);
            sum += pos->b[i][j]  ;
          }
      break;
  case HARMHAPKE:
      for (i=pos->xlim[0]; i<=pos->xlim[1]; i++)
        for (j=pos->ylim[0]; j<=pos->ylim[1]; j++) {
          if (pos->cose[i][j] > 0.0 && pos->cosi[i][j] > 0.0
                                      && pos->body[i][j] == body && pos->f[i][j] >= 0) {
            c = pos->comp[i][j];
            f = pos->f[i][j];
            pos->b[i][j] = intensityfactor
                             * hapke( pos->cosi[i][j], pos->cose[i][j],
                                      phase, 
                                      mod->photo.optical[ilaw].harmhapke.local[c][f].w.val,
                                      mod->photo.optical[ilaw].harmhapke.local[c][f].h.val,
                                      mod->photo.optical[ilaw].harmhapke.local[c][f].B0.val,
                                      mod->photo.optical[ilaw].harmhapke.local[c][f].g.val,
                                      mod->photo.optical[ilaw].harmhapke.local[c][f].theta.val);
            sum += pos->b[i][j];
          }
        }
      break;
  case INHOHAPKE:
      for (i=pos->xlim[0]; i<=pos->xlim[1]; i++)
        for (j=pos->ylim[0]; j<=pos->ylim[1]; j++) {
          if (pos->cose[i][j] > 0.0 && pos->cosi[i][j] > 0.0
                                      && pos->body[i][j] == body && pos->f[i][j] >= 0) {
            c = pos->comp[i][j];
            f = pos->f[i][j];
            pos->b[i][j] = intensityfactor
                             * hapke( pos->cosi[i][j], pos->cose[i][j],
                                      phase, 
                                      mod->photo.optical[ilaw].inhohapke.local[c][f].w.val,
                                      mod->photo.optical[ilaw].inhohapke.local[c][f].h.val,
                                      mod->photo.optical[ilaw].inhohapke.local[c][f].B0.val,
                                      mod->photo.optical[ilaw].inhohapke.local[c][f].g.val,
                                      mod->photo.optical[ilaw].inhohapke.local[c][f].theta.val);
            sum += pos->b[i][j];
          }
        }
      break;
  case KAASALAINEN:
      phasefunc = mod->photo.optical[ilaw].kaas.A0.val
                  * exp( -phase / mod->photo.optical[ilaw].kaas.D.val)
                  + mod->photo.optical[ilaw].kaas.k.val * phase
                  + 1;
      scale_lommsee = (1 - mod->photo.optical[ilaw].kaas.wt.val)
                      * phasefunc * mod->photo.optical[ilaw].kaas.R.val/(4*PIE);
      scale_lambert = mod->photo.optical[ilaw].kaas.wt.val
                      * phasefunc * mod->photo.optical[ilaw].kaas.R.val/PIE;

      for (i=pos->xlim[0]; i<=pos->xlim[1]; i++)
        for (j=pos->ylim[0]; j<=pos->ylim[1]; j++)
          if (pos->cose[i][j] > 0.0 && pos->cosi[i][j] > 0.0
                                      && pos->body[i][j] == body) {
            pos->b[i][j] = intensityfactor
                             * pos->cosi[i][j]
                             * ( scale_lommsee / (pos->cosi[i][j] + pos->cose[i][j])
                                 + scale_lambert                                          );
//            dbg_occ++;
            sum += pos->b[i][j];
          }

      break;
  case HARMKAAS:
      for (i=pos->xlim[0]; i<=pos->xlim[1]; i++)
        for (j=pos->ylim[0]; j<=pos->ylim[1]; j++) {
          if (pos->cose[i][j] > 0.0 && pos->cosi[i][j] > 0.0
                                      && pos->body[i][j] == body && pos->f[i][j] >= 0) {
            c = pos->comp[i][j];
            f = pos->f[i][j];
            phasefunc = mod->photo.optical[ilaw].harmkaas.local[c][f].A0.val
                        * exp( -phase / mod->photo.optical[ilaw].harmkaas.local[c][f].D.val)
                        + mod->photo.optical[ilaw].harmkaas.local[c][f].k.val * phase
                        + 1;
            scale_lommsee = (1 - mod->photo.optical[ilaw].harmkaas.local[c][f].wt.val)
                            * phasefunc * mod->photo.optical[ilaw].harmkaas.local[c][f].R.val
                            / (4*PIE);
            scale_lambert = mod->photo.optical[ilaw].harmkaas.local[c][f].wt.val
                            * phasefunc * mod->photo.optical[ilaw].harmkaas.local[c][f].R.val
                            / PIE;
            pos->b[i][j] = intensityfactor
                             * pos->cosi[i][j]
                             * ( scale_lommsee / (pos->cosi[i][j] + pos->cose[i][j])
                                 + scale_lambert                                          );
            sum += pos->b[i][j];
          }
        }
      break;
  case INHOKAAS:
      for (i=pos->xlim[0]; i<=pos->xlim[1]; i++)
        for (j=pos->ylim[0]; j<=pos->ylim[1]; j++)
          if (pos->cose[i][j] > 0.0 && pos->cosi[i][j] > 0.0
                                      && pos->body[i][j] == body && pos->f[i][j] >= 0) {
            c = pos->comp[i][j];
            f = pos->f[i][j];
            phasefunc = mod->photo.optical[ilaw].inhokaas.local[c][f].A0.val
                        * exp( -phase / mod->photo.optical[ilaw].inhokaas.local[c][f].D.val)
                        + mod->photo.optical[ilaw].inhokaas.local[c][f].k.val * phase
                        + 1;
            scale_lommsee = (1 - mod->photo.optical[ilaw].inhokaas.local[c][f].wt.val)
                            * phasefunc * mod->photo.optical[ilaw].inhokaas.local[c][f].R.val
                            / (4*PIE);
            scale_lambert = mod->photo.optical[ilaw].inhokaas.local[c][f].wt.val
                            * phasefunc * mod->photo.optical[ilaw].inhokaas.local[c][f].R.val
                            / PIE;
            pos->b[i][j] = intensityfactor
                             * pos->cosi[i][j]
                             * ( scale_lommsee / (pos->cosi[i][j] + pos->cose[i][j])
                                 + scale_lambert                                          );
            sum += pos->b[i][j];
          }
      break;
  case NOLAW:
      bailout("apply_photo.c: can't set optical scattering law = \"none\" when optical data are used\n");
      break;
  default:
      bailout("apply_photo.c: can't handle that optical scattering law yet\n");
  }

  if (sum == 0.0)
    sum = TINY;

  return sum;
}

#undef TINY
