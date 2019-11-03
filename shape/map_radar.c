/*****************************************************************************************
                                                                              map_radar.c

This routine implements the "map" action, taking a realized model and producing graphical
output that describes the mapping from (a) surface elements projected onto the plane of
the sky to (b) delay-Doppler space (or vice versa).

The various subroutines called by map_radar are stripped-down, modified versions of
routines found in calc_fits.c, chi2.c, and write_pos.c.

Modified 2015 June 10 by CM:
    Implement smearing for the "fit" and "write" actions

Modified 2014 February 11 by CM:
    Replace the "is_optical" argument to the plot_surface routine with the "iradlaw"
        argument
    Initialize variables to avoid compilation warnings

Modified 2013 June 25 by CM:
    In the write_map_pos routine fix a bug with setting brightness levels for blue-shaded
        regions when the "map_poscutoff" parameter is used
    Get colors of plot annotations back into synch with the colors assigned by the
        write_pos routine
    When map_mode = "deldop" or "pos" check that "map_frame" has a legal value:
        0 <= map_frame < # frames in the dataset specified by "map_set"
    Add "is_optical" argument to the plot_surface routine, and use that routine to
        display the names of POS images (sky renderings) rather than doing it here
    Add "posmax" argument to the annotate_plot routine and remove "is_optical" argument

Modified 2013 April 24 by CM:
    Adjust names of output images so they are in alphanumeric order if > 100 per dataset

Modified 2012 March 5 by CM:
    Rename MAXMAPFACETS defined constant as MAXCHOSENFACETS

Modified 2010 September 1 by CM:
    Eliminate unused variables, initialize variables to avoid compilation
        warnings

Modified 2010 August 26 by CM:
    Change the "map_forward" parameter to "map_mode" and implement
        map_mode = 'facets'
    Implement the "map_verbose" parameter
    For delay-Doppler datasets, output pgm images for the data (obs_*.pgm)
        and residuals (res_*.pgm)
    For Doppler datasets, change the fit_*.map.dat output file to look
        like the fit_*.dat file from the "write" action but with one
        additional column giving the mapping

Written 2010 June 15 by CM
*****************************************************************************************/

#include "head.h"

void map_deldop( struct par_t *par, struct mod_t *mod, struct deldop_t *deldop,
                 int s);
void map_doppler( struct par_t *par, struct mod_t *mod, struct doppler_t *doppler,
                  int s);
void write_map_pos( struct par_t *par, struct mod_t *mod, struct pos_t *pos,
                    double **map_pos, double spin_ecl[3], int iradlaw, int s, int f,
                    int nframes);
void write_map_fit_deldop( struct par_t *par, struct deldopfrm_t *frame,
                           int s, int f, int nframes);
void write_map_fit_doppler( struct par_t *par, struct dopfrm_t *frame,
                            int s, int f, int nframes);


void map_radar( struct par_t *par, struct mod_t *mod, struct dat_t *dat)
{
  int s1, s2, s;

  /*  Initialize the flags that indicate that the model extends beyond the
      POS frame and that the model is too wide in (delay-)Doppler space to
      create (delay-)Doppler fit frames                                     */

  par->posbnd = 0;
  par->badradar = 0;

  /*  Initialize the flag that indicates that the specified POS region
      (for the map action in the forward direction) maps onto a
      (delay-)Doppler region that overflows beyond the fit frame        */

  par->map_overflow = 0;

  /*  Shift user-specified delay-Doppler or POS limits, if any, from 0-based
      counting of delay-Doppler or POS rows and columns to the appropriate method  */

  if (par->map_mode == MAPMODE_DELDOP) {

      /* delay-Doppler fit image or Doppler fit spectrum: 1-based counting */

      par->map_dellim[0] += 1;
      par->map_dellim[1] += 1;
      par->map_doplim[0] += 1;
      par->map_doplim[1] += 1;
  } else if (par->map_mode == MAPMODE_POS) {

      /* POS image: origin at center */

      par->map_xposlim[0] -= dat->pos.n;
      par->map_xposlim[1] -= dat->pos.n;
      par->map_yposlim[0] -= dat->pos.n;
      par->map_yposlim[1] -= dat->pos.n;
  }

  /*  Depending on the "map_mode" parameter, we may loop through
      all datasets or else look at just one                       */

  if (par->map_mode == MAPMODE_FACETS) {
      s1 = 0;
      s2 = dat->nsets - 1;
  } else {
      s1 = par->map_set;
      s2 = par->map_set;
  }

  /*  Look at the dataset(s) and call the appropriate routine(s)
      depending on the dataset type                               */

  printf("#\n");

  for (s=s1; s<=s2; s++)
    if (dat->set[s].type == DELAY)
      map_deldop( par, mod, &dat->set[s].desc.deldop, s);
    else if (dat->set[s].type == DOPPLER)
      map_doppler( par, mod, &dat->set[s].desc.doppler, s);
    else if (par->map_mode != MAPMODE_FACETS)
      bailout("map_radar.c: map_set doesn't specify a radar dataset\n");
}


void map_deldop( struct par_t *par, struct mod_t *mod, struct deldop_t *deldop,
                 int s)
{
  double orbit_offset[3] = {0.0, 0.0, 0.0};
  int f1, f2, f, i, j, c, ndel, ndop, n, k, matched_facet, x, y, cmax, fmax,
      ncontributions, fac, v;
  int *clist, *flist;
  double maxval, sum_signal, total_contribution, fcoords[3];
  double *contribution_list;

  /*  Initialize variables to avoid compilation warnings  */

  f1 = f2 = cmax = fmax = 0;

  /*  Depending on the "map_mode" parameter, we may loop through
      all frames in this dataset or else look at just one         */

  if (par->map_mode == MAPMODE_FACETS) {
      f1 = 0;
      f2 = deldop->nframes - 1;
  } else if (par->map_frame < 0 || par->map_frame >= deldop->nframes) {
      fprintf( stderr, "ERROR: for map_set = %2d must have 0 <= map_frame <= %2d\n",
               par->map_set, deldop->nframes - 1);
      bailout("map_deldop in map_radar.c\n");
  } else {
      f1 = par->map_frame;
      f2 = par->map_frame;
  }

  /*  Look at each frame in turn  */

  for (f=f1; f<=f2; f++) {

    /*  Initialize some values  */

    ndel = deldop->frame[f].ndel;
    ndop = deldop->frame[f].ndop;
    n = deldop->frame[f].pos.n;

    for (i=0; i<=2; i++)
      for (j=0; j<=2; j++) {
        deldop->frame[f].pos.ae[i][j] = deldop->frame[f].view[deldop->v0].ae[i][j];
        deldop->frame[f].pos.oe[i][j] = deldop->frame[f].view[deldop->v0].oe[i][j];
      }
    deldop->frame[f].pos.bistatic = 0;

    /*  Initialize the plane-of-sky view  */

    posclr( &deldop->frame[f].pos);

    /*  Call routine posvis to get the facet number, scattering angle,
        and distance toward Earth at the center of each POS pixel;
        set the posbnd parameter to 1 if any portion of the model
        extends beyond the POS frame limits.                            */

    for (c=0; c<mod->shape.ncomp; c++)
      if (posvis( &mod->shape.comp[c].real, orbit_offset, &deldop->frame[f].pos,
                  (int) par->pos_smooth, 0, 0, c))
        par->posbnd = 1;

    /*  Allocate and initialize memory needed by the pos2deldop routine for the map action  */

    deldop->frame[f].map_fit = matrix( 1, ndel, 1, ndop);
    deldop->frame[f].map_pos = matrix( -n, n, -n, n);

    if (par->map_mode == MAPMODE_DELDOP) {
        for (i=1; i<=ndel; i++)
          for (j=1; j<=ndop; j++)
            if (i >= par->map_dellim[0] && i <= par->map_dellim[1] &&
                j >= par->map_doplim[0] && j <= par->map_doplim[1]    )
              deldop->frame[f].map_fit[i][j] = 1.0;
            else
              deldop->frame[f].map_fit[i][j] = 0.0;
    } else {
        for (i=1; i<=ndel; i++)
          for (j=1; j<=ndop; j++)
            deldop->frame[f].map_fit[i][j] = 0.0;
    }

    if (par->map_mode == MAPMODE_DELDOP) {
        for (x=-n; x<=n; x++)
          for (y=-n; y<=n; y++)
            deldop->frame[f].map_pos[x][y] = 0.0;
    } else if (par->map_mode == MAPMODE_POS) {
        for (x=-n; x<=n; x++)
          for (y=-n; y<=n; y++)
            if (x >= par->map_xposlim[0] && x <= par->map_xposlim[1] &&
                y >= par->map_yposlim[0] && y <= par->map_yposlim[1]    )
              deldop->frame[f].map_pos[x][y] = 1.0;
            else
              deldop->frame[f].map_pos[x][y] = 0.0;
    } else {
        for (x=-n; x<=n; x++)
          for (y=-n; y<=n; y++) {
            matched_facet = 0;
            k = 0;
            while (!matched_facet && k < MAXCHOSENFACETS && par->map_facet[k] >= 0) {
              if (par->map_comp[k] == deldop->frame[f].pos.comp[x][y]
                          && par->map_facet[k] == deldop->frame[f].pos.f[x][y])
                matched_facet = 1;
              else
                k++;
            }
            if (matched_facet)
              deldop->frame[f].map_pos[x][y] = 1.0;
            else
              deldop->frame[f].map_pos[x][y] = 0.0;
          }
    }

    if (par->map_mode != MAPMODE_FACETS) {
      cmax = fmax = -1;
      for (x=deldop->frame[f].pos.xlim[0]; x<=deldop->frame[f].pos.xlim[1]; x++)
        for (y=deldop->frame[f].pos.ylim[0]; y<=deldop->frame[f].pos.ylim[1]; y++) {
          cmax = MAX( cmax, deldop->frame[f].pos.comp[x][y]);
          fmax = MAX( fmax, deldop->frame[f].pos.f[x][y]);
        }
      deldop->frame[f].map_facet_power = matrix( 0, cmax, 0, fmax);
      for (c=0; c<=cmax; c++)
        for (fac=0; fac<=fmax; fac++)
          deldop->frame[f].map_facet_power[c][fac] = 0.0;
    }

    if (par->map_verbose) {
      printf("# set %2d frame %2d:\n", s, f);
      printf("#\n");
    }

    /*  Zero out the fit delay-Doppler image, then call pos2deldop
        to create the fit image by mapping power from the plane of
        the sky to delay-Doppler space.                             */

    clrmat( deldop->frame[f].fit, 1, deldop->frame[f].ndel,
                                    1, deldop->frame[f].ndop);
    if (pos2deldop( par, &mod->photo, 0.0, 0.0, 0.0, deldop, 0, s, f, 0))
      par->badradar = 1;

    /*  For the inverse mapping with map_mode = 'deldop', compute the fractional
        contribution of each POS pixel to the summed signal in the specified delay-Doppler
        fit region, then rescale to a maximum of 1.0.

        For the forward mapping with map_mode = 'pos' or 'facets', compute the fractional
        contribution of the specified POS region or facets to the signal in each
        delay-Doppler fit pixel, then rescale to a maximum of 1.0.                          */

    if (par->map_mode == MAPMODE_DELDOP) {
        sum_signal = 0.0;
        for (i=par->map_dellim[0]; i<=par->map_dellim[1]; i++)
          for (j=par->map_doplim[0]; j<=par->map_doplim[1]; j++)
            if (deldop->frame[f].fit[i][j] > 0.0)
              sum_signal += deldop->frame[f].fit[i][j];
        if (sum_signal == 0.0)
          bailout("map_radar.c: no model power in selected region\n");
        maxval = 0.0;
        for (x=deldop->frame[f].pos.xlim[0]; x<=deldop->frame[f].pos.xlim[1]; x++)
          for (y=deldop->frame[f].pos.ylim[0]; y<=deldop->frame[f].pos.ylim[1]; y++)
            if (deldop->frame[f].map_pos[x][y] > 0.0) {
              deldop->frame[f].map_pos[x][y] /= sum_signal;
              maxval = MAX( maxval, deldop->frame[f].map_pos[x][y]);
            }
        if (par->map_verbose) {
          printf("#\n");
          printf("# set %2d frame %2d rescaled mapping:\n", s, f);
          printf("#\n");
        }
        for (x=deldop->frame[f].pos.xlim[0]; x<=deldop->frame[f].pos.xlim[1]; x++)
          for (y=deldop->frame[f].pos.ylim[0]; y<=deldop->frame[f].pos.ylim[1]; y++)
            if (deldop->frame[f].map_pos[x][y] > 0.0) {
              deldop->frame[f].map_pos[i][j] /= maxval;
              if (par->map_verbose)
                printf("# POS (%3d, %3d)  %e\n",
                       x+n, y+n, deldop->frame[f].map_pos[x][y]);
            }
    } else {
        maxval = 0.0;
        for (i=1; i<=ndel; i++)
          for (j=1; j<=ndop; j++)
            if (deldop->frame[f].map_fit[i][j] > 0.0) {
              deldop->frame[f].map_fit[i][j] /= deldop->frame[f].fit[i][j];
              maxval = MAX( maxval, deldop->frame[f].map_fit[i][j]);
            }
        if (maxval == 0.0) {
            if (par->map_mode == MAPMODE_POS)
              bailout("map_radar.c: no model power in selected region\n");
            else
              printf("# no contribution from selected facets\n");
        } else if (par->map_verbose) {
            printf("#\n");
            printf("# set %2d frame %2d rescaled mapping:\n", s, f);
            printf("#\n");
        }
        for (i=1; i<=ndel; i++)
          for (j=1; j<=ndop; j++)
            if (deldop->frame[f].map_fit[i][j] > 0.0) {
              deldop->frame[f].map_fit[i][j] /= maxval;
              if (par->map_verbose)
                printf("# d-D (%3d, %3d)  %e\n",
                       i-1, j-1, deldop->frame[f].map_fit[i][j]);
            }
    }

    /*  If map_mode = 'deldop' or 'pos', list the fractional contribution
        of each facet, ordered from high to low                            */

    ncontributions = 0;
    total_contribution = 0.0;
    v = 0;
    for (j=0; j<=2; j++)
      fcoords[j] = 0.0;
    if (par->map_mode != MAPMODE_FACETS) {
      for (c=0; c<=cmax; c++)
        for (fac=0; fac<=fmax; fac++)
          if (deldop->frame[f].map_facet_power[c][fac] > 0.0) {
            total_contribution += deldop->frame[f].map_facet_power[c][fac];
            ncontributions++;
          }
      if (ncontributions == 0)
        bailout("map_radar.c: no model power in selected region\n");
      clist = ivector( 1, ncontributions);
      flist = ivector( 1, ncontributions);
      contribution_list = vector( 1, ncontributions);
      k = 0;
      for (c=0; c<=cmax; c++)
        for (fac=0; fac<=fmax; fac++)
          if (deldop->frame[f].map_facet_power[c][fac] > 0.0) {
            k++;
            clist[k] = c;
            flist[k] = fac;
            contribution_list[k] = deldop->frame[f].map_facet_power[c][fac]
                                   / total_contribution;
          }
      sort3dii( ncontributions, contribution_list, clist, flist);
      if (par->map_verbose)
        printf("#\n");
      for (k=ncontributions; k>=1; k--) {
        c = clist[k];
        fac = flist[k];
        for (i=0; i<=2; i++) {
          v = mod->shape.comp[c].real.f[fac].v[i];
          for (j=0; j<=2; j++)
            fcoords[j] += mod->shape.comp[c].real.v[v].x[j];
        }
        for (j=0; j<=2; j++)
          fcoords[j] /= 3;
        cotrans( fcoords, deldop->frame[f].view[deldop->v0].ae, fcoords, -1);
        cotrans( fcoords, deldop->frame[f].view[deldop->v0].oe, fcoords, 1);
        x = iround( fcoords[0]/deldop->frame[f].pos.km_per_pixel);
        y = iround( fcoords[1]/deldop->frame[f].pos.km_per_pixel);
        printf("# comp %d facet %4d centered at POS (%3d, %3d): fractional contribution %e\n",
               c, fac, x+n, y+n, contribution_list[k]);
      }
      if (!par->map_verbose)
        printf("#\n");
      free_matrix( deldop->frame[f].map_facet_power, 0, cmax, 0, fmax);
      free_ivector( clist, 1, ncontributions);
      free_ivector( flist, 1, ncontributions);
      free_vector( contribution_list, 1, ncontributions);
    }

    /*  Write the POS frame to disk  */

    write_map_pos( par, mod, &deldop->frame[f].pos, deldop->frame[f].map_pos,
                   deldop->frame[f].view[deldop->v0].intspin, deldop->iradlaw,
                   s, f, deldop->nframes);

    /*  Write the fit frame to disk  */

    write_map_fit_deldop( par, &deldop->frame[f], s, f, deldop->nframes);

    /*  Clean up  */

    free_matrix( deldop->frame[f].map_fit, 1, ndel, 1, ndop);
    free_matrix( deldop->frame[f].map_pos, -n, n, -n, n);

    /*  Loop back to the next frame  */
  }
}


void map_doppler( struct par_t *par, struct mod_t *mod, struct doppler_t *doppler,
                  int s)
{
  double orbit_offset[3] = {0.0, 0.0, 0.0};
  int f1, f2, f, i, j, c, ndop, n, k, matched_facet, x, y, cmax, fmax, fac,
      ncontributions, v;
  int *clist, *flist;
  double maxval, sum_signal, total_contribution, fcoords[3];
  double *contribution_list;

  /*  Initialize variables to avoid compilation warnings  */

  f1 = f2 = cmax = fmax = 0;

  /*  Depending on the "map_mode" parameter, we may loop through
      all frames in this dataset or else look at just one         */

  if (par->map_mode == MAPMODE_FACETS) {
      f1 = 0;
      f2 = doppler->nframes - 1;
  } else if (par->map_frame < 0 || par->map_frame >= doppler->nframes) {
      fprintf( stderr, "ERROR: for map_set = %2d must have 0 <= map_frame <= %2d\n",
               par->map_set, doppler->nframes - 1);
      bailout("map_doppler in map_radar.c\n");
  } else {
      f1 = par->map_frame;
      f2 = par->map_frame;
  }

  /*  Look at each frame in turn  */

  for (f=f1; f<=f2; f++) {

    /*  Initialize some values  */

    ndop = doppler->frame[f].ndop;
    n = doppler->frame[f].pos.n;

    for (i=0; i<=2; i++)
      for (j=0; j<=2; j++) {
        doppler->frame[f].pos.ae[i][j] = doppler->frame[f].view[doppler->v0].ae[i][j];
        doppler->frame[f].pos.oe[i][j] = doppler->frame[f].view[doppler->v0].oe[i][j];
      }
    doppler->frame[f].pos.bistatic = 0;

    /*  Initialize the plane-of-sky view  */

    posclr( &doppler->frame[f].pos);

    /*  Call routine posvis to get the facet number, scattering angle,
        and distance toward Earth at the center of each POS pixel;
        set the posbnd parameter to 1 if any portion of the model
        extends beyond the POS frame limits.                            */

    for (c=0; c<mod->shape.ncomp; c++)
      if (posvis( &mod->shape.comp[c].real, orbit_offset, &doppler->frame[f].pos,
                  (int) par->pos_smooth, 0, 0, c))
        par->posbnd = 1;

    /*  Allocate and initialize memory needed by the pos2doppler routine for the map action  */

    doppler->frame[f].map_fit = vector( 1, ndop);
    doppler->frame[f].map_pos = matrix( -n, n, -n, n);

    if (par->map_mode == MAPMODE_DELDOP) {
        for (j=1; j<=ndop; j++)
          if (j >= par->map_doplim[0] && j <= par->map_doplim[1])
            doppler->frame[f].map_fit[j] = 1.0;
          else
            doppler->frame[f].map_fit[j] = 0.0;
    } else {
        for (j=1; j<=ndop; j++)
          doppler->frame[f].map_fit[j] = 0.0;
    }

    if (par->map_mode == MAPMODE_DELDOP) {
        for (x=-n; x<=n; x++)
          for (y=-n; y<=n; y++)
            doppler->frame[f].map_pos[x][y] = 0.0;
    } else if (par->map_mode == MAPMODE_POS) {
        for (x=-n; x<=n; x++)
          for (y=-n; y<=n; y++)
            if (x >= par->map_xposlim[0] && x <= par->map_xposlim[1] &&
                y >= par->map_yposlim[0] && y <= par->map_yposlim[1]    )
              doppler->frame[f].map_pos[x][y] = 1.0;
            else
              doppler->frame[f].map_pos[x][y] = 0.0;
    } else {
        for (x=-n; x<=n; x++)
          for (y=-n; y<=n; y++) {
            matched_facet = 0;
            k = 0;
            while (!matched_facet && k < MAXCHOSENFACETS && par->map_facet[k] >= 0) {
              if (par->map_comp[k] == doppler->frame[f].pos.comp[x][y]
                          && par->map_facet[k] == doppler->frame[f].pos.f[x][y])
                matched_facet = 1;
              else
                k++;
            }
            if (matched_facet)
              doppler->frame[f].map_pos[x][y] = 1.0;
            else
              doppler->frame[f].map_pos[x][y] = 0.0;
          }
    }

    if (par->map_mode != MAPMODE_FACETS) {
      cmax = fmax = -1;
      for (x=doppler->frame[f].pos.xlim[0]; x<=doppler->frame[f].pos.xlim[1]; x++)
        for (y=doppler->frame[f].pos.ylim[0]; y<=doppler->frame[f].pos.ylim[1]; y++) {
          cmax = MAX( cmax, doppler->frame[f].pos.comp[x][y]);
          fmax = MAX( fmax, doppler->frame[f].pos.f[x][y]);
        }
      doppler->frame[f].map_facet_power = matrix( 0, cmax, 0, fmax);
      for (c=0; c<=cmax; c++)
        for (fac=0; fac<=fmax; fac++)
          doppler->frame[f].map_facet_power[c][fac] = 0.0;
      printf("#\n");
    }

    if (par->map_verbose) {
      printf("# set %2d frame %2d:\n", s, f);
      printf("#\n");
    }

    /*  Zero out the fit Doppler spectrum, then call pos2doppler
        to create the fit spectrum by mapping power from the
        plane of sky to Doppler space.                            */

    clrvect( doppler->frame[f].fit, 1, doppler->frame[f].ndop);
    if (pos2doppler( par, &mod->photo, 0.0, 0.0, 0.0, doppler, 0, s, f, 0))
      par->badradar = 1;

    /*  For the inverse mapping with map_mode = 'deldop', compute the fractional
        contribution of each POS pixel to the summed signal in the specified delay-Doppler
        fit region, then rescale to a maximum of 1.0.

        For the forward mapping with map_mode = 'pos' or 'facets', compute the fractional
        contribution of the specified POS region or facets to the signal in each
        Doppler fit bin, then rescale to a maximum of 1.0.                          */

    if (par->map_mode == MAPMODE_DELDOP) {
        sum_signal = 0.0;
        for (j=par->map_doplim[0]; j<=par->map_doplim[1]; j++)
          if (doppler->frame[f].fit[j] > 0.0)
            sum_signal += doppler->frame[f].fit[j];
        if (sum_signal == 0.0)
          bailout("map_radar.c: no model power in selected region\n");
        maxval = 0.0;
        for (x=doppler->frame[f].pos.xlim[0]; x<=doppler->frame[f].pos.xlim[1]; x++)
          for (y=doppler->frame[f].pos.ylim[0]; y<=doppler->frame[f].pos.ylim[1]; y++)
            if (doppler->frame[f].map_pos[x][y] > 0.0) {
              doppler->frame[f].map_pos[x][y] /= sum_signal;
              maxval = MAX( maxval, doppler->frame[f].map_pos[x][y]);
            }
        if (par->map_verbose) {
          printf("#\n");
          printf("# set %2d frame %2d rescaled mapping:\n", s, f);
          printf("#\n");
        }
        for (x=doppler->frame[f].pos.xlim[0]; x<=doppler->frame[f].pos.xlim[1]; x++)
          for (y=doppler->frame[f].pos.ylim[0]; y<=doppler->frame[f].pos.ylim[1]; y++)
            if (doppler->frame[f].map_pos[x][y] > 0.0) {
              doppler->frame[f].map_pos[i][j] /= maxval;
              if (par->map_verbose)
                printf("# POS (%3d, %3d)  %e\n",
                       x+n, y+n, doppler->frame[f].map_pos[x][y]);
            }
    } else {
        maxval = 0.0;
        for (j=1; j<=ndop; j++)
          if (doppler->frame[f].map_fit[j] > 0.0) {
            doppler->frame[f].map_fit[j] /= doppler->frame[f].fit[j];
            maxval = MAX( maxval, doppler->frame[f].map_fit[j]);
          }
        if (maxval == 0.0) {
            if (par->map_mode == MAPMODE_POS)
              bailout("map_radar.c: no model power in selected region\n");
            else
              printf("# no contribution from selected facets\n");
        } else if (par->map_verbose) {
            printf("#\n");
            printf("# set %2d frame %2d rescaled mapping:\n", s, f);
            printf("#\n");
        }
        for (j=1; j<=ndop; j++)
          if (doppler->frame[f].map_fit[j] > 0.0) {
            doppler->frame[f].map_fit[j] /= maxval;
            if (par->map_verbose)
              printf("# Dop (%3d)       %e\n",
                     j-1, doppler->frame[f].map_fit[j]);
          }
    }

    /*  If map_mode = 'deldop' or 'pos', list the fractional contribution
        of each facet, ordered from high to low                            */

    ncontributions = 0;
    total_contribution = 0.0;
    v = 0;
    for (j=0; j<=2; j++)
      fcoords[j] = 0.0;
    if (par->map_mode != MAPMODE_FACETS) {
      for (c=0; c<=cmax; c++)
        for (fac=0; fac<=fmax; fac++)
          if (doppler->frame[f].map_facet_power[c][fac] > 0.0) {
            total_contribution += doppler->frame[f].map_facet_power[c][fac];
            ncontributions++;
          }
      if (ncontributions == 0)
        bailout("map_radar.c: no model power in selected region\n");
      clist = ivector( 1, ncontributions);
      flist = ivector( 1, ncontributions);
      contribution_list = vector( 1, ncontributions);
      k = 0;
      for (c=0; c<=cmax; c++)
        for (fac=0; fac<=fmax; fac++)
          if (doppler->frame[f].map_facet_power[c][fac] > 0.0) {
            k++;
            clist[k] = c;
            flist[k] = fac;
            contribution_list[k] = doppler->frame[f].map_facet_power[c][fac]
                                   / total_contribution;
          }
      sort3dii( ncontributions, contribution_list, clist, flist);
      if (par->map_verbose)
        printf("#\n");
      for (k=ncontributions; k>=1; k--) {
        c = clist[k];
        fac = flist[k];
        for (i=0; i<=2; i++) {
          v = mod->shape.comp[c].real.f[fac].v[i];
          for (j=0; j<=2; j++)
            fcoords[j] += mod->shape.comp[c].real.v[v].x[j];
        }
        for (j=0; j<=2; j++)
          fcoords[j] /= 3;
        cotrans( fcoords, doppler->frame[f].view[doppler->v0].ae, fcoords, -1);
        cotrans( fcoords, doppler->frame[f].view[doppler->v0].oe, fcoords, 1);
        x = iround( fcoords[0]/doppler->frame[f].pos.km_per_pixel);
        y = iround( fcoords[1]/doppler->frame[f].pos.km_per_pixel);
        printf("# comp %d facet %4d centered at POS (%3d, %3d): fractional contribution %e\n",
               c, fac, x+n, y+n, contribution_list[k]);
      }
      if (!par->map_verbose)
        printf("#\n");
      free_matrix( doppler->frame[f].map_facet_power, 0, cmax, 0, fmax);
      free_ivector( clist, 1, ncontributions);
      free_ivector( flist, 1, ncontributions);
      free_vector( contribution_list, 1, ncontributions);
    }

    /*  Write the POS frame to disk  */

    write_map_pos( par, mod, &doppler->frame[f].pos, doppler->frame[f].map_pos,
                   doppler->frame[f].view[doppler->v0].intspin, doppler->iradlaw,
                   s, f, doppler->nframes);

    /*  Write the fit frame to disk  */

    write_map_fit_doppler( par, &doppler->frame[f], s, f, doppler->nframes);

    /*  Clean up  */

    free_vector( doppler->frame[f].map_fit, 1, ndop);
    free_matrix( doppler->frame[f].map_pos, -n, n, -n, n);

    /*  Loop back to the next frame  */
  }
}


void write_map_pos( struct par_t *par, struct mod_t *mod, struct pos_t *pos,
                    double **map_pos, double spin_ecl[3], int iradlaw, int s, int f,
                    int nframes)
{
  /* max RGB levels for various plotting elements */
  const int clevels[10][3] = {{255, 255, 255},     /* asteroid surface  */
                              {255, 255,   0},     /* unseen ast. surf. */
                              {  0,   0, 255},     /* highlighted facet */
                              {255,   0, 255},     /* spin vector       */
                              {  0,   0,   0},     /* COM               */
                              {127,  63,   0},     /* subradar point    */
                              {255,   0,   0},     /* long PA           */
                              {  0, 255,   0},     /* intermediate PA   */
                              {  0, 127, 255},     /* short PA          */
                              {127, 255, 255}};    /* angular momentum  */

  /* max RGB levels for maximally mapped asteroid surface */
  const int maplevels[3] = {  0,   0, 255};

  char name[MAXLEN];
  int n, i, j, icolor, use_cutoff;
  int **color;
  double maxbrightness, posmax, scaled_brightness, level;
  double **brightness, **z, ***cbrightness;

  n = pos->n;

  /*  Assign the name of the color (ppm) image file that will be written to disk  */

  if (nframes > 100)
    sprintf( name, "sky_%02d_%03d.map.ppm", s, f);
  else
    sprintf( name, "sky_%02d_%02d.map.ppm", s, f);

  /*  If necessary, give a warning that the model is larger than the POS image  */

  if (par->map_verbose)
    printf("#\n");
  if (par->posbnd)
    printf("WARNING: model extends beyond the POS frame\n");

  /*
      Prepare storage matrices to list three quantities for each
         line of sight (each POS pixel):

         1) color type (0 = asteroid surface, 1 = unseen ast. surf., 2 = spin vector, etc.)
         2) brightness (grayscale)
         3) z-coordinate (distance toward observer) of the closest
               plotting element (asteroid surface, arrow, etc.)
         4) brightness (RGB)
  */

  color = imatrix( -n, n, -n, n);
  brightness = matrix( -n, n, -n, n);
  z = matrix( -n, n, -n, n);
  cbrightness = d3tensor( -n, n, -n, n, 0, 2);

  /*  Start the plot by filling in the asteroid surface
      (this will also print the image filename to the screen)  */

  plot_surface( par, mod, par->sky_radlaw, iradlaw, name,
                &maxbrightness, &posmax, pos, color, brightness, z);

  /*  Add each of the plot annotations that has been specified:
      spin vector, center of mass, subradar point, angular momentum vector  */

  annotate_plot( par, mod, spin_ecl, maxbrightness, posmax,
                 pos, color, brightness, z);

  /*  Set RGB levels.  For the asteroid's surface, these levels depend on
      whether or not a cutoff is being applied.  If no cutoff is being
      applied, each pixel's relative brightness was determined in the call
      to plot_surface, while the color is determined by the values in the
      map_pos matrix (white = unmapped, saturated blue = maximally mapped).

      However, if a cutoff is being applied, pixels are left uncolored
      with brightness as determined by plot_surface (below the cutoff) or
      else maximally colored with maximum brightness (above the cutoff).
      For this case, scale the brightness so as to undo the effect of the
      "radposmax" parameter on the colored region.                           */

  use_cutoff = (par->map_poscutoff >= 0.0 && par->map_poscutoff <= 1.0) ? 1 : 0;

  scaled_brightness = (posmax != 0.0) ? posmax: maxbrightness;

  for (i=(-n); i<=n; i++)
    for (j=(-n); j<=n; j++)
      if (color[i][j] == 0) {

          /* asteroid surface (or blank sky) */

          if (use_cutoff) {
              if (map_pos[i][j] >= par->map_poscutoff)
                for (icolor=0; icolor<=2; icolor++)
                  cbrightness[i][j][icolor] = scaled_brightness*maplevels[icolor]/255.0;
              else
                for (icolor=0; icolor<=2; icolor++)
                  cbrightness[i][j][icolor] = brightness[i][j]*clevels[0][icolor]/255.0;
          } else {
              for (icolor=0; icolor<=2; icolor++) {
                level = clevels[0][icolor]
                        + (maplevels[icolor] - clevels[0][icolor]) * map_pos[i][j];
                cbrightness[i][j][icolor] = brightness[i][j]*level/255.0;
              }
          }
      } else {

          /* plot annotation */

          for (icolor=0; icolor<=2; icolor++) {
            level = clevels[color[i][j]][icolor];
            cbrightness[i][j][icolor] = brightness[i][j]*level/255.0;
          }
      }

  /*  Write the image to disk  */

  if (posmax == 0.0)
    wimasppm0( cbrightness, -n, n, -n, n, 0, 0, 0, name);
  else
    wimasppmsc( cbrightness, -n, n, -n, n, 0.0, posmax, 0, 0, 0, name);
  fflush(stdout);

  /*  Clean up  */

  free_imatrix( color, -n, n, -n, n);
  free_matrix( brightness, -n, n, -n, n);
  free_matrix( z, -n, n, -n, n);
  free_d3tensor( cbrightness, -n, n, -n, n, 0, 2);
}


void write_map_fit_deldop( struct par_t *par, struct deldopfrm_t *frame,
                           int s, int f, int nframes)
{
  /* RGB levels for unmapped delay-Doppler region */
  const int clevels[3] = {255, 255, 255};

  /* max RGB levels for maximally mapped delay-Doppler region */
  const int maplevels[3] = {  0,   0, 255};

  char fitname[MAXLEN], obsname[MAXLEN], resname[MAXLEN];
  int ndel, ndop, i, j, icolor, use_cutoff;
  double o2, m2, om, calval, fit255, obs255, fit255_use, obs255_use, level;
  double **obs, **fit, **oneovervar, ***cfit, **res;

  /*  Assign variables for this frame  */

  ndel = frame->ndel;
  ndop = frame->ndop;
  obs = frame->obs;
  fit = frame->fit;
  oneovervar = frame->oneovervar;  /* 1/variance */

  /*  In order to get this frame's calibration factor we must go through most of
      the chi-square calculation.  Initialize contributions to chi-square to
      values that account for overflow (if any) beyond the limits of the data
      frame.  These contributions were computed by routine pos2deldop.            */

  o2 = frame->overflow_o2;
  m2 = frame->overflow_m2;
  om = 0.0;

  /*  Now add the contributions from power within the limits of the data frame.  */

  for (i=1; i<=ndel; i++)
    for (j=1; j<=ndop; j++) {
      o2 += obs[i][j]*obs[i][j]*oneovervar[i][j];
      m2 += fit[i][j]*fit[i][j]*oneovervar[i][j];
      om += fit[i][j]*obs[i][j]*oneovervar[i][j];
    }

  /*  If this frame's calibration factor is allowed to float,
      set it to minimize chi-square, the sum over all pixels of
      { (obs - calfact*fit)^2 / variance }.                       */

  if (frame->cal.state == 'f') {
    if (om > 0.0) {
        frame->cal.val = om/m2;
    } else {
        frame->cal.val = TINYCALFACT;
        printf("WARNING: set %2d frame %2d had negative calfact reset to %10.4e\n",
               s, f, frame->cal.val);
    }
  }
  calval = frame->cal.val;

  /*  Zero out obs and fit pixels that have been zeroed out in a pixel mask  */

  if (frame->pixels_weighted)
    for (i=1; i<=ndel; i++)
      for (j=1; j<=ndop; j++)
        if (oneovervar[i][j] == 0.0) {
          obs[i][j] = 0.0;
          fit[i][j] = 0.0;
        }

  /*  Compute the maximum pixel value for the data and (cal*fit) frames,
      so that those two images can be on the same scale if desired.
      (fit255 and obs255 are the values that map to 255 = bright white.)  */

  if (nframes > 100) {
      sprintf( fitname, "fit_%02d_%03d.map.ppm", s, f);
      sprintf( obsname, "obs_%02d_%03d.pgm", s, f);
  } else {
      sprintf( fitname, "fit_%02d_%02d.map.ppm", s, f);
      sprintf( obsname, "obs_%02d_%02d.pgm", s, f);
  }
  fit255 = -HUGENUMBER;
  obs255 = -HUGENUMBER;
  for (i=1; i<=ndel; i++)
    for (j=1; j<=ndop; j++) {
      fit255 = MAX( fit255, fit[i][j]);
      obs255 = MAX( obs255, obs[i][j]);
    }
  fit255_use = (par->radfitmax == 0.0) ? fit255 : par->radfitmax / calval;
  obs255_use = (par->radobsmax == 0.0) ? obs255 : par->radobsmax;
  if (par->scalefitobs == SCALE_MAXFITOBS) {
      if (obs255_use > calval*fit255_use)
        fit255_use = obs255_use/calval;
      else
        obs255_use = calval*fit255_use;
  } else if (par->scalefitobs == SCALE_FIT) {
      obs255_use = calval*fit255_use;
  } else if (par->scalefitobs == SCALE_OBS) {
      fit255_use = obs255_use/calval;
  }
  if (par->radfitmax != 0.0)
    printf("# %s  (max pixel value = %10.4e, bright white >= %10.4e)\n",
           fitname, calval*fit255, calval*fit255_use);
  if (par->radobsmax != 0.0)
    printf("# %s  (max pixel value = %10.4e, bright white >= %10.4e)\n",
           obsname, obs255, obs255_use);

  /*  Set RGB levels for the fit image.  If no cutoff is being applied,
      each pixel's relative brightness is determined by the fit frame,
      while the color is determined by the values in the map_fit matrix
      (white = unmapped, saturated blue = maximally mapped).

      However, if a cutoff is being applied, pixels are left uncolored
      with brightness as determined by fit frame (below the cutoff) or
      else maximally colored with maximum brightness (above the cutoff).  */

  cfit = d3tensor( 1, ndel, 1, ndop, 0, 2);
  use_cutoff = (par->map_fitcutoff >= 0.0 && par->map_fitcutoff <= 1.0) ? 1 : 0;

  for (i=1; i<=ndel; i++)
    for (j=1; j<=ndop; j++)
      if (use_cutoff) {
          if (frame->map_fit[i][j] >= par->map_fitcutoff)
            for (icolor=0; icolor<=2; icolor++)
              cfit[i][j][icolor] = fit255_use*maplevels[icolor]/255.0;
          else
            for (icolor=0; icolor<=2; icolor++)
              cfit[i][j][icolor] = frame->fit[i][j]*clevels[icolor]/255.0;
      } else {
          for (icolor=0; icolor<=2; icolor++) {
            level = clevels[icolor]
                    + (maplevels[icolor] - clevels[icolor]) * frame->map_fit[i][j];
            cfit[i][j][icolor] = frame->fit[i][j]*level/255.0;
          }
      }

  /*  Write the ppm file for the fit image  */

  if (par->map_overflow)
    printf("WARNING: specified POS region maps beyond the delay-Doppler fit frame\n");
  printf("# %s\n", fitname);
  wimasppmsc( cfit,
              1, ndel, 1, ndop, par->radfitmin, fit255_use,
              par->dd_clockwiserot, par->dd_xflip, par->dd_yflip,
              fitname);
  fflush(stdout);

  /*  Write the pgm file for the obs image  */

  printf("# %s\n", obsname);
  wimaspgmsc( obs,
              1, ndel, 1, ndop, par->radobsmin, obs255_use,
              par->dd_clockwiserot, par->dd_xflip, par->dd_yflip,
              obsname);

  /*  Compute the array of normalized fit residuals  */

  res = matrix( 1, ndel, 1, ndop);
  for (i=1; i<=ndel; i++)
    for (j=1; j<=ndop; j++)
      res[i][j] = (obs[i][j] - calval*fit[i][j]) * sqrt(oneovervar[i][j]);

  /*  Write the floating-point absolute values of the residuals as a pgm file  */

  if (par->dd_resid > 0.0) {
    for (i=1; i<=ndel; i++)
      for (j=1; j<=ndop; j++)
        res[i][j] = fabs(res[i][j]);
    if (nframes > 100)
      sprintf( resname, "res_%02d_%03d.pgm", s, f);
    else
      sprintf( resname, "res_%02d_%02d.pgm", s, f);
    printf("# %s\n", resname);
    printf("#\n");
    wimaspgmsc( res, 1, ndel, 1, ndop,
                0.0, par->dd_resid,
                par->dd_clockwiserot, par->dd_xflip, par->dd_yflip,
                resname);
  }

  /*  Clean up  */

  free_d3tensor( cfit, 1, ndel, 1, ndop, 0, 2);
  free_matrix( res, 1, ndel, 1, ndop);
}


void write_map_fit_doppler( struct par_t *par, struct dopfrm_t *frame,
                            int s, int f, int nframes)
{
  FILE *fp;
  char fitname[MAXLEN];
  int ndop, j, use_cutoff;
  double o2, m2, om, calval, sc, maplevel;
  double *obs, *fit, *oneovervar;

  /*  Assign variables for this frame  */

  ndop = frame->ndop;
  obs = frame->obs;
  fit = frame->fit;
  oneovervar = frame->oneovervar;  /* 1/variance */

  /*  In order to get this frame's calibration factor we must go through most of
      the chi-square calculation.  Initialize contributions to chi-square to
      values that account for overflow (if any) beyond the limits of the data
      frame.  These contributions were computed by routine pos2doppler.           */

  o2 = frame->overflow_o2;
  m2 = frame->overflow_m2;
  om = 0.0;

  /*  Now add the contributions from power within the limits of the data frame.  */

  for (j=1; j<=ndop; j++) {
    o2 += obs[j]*obs[j]*oneovervar[j];
    m2 += fit[j]*fit[j]*oneovervar[j];
    om += fit[j]*obs[j]*oneovervar[j];
  }

  /*  If this frame's calibration factor is allowed to float,
      set it to minimize chi-square, the sum over all bins of
      { (obs - calfact*fit)^2 / variance }.                     */

  if (frame->cal.state == 'f') {
    if (om > 0.0) {
        frame->cal.val = om/m2;
    } else {
        frame->cal.val = TINYCALFACT;
        printf("WARNING: set %2d frame %2d had negative calfact reset to %10.4e\n",
               s, f, frame->cal.val);
    }
  }
  calval = frame->cal.val;

  /*  Zero out obs and fit bins that have been zeroed out in a pixel mask  */

  if (frame->pixels_weighted)
    for (j=1; j<=ndop; j++)
      if (oneovervar[j] == 0.0) {
        obs[j] = 0.0;
        fit[j] = 0.0;
      }

  /*  Determine whether or not we're using a mapping cutoff  */

  use_cutoff = (par->map_fitcutoff >= 0.0 && par->map_fitcutoff <= 1.0) ? 1 : 0;

  /*  Write the obs, fit, and res spectra and the fit mapping to disk  */

  if (par->map_overflow)
    printf("WARNING: specified POS region maps beyond the Doppler fit frame\n");
  if (nframes > 100)
    sprintf( fitname, "fit_%02d_%03d.map.dat", s, f);
  else
    sprintf( fitname, "fit_%02d_%02d.map.dat", s, f);
  printf("# %s\n", fitname);
  printf("#\n");
  fflush(stdout);
  FOPEN( fp, fitname, "w");
  sc = 1/frame->sdev;
  for (j=1; j<=ndop; j++) {
    if (use_cutoff)
      maplevel = (frame->map_fit[j] >= par->map_fitcutoff) ? 1.0 : 0.0;
    else
      maplevel = frame->map_fit[j];
    fprintf( fp, "%d %f %f %f %f\n", j,
             obs[j]*sc, calval*fit[j]*sc, (obs[j] - calval*fit[j])*sc,
             maplevel*calval*fit[j]*sc);
  }
  fclose( fp);
}
