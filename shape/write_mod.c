/*****************************************************************************************
                                                                              write_mod.c

Writes an asteroid model of the type used by shape.c.  Provide a file name and a mod_t
structure.

Modified 2015 May 9 by CM:
    Tiny aesthetic fix: add a space before each of the new libration parameters

Modified 2014 Aug 25 by SN:
    Add new parameters "spin.lib_amp", "spin.lib_freq","spin.lib_phase"

Modified 2014 February 12 by CM:
    Implement multiple radar and optical scattering laws

Modified 2013 May 20 by CM:
    Implement ovoid shape components

Modified 2011 September 2 by CM:
    Add the "harmlambert" and "inholambert" scattering laws

Modified 2011 August 12 by CM:
    Write spin impulses

Modified 2010 June 1 by CM:
    Change "scalefactor" parameter from a scalar to a 3-component vector
        for harmonic and vertex shape structures

Modified 2010 April 27 by CM:
    Implement "tabular" radar scattering law

Modified 2009 March 25 by CM:
    In write_spin, leave the model's Euler angles alone and just shift the
        values that are written to disk (or else we'd have to tell the
        branch nodes about the shifted values in a parallel-processing fit)
    In write_spin, allow for a bit of slop in the last floating-point digit
        so that we don't shift the Euler angles when, for example,
        "angle 1" = -1e-15 (i.e., when pole latitude = pi/2 + 1e-15)

Modified 2007 August 18 by CM:
    Don't include build.h (which is now included in head.h)

Modified 2007 August 4 by CM:
    Shift spin Euler angles to the range [0,360) deg rather than
        (-180,180] deg before writing them to disk

Modified 2006 October 1 by CM:
    Add "scalefactor" to harmonic and vertex shape structures
    Replace ellipsoid diameters D with two_a, a_over_b, b_over_c

Modified 2006 March 6 by PT:
    Add new parameter "spin.omegadot"

Modified 2005 September 11 by CM:
    Implement "photoharm" action
    Move "photofacets" routine to photoconvert.c

Modified 2005 September 8 by CM:
    Implement "harmlommel" "harmhapke" and "harmkaas" optical
        scattering laws
    Implement "harmcosine" radar scattering law
    Implement "photofacets" action by adding photofacets routine
        (and adding the par argument to write_photo)

Modified 2005 August 8 by CM:
    Implement "inhokaas" optical scattering law

Modified 2005 July 20 by CM:
    Implement "hagfors" and "cosine_qs" and "gauss+cosine" and
        "hagfors+cosine" and "cosine+cosine" and inhomogeneous "inhocosine"
        radar scattering laws
    Eliminate "flat" radar scattering law

Modified 2005 July 4 by CM:
    Adjust code for inhomogeneous "inholommel" and "inhohapke" optical
        scattering laws so that they can be used for ellipsoid and
        harmonic models, not just for vertex models

Modified 2005 April 21 by CM:
    Convert pole directions with |ecliptic latitude| > 90 deg to more
        conventional directions before writing to disk

Modified 2005 January 25 by CM:
    Eliminated unused variables

Modified 2004 July 2 by CM:
    For "facets" action, don't bother computing the direction cosines
        for the vertices in a model component if that component is already
        a vertex component

Modified 2004 April 29 by CM:
    For Kaasalainen scattering law, switch from weighting factor "c"
        (ranging from 0 to infinity) to "wt" (ranging from 0 to 1)

Modified 2004 March 15 by CM:
    Changed "%12.6e" output to "%13.6e" so that columns are aligned
    even when negative values are present

Modified 2004 February 25 by CM:
    Add Kaasalainen "Lambert + Lommel-Seeliger" scattering law

Modified 2003 October 27 by CM:
    Read version number from build.h
*****************************************************************************************/

#include "head.h"

void write_shape( FILE *fp, struct par_t *par, struct mod_t *mod);
void write_photo( FILE *fp, struct par_t *par, struct mod_t *mod);
void write_spin( FILE *fp, struct mod_t *mod);
void write_misc( FILE *fp, struct mod_t *mod);


void write_mod( struct par_t *par, struct mod_t *mod)
{
  FILE *fp;
  
  printf("# writing model to file: %s ...\n", mod->name);
  FOPEN( fp, mod->name, "w");
  fprintf( fp, "{MODEL FILE FOR SHAPE.C VERSION %s BUILD %s}\n\n", VERSION, BUILD);
  write_shape( fp, par, mod);
  write_photo( fp, par, mod);
  write_spin( fp, mod);
  write_misc( fp, mod);
  fclose( fp); 
  printf("# writing completed\n");
  fflush(stdout);
}


void write_shape( FILE *fp, struct par_t *par, struct mod_t *mod)
{
  int i, j, k, l, input_component_type;
  struct vertices_t *verts;
  double u[3];

  fprintf( fp, "{SHAPE DESCRIPTION}\n");
  fprintf( fp, "%14d {number of components}\n", mod->shape.ncomp);

  for (i=0; i<mod->shape.ncomp; i++) {
    fprintf( fp, "{COMPONENT %d}\n", i);
    for (j=0; j<=2; j++)
      fprintf( fp, " %c %13.6e {linear offset %d}\n",
                   mod->shape.comp[i].off[j].state,
                   mod->shape.comp[i].off[j].val, j);
    for (j=0; j<=2; j++)
      fprintf( fp, " %c %13.6e {rotational offset %d}\n", 
                   mod->shape.comp[i].rot[j].state,
                   mod->shape.comp[i].rot[j].val*R2D, j);

    input_component_type = mod->shape.comp[i].type;
    if (par->action == FACETS)
      mod->shape.comp[i].type = VERTEX;

    if (mod->shape.comp[i].type == ELLIPSE) {

        /*  triaxial ellipsoid component  */
        fprintf( fp, "%14s {component type}\n", "ellipse");
        fprintf( fp, " %c %12.6f {2a}\n", 
                     mod->shape.comp[i].desc.ell.two_a.state,
                     mod->shape.comp[i].desc.ell.two_a.val);
        fprintf( fp, " %c %12.6f {a/b}\n", 
                     mod->shape.comp[i].desc.ell.a_over_b.state,
                     mod->shape.comp[i].desc.ell.a_over_b.val);
        fprintf( fp, " %c %12.6f {b/c}\n", 
                     mod->shape.comp[i].desc.ell.b_over_c.state,
                     mod->shape.comp[i].desc.ell.b_over_c.val);
        fprintf( fp, "%14d {number of theta steps}\n",
                     mod->shape.comp[i].desc.ell.ntheta);

    } else if (mod->shape.comp[i].type == OVOID) {

        /*  ovoid component  */
        fprintf( fp, "%14s {component type}\n", "ovoid");
        fprintf( fp, " %c %12.6f {2a}\n", 
                     mod->shape.comp[i].desc.ovoid.two_a.state,
                     mod->shape.comp[i].desc.ovoid.two_a.val);
        fprintf( fp, " %c %12.6f {a/b}\n", 
                     mod->shape.comp[i].desc.ovoid.a_over_b.state,
                     mod->shape.comp[i].desc.ovoid.a_over_b.val);
        fprintf( fp, " %c %12.6f {b/c}\n", 
                     mod->shape.comp[i].desc.ovoid.b_over_c.state,
                     mod->shape.comp[i].desc.ovoid.b_over_c.val);
        fprintf( fp, " %c %12.6f {k}\n", 
                     mod->shape.comp[i].desc.ovoid.k.state,
                     mod->shape.comp[i].desc.ovoid.k.val);
        fprintf( fp, "%14d {number of theta steps}\n",
                     mod->shape.comp[i].desc.ovoid.ntheta);

    } else if (mod->shape.comp[i].type == HARMONIC) {

        /*  spherical harmonic component  */
        fprintf( fp, "%14s {component type}\n", "harmonic");
        fprintf( fp, "%14d {number of harmonics}\n",
                     mod->shape.comp[i].desc.har.nhar);
        for (j=0; j<=2; j++)
          fprintf( fp, " %c %13.6e {scale factor %d}\n", 
                       mod->shape.comp[i].desc.har.scalefactor[j].state,
                       mod->shape.comp[i].desc.har.scalefactor[j].val, j);
        for (j=0; j<=mod->shape.comp[i].desc.har.nhar; j++) {
          fprintf( fp, " %c %13.6e {a[%2d][ 0]}\n", 
                       mod->shape.comp[i].desc.har.a[j][0].state,
                       mod->shape.comp[i].desc.har.a[j][0].val, j);
          for (k=1; k<=j; k++) {
            fprintf( fp, " %c %13.6e {a[%2d][%2d]}\n", 
                         mod->shape.comp[i].desc.har.a[j][k].state,
                         mod->shape.comp[i].desc.har.a[j][k].val, j, k);
            fprintf( fp, " %c %13.6e {b[%2d][%2d]}\n", 
                         mod->shape.comp[i].desc.har.b[j][k].state,
                         mod->shape.comp[i].desc.har.b[j][k].val, j, k);
          }
        }
        fprintf( fp, "%14d {number of theta steps}\n",
                     mod->shape.comp[i].desc.har.ntheta);

    } else if (mod->shape.comp[i].type == VERTEX) {

        /*  (triangular) faceted vertex component  */
        if (par->action != FACETS || input_component_type == VERTEX) {
            verts = &mod->shape.comp[i].desc.ver;
        } else {

            /*  For the "facets" action, if we are converting an ellipsoid or
                spherical harmonic model to its vertex realization, compute the
                direction cosines of each vertex: sum the unit normals to all
                facets attached to that vertex; normalize the sum; and set the
                direction cosines according to the direction of this unit vector.  */

            verts = &mod->shape.comp[i].real;
            for (j=0; j<verts->nv; j++) {
              u[0] = u[1] = u[2] = 0.0;
              for (k=0; k<verts->v[j].naf; k++)
                for (l=0; l<=2; l++)
                  u[l] += verts->f[verts->v[j].af[k]].n[l];
              normalize( u);
              for (l=0; l<=2; l++) {
                verts->v[j].u[l] = u[l];
                verts->v[j].a[l] = verts->v[j].x[l];
              }
              verts->v[j].r.val = 0.0;
            }
        }
        fprintf( fp, "%14s {component type}\n", "vertex");
        fprintf( fp, "%14d {number of vertices}\n", verts->nv);
        for (j=0; j<=2; j++)
          fprintf( fp, " %c %13.6e {scale factor %d}\n", 
                       verts->scalefactor[j].state, verts->scalefactor[j].val, j);
        for (j=0; j<verts->nv; j++) {
          fprintf( fp, " %c %13.6e   %13.6e %13.6e %13.6e\n", 
                       verts->v[j].r.state, verts->v[j].r.val,
                       verts->v[j].u[0],
                       verts->v[j].u[1],
                       verts->v[j].u[2]);
          fprintf( fp, "\t\t  %13.6e %13.6e %13.6e {v %d}\n", 
                       verts->v[j].a[0],
                       verts->v[j].a[1],
                       verts->v[j].a[2], j);
        }
        fprintf( fp, "%14d {number of facets}\n", verts->nf);
        for (j=0; j<verts->nf; j++) {
          fprintf( fp, " %6d %6d %6d {f %d}\n", 
                       verts->f[j].v[0],
                       verts->f[j].v[1],
                       verts->f[j].v[2], j);
        }
    }
  }
}


void write_photo( FILE *fp, struct par_t *par, struct mod_t *mod)
{
  int ilaw, c, f, L, l, m, i;

  /*  Convert the radar and/or optical scattering law to its faceted
      or spherical harmonic inhomogeneous realization if requested    */
  if (par->action == PHOTOFACETS)
    photofacets( par, mod);
  else if (par->action == PHOTOHARM)
    photoharm( par, mod);

  /*  Write the radar scattering law(s) to disk  */
  fprintf( fp, "\n\n{PHOTOMETRIC FUNCTIONS}\n");
  fprintf( fp, "%14d {number of radar scattering laws}\n", mod->photo.nradlaws);

  for (ilaw=0; ilaw<mod->photo.nradlaws; ilaw++) {
    fprintf( fp, "{RADAR SCATTERING LAW %d}\n", ilaw);
    if (mod->photo.radtype[ilaw] == NOLAW) {
        fprintf( fp, "%14s {type}\n", "none");
    } else if (mod->photo.radtype[ilaw] == COSINELAW_DIFF) {
        fprintf( fp, "%14s {type}\n", "cosine");
        fprintf( fp, " %c %12.6f {R}\n", mod->photo.radar[ilaw].RC.R.state,
                     mod->photo.radar[ilaw].RC.R.val);
        fprintf( fp, " %c %12.6f {C}\n", mod->photo.radar[ilaw].RC.C.state,
                     mod->photo.radar[ilaw].RC.C.val);
    } else if (mod->photo.radtype[ilaw] == TABULARLAW) {
        fprintf( fp, "%14s {type}\n", "tabular");
        fprintf( fp, "%14d {number of coefficients}\n", mod->photo.radar[ilaw].tabular.n);
        for (i=0; i<mod->photo.radar[ilaw].tabular.n; i++)
          fprintf( fp, " %c %13.6e {rho[%2d]}\n", mod->photo.radar[ilaw].tabular.rho[i].state,
                       mod->photo.radar[ilaw].tabular.rho[i].val, i);
    } else if (mod->photo.radtype[ilaw] == GAUSSIANLAW) {
        fprintf( fp, "%14s {type}\n", "gaussian");
        fprintf( fp, " %c %12.6f {R}\n", mod->photo.radar[ilaw].quasispec.R.state,
                     mod->photo.radar[ilaw].quasispec.R.val);
        fprintf( fp, " %c %12.6f {C}\n", mod->photo.radar[ilaw].quasispec.C.state,
                     mod->photo.radar[ilaw].quasispec.C.val);
        fprintf( fp, "   %12.6f {cutoff angle}\n",
                     acos( mod->photo.radar[ilaw].quasispec.cos_cutoff)*R2D);
    } else if (mod->photo.radtype[ilaw] == HAGFORSLAW) {
        fprintf( fp, "%14s {type}\n", "hagfors");
        fprintf( fp, " %c %12.6f {R}\n", mod->photo.radar[ilaw].quasispec.R.state,
                     mod->photo.radar[ilaw].quasispec.R.val);
        fprintf( fp, " %c %12.6f {C}\n", mod->photo.radar[ilaw].quasispec.C.state,
                     mod->photo.radar[ilaw].quasispec.C.val);
        fprintf( fp, "   %12.6f {cutoff angle}\n",
                     acos( mod->photo.radar[ilaw].quasispec.cos_cutoff)*R2D);
    } else if (mod->photo.radtype[ilaw] == COSINELAW_QS) {
        fprintf( fp, "%14s {type}\n", "cosine_qs");
        fprintf( fp, " %c %12.6f {R}\n", mod->photo.radar[ilaw].quasispec.R.state,
                     mod->photo.radar[ilaw].quasispec.R.val);
        fprintf( fp, " %c %12.6f {C}\n", mod->photo.radar[ilaw].quasispec.C.state,
                     mod->photo.radar[ilaw].quasispec.C.val);
        fprintf( fp, "   %12.6f {cutoff angle}\n",
                     acos( mod->photo.radar[ilaw].quasispec.cos_cutoff)*R2D);
    } else if (mod->photo.radtype[ilaw] == GAUSSIAN_COSINE) {
        fprintf( fp, "%14s {type}\n", "gauss+cosine");
        fprintf( fp, " %c %12.6f {R_qs}\n", mod->photo.radar[ilaw].hybrid.qs.R.state,
                     mod->photo.radar[ilaw].hybrid.qs.R.val);
        fprintf( fp, " %c %12.6f {C_qs}\n", mod->photo.radar[ilaw].hybrid.qs.C.state,
                     mod->photo.radar[ilaw].hybrid.qs.C.val);
        fprintf( fp, "   %12.6f {qs cutoff angle}\n",
                     acos( mod->photo.radar[ilaw].hybrid.qs.cos_cutoff)*R2D);
        fprintf( fp, " %c %12.6f {R_diff}\n", mod->photo.radar[ilaw].hybrid.diff.R.state,
                     mod->photo.radar[ilaw].hybrid.diff.R.val);
        fprintf( fp, " %c %12.6f {C_diff}\n", mod->photo.radar[ilaw].hybrid.diff.C.state,
                     mod->photo.radar[ilaw].hybrid.diff.C.val);
    } else if (mod->photo.radtype[ilaw] == HAGFORS_COSINE) {
        fprintf( fp, "%14s {type}\n", "hagfors+cosine");
        fprintf( fp, " %c %12.6f {R_qs}\n", mod->photo.radar[ilaw].hybrid.qs.R.state,
                     mod->photo.radar[ilaw].hybrid.qs.R.val);
        fprintf( fp, " %c %12.6f {C_qs}\n", mod->photo.radar[ilaw].hybrid.qs.C.state,
                     mod->photo.radar[ilaw].hybrid.qs.C.val);
        fprintf( fp, "   %12.6f {qs cutoff angle}\n",
                     acos( mod->photo.radar[ilaw].hybrid.qs.cos_cutoff)*R2D);
        fprintf( fp, " %c %12.6f {R_diff}\n", mod->photo.radar[ilaw].hybrid.diff.R.state,
                     mod->photo.radar[ilaw].hybrid.diff.R.val);
        fprintf( fp, " %c %12.6f {C_diff}\n", mod->photo.radar[ilaw].hybrid.diff.C.state,
                     mod->photo.radar[ilaw].hybrid.diff.C.val);
    } else if (mod->photo.radtype[ilaw] == COSINE_COSINE) {
        fprintf( fp, "%14s {type}\n", "cosine+cosine");
        fprintf( fp, " %c %12.6f {R_qs}\n", mod->photo.radar[ilaw].hybrid.qs.R.state,
                     mod->photo.radar[ilaw].hybrid.qs.R.val);
        fprintf( fp, " %c %12.6f {C_qs}\n", mod->photo.radar[ilaw].hybrid.qs.C.state,
                     mod->photo.radar[ilaw].hybrid.qs.C.val);
        fprintf( fp, "   %12.6f {qs cutoff angle}\n",
                     acos( mod->photo.radar[ilaw].hybrid.qs.cos_cutoff)*R2D);
        fprintf( fp, " %c %12.6f {R_diff}\n", mod->photo.radar[ilaw].hybrid.diff.R.state,
                     mod->photo.radar[ilaw].hybrid.diff.R.val);
        fprintf( fp, " %c %12.6f {C_diff}\n", mod->photo.radar[ilaw].hybrid.diff.C.state,
                     mod->photo.radar[ilaw].hybrid.diff.C.val);
    } else if (mod->photo.radtype[ilaw] == HARMCOSINE_DIFF) {
        fprintf( fp, "%14s {type}\n", "harmcosine");

        L = mod->photo.radar[ilaw].harmcosine.R.nhar;
        fprintf( fp, "%14d {number of harmonics: R}\n", L);
        for (l=0; l<=L; l++)
          for (m=0; m<=l; m++) {
            fprintf( fp, " %c %13.6e {a[%2d][%2d]: R}\n",
                         mod->photo.radar[ilaw].harmcosine.R.a[l][m].state,
                         mod->photo.radar[ilaw].harmcosine.R.a[l][m].val, l, m);
            if (m > 0)
              fprintf( fp, " %c %13.6e {b[%2d][%2d]: R}\n",
                           mod->photo.radar[ilaw].harmcosine.R.b[l][m].state,
                           mod->photo.radar[ilaw].harmcosine.R.b[l][m].val, l, m);
          }

        L = mod->photo.radar[ilaw].harmcosine.C.nhar;
        fprintf( fp, "%14d {number of harmonics: C}\n", L);
        for (l=0; l<=L; l++)
          for (m=0; m<=l; m++) {
            fprintf( fp, " %c %13.6e {a[%2d][%2d]: C}\n",
                         mod->photo.radar[ilaw].harmcosine.C.a[l][m].state,
                         mod->photo.radar[ilaw].harmcosine.C.a[l][m].val, l, m);
            if (m > 0)
              fprintf( fp, " %c %13.6e {b[%2d][%2d]: C}\n",
                           mod->photo.radar[ilaw].harmcosine.C.b[l][m].state,
                           mod->photo.radar[ilaw].harmcosine.C.b[l][m].val, l, m);
          }
    } else if (mod->photo.radtype[ilaw] == INHOCOSINE_DIFF) {
        fprintf( fp, "%14s {type}\n", "inhocosine");
        fprintf( fp, " %c %13.6e {global R}\n",
                     mod->photo.radar[ilaw].inhocosine.global.R.state,
                     mod->photo.radar[ilaw].inhocosine.global.R.val);
        fprintf( fp, " %c %13.6e {global C}\n",
                     mod->photo.radar[ilaw].inhocosine.global.C.state,
                     mod->photo.radar[ilaw].inhocosine.global.C.val);
        fprintf( fp, "%14d {number of components}\n", mod->shape.ncomp);
        for (c=0; c<mod->shape.ncomp; c++) {
          fprintf( fp, "%14d {number of facets in component %d}\n", 
                       mod->shape.comp[c].real.nf, c);
          for (f=0; f<mod->shape.comp[c].real.nf; f++) {
            fprintf( fp, " %c %13.6e",
                         mod->photo.radar[ilaw].inhocosine.local[c][f].R.state,
                         mod->photo.radar[ilaw].inhocosine.local[c][f].R.val);
            fprintf( fp, " %c %13.6e {facet[%d][%d]}\n", 
                         mod->photo.radar[ilaw].inhocosine.local[c][f].C.state,
                         mod->photo.radar[ilaw].inhocosine.local[c][f].C.val,
                         c, f);
          }
        }
    } else
        bailout("write_mod.c: invalid radar scattering law\n");
  }

  /*  Write the optical scattering law(s) to disk  */
  fprintf( fp, "%14d {number of optical scattering laws}\n", mod->photo.noptlaws);

  for (ilaw=0; ilaw<mod->photo.noptlaws; ilaw++) {
    fprintf( fp, "{OPTICAL SCATTERING LAW %d}\n", ilaw);
    if (mod->photo.opttype[ilaw] == NOLAW) {
        fprintf( fp, "%14s {type}\n", "none");
    } else if (mod->photo.opttype[ilaw] == LAMBERTLAW) {
        fprintf( fp, "%14s {type}\n", "lambertian");
        fprintf( fp, " %c %12.6f {R}\n", mod->photo.optical[ilaw].R.R.state,
                     mod->photo.optical[ilaw].R.R.val);
    } else if (mod->photo.opttype[ilaw] == GEOMETRICAL) {
        fprintf( fp, "%14s {type}\n", "geometrical");
        fprintf( fp, " %c %12.6f {R}\n", mod->photo.optical[ilaw].R.R.state,
                     mod->photo.optical[ilaw].R.R.val);
    } else if (mod->photo.opttype[ilaw] == LAMBERTLAW || mod->photo.opttype[ilaw] == LOMMEL) {
        if (mod->photo.opttype[ilaw] == LAMBERTLAW)
          fprintf( fp, "%14s {type}\n", "lambertian");
        else
          fprintf( fp, "%14s {type}\n", "lommel");
        fprintf( fp, " %c %12.6f {R}\n", mod->photo.optical[ilaw].R.R.state,
                     mod->photo.optical[ilaw].R.R.val);
    } else if (mod->photo.opttype[ilaw] == HARMLAMBERT || mod->photo.opttype[ilaw] == HARMLOMMEL) {
        if (mod->photo.opttype[ilaw] == HARMLAMBERT)
          fprintf( fp, "%14s {type}\n", "harmlambert");
        else
          fprintf( fp, "%14s {type}\n", "harmlommel");
        L = mod->photo.optical[ilaw].harmR.R.nhar;
        fprintf( fp, "%14d {number of harmonics: R}\n", L);
        for (l=0; l<=L; l++)
          for (m=0; m<=l; m++) {
            fprintf( fp, " %c %13.6e {a[%2d][%2d]: R}\n",
                         mod->photo.optical[ilaw].harmR.R.a[l][m].state,
                         mod->photo.optical[ilaw].harmR.R.a[l][m].val, l, m);
            if (m > 0)
              fprintf( fp, " %c %13.6e {b[%2d][%2d]: R}\n",
                           mod->photo.optical[ilaw].harmR.R.b[l][m].state,
                           mod->photo.optical[ilaw].harmR.R.b[l][m].val, l, m);
          }
    } else if (mod->photo.opttype[ilaw] == INHOLAMBERT || mod->photo.opttype[ilaw] == INHOLOMMEL) {
        if (mod->photo.opttype[ilaw] == INHOLAMBERT)
          fprintf( fp, "%14s {type}\n", "inholambert");
        else
          fprintf( fp, "%14s {type}\n", "inholommel");
        fprintf( fp, " %c %12.6e {global R}\n",
                     mod->photo.optical[ilaw].inhoR.global.R.state,
                     mod->photo.optical[ilaw].inhoR.global.R.val);
        fprintf( fp, "%14d {number of components}\n", mod->shape.ncomp);
        for (c=0; c<mod->shape.ncomp; c++) {
          fprintf( fp, "%14d {number of facets in component %d}\n", 
                       mod->shape.comp[c].real.nf, c);
          for (f=0; f<mod->shape.comp[c].real.nf; f++)
            fprintf( fp, " %c %12.6e {facet[%d][%d]}\n", 
                         mod->photo.optical[ilaw].inhoR.local[c][f].R.state,
                         mod->photo.optical[ilaw].inhoR.local[c][f].R.val, c, f);
        }
    } else if (mod->photo.opttype[ilaw] == HAPKE) {
        fprintf( fp, "%14s {type}\n", "hapke");
        fprintf( fp, " %c %13.6e {w}\n", mod->photo.optical[ilaw].hapke.w.state,
                     mod->photo.optical[ilaw].hapke.w.val);
        fprintf( fp, " %c %13.6e {h}\n", mod->photo.optical[ilaw].hapke.h.state,
                     mod->photo.optical[ilaw].hapke.h.val);
        fprintf( fp, " %c %13.6e {B0}\n", mod->photo.optical[ilaw].hapke.B0.state,
                     mod->photo.optical[ilaw].hapke.B0.val);
        fprintf( fp, " %c %13.6e {g}\n", mod->photo.optical[ilaw].hapke.g.state,
                     mod->photo.optical[ilaw].hapke.g.val);
        fprintf( fp, " %c %13.6e {theta}\n",
                     mod->photo.optical[ilaw].hapke.theta.state,
                     mod->photo.optical[ilaw].hapke.theta.val*R2D);
    } else if (mod->photo.opttype[ilaw] == HARMHAPKE) {
        fprintf( fp, "%14s {type}\n", "harmhapke");

        L = mod->photo.optical[ilaw].harmhapke.w.nhar;
        fprintf( fp, "%14d {number of harmonics: w}\n", L);
        for (l=0; l<=L; l++)
          for (m=0; m<=l; m++) {
            fprintf( fp, " %c %13.6e {a[%2d][%2d]: w}\n",
                         mod->photo.optical[ilaw].harmhapke.w.a[l][m].state,
                         mod->photo.optical[ilaw].harmhapke.w.a[l][m].val, l, m);
            if (m > 0)
              fprintf( fp, " %c %13.6e {b[%2d][%2d]: w}\n",
                           mod->photo.optical[ilaw].harmhapke.w.b[l][m].state,
                           mod->photo.optical[ilaw].harmhapke.w.b[l][m].val, l, m);
          }

        L = mod->photo.optical[ilaw].harmhapke.h.nhar;
        fprintf( fp, "%14d {number of harmonics: h}\n", L);
        for (l=0; l<=L; l++)
          for (m=0; m<=l; m++) {
            fprintf( fp, " %c %13.6e {a[%2d][%2d]: h}\n",
                         mod->photo.optical[ilaw].harmhapke.h.a[l][m].state,
                         mod->photo.optical[ilaw].harmhapke.h.a[l][m].val, l, m);
            if (m > 0)
              fprintf( fp, " %c %13.6e {b[%2d][%2d]: h}\n",
                           mod->photo.optical[ilaw].harmhapke.h.b[l][m].state,
                           mod->photo.optical[ilaw].harmhapke.h.b[l][m].val, l, m);
          }

        L = mod->photo.optical[ilaw].harmhapke.B0.nhar;
        fprintf( fp, "%14d {number of harmonics: B0}\n", L);
        for (l=0; l<=L; l++)
          for (m=0; m<=l; m++) {
            fprintf( fp, " %c %13.6e {a[%2d][%2d]: B0}\n",
                         mod->photo.optical[ilaw].harmhapke.B0.a[l][m].state,
                         mod->photo.optical[ilaw].harmhapke.B0.a[l][m].val, l, m);
            if (m > 0)
              fprintf( fp, " %c %13.6e {b[%2d][%2d]: B0}\n",
                           mod->photo.optical[ilaw].harmhapke.B0.b[l][m].state,
                           mod->photo.optical[ilaw].harmhapke.B0.b[l][m].val, l, m);
          }

        L = mod->photo.optical[ilaw].harmhapke.g.nhar;
        fprintf( fp, "%14d {number of harmonics: g}\n", L);
        for (l=0; l<=L; l++)
          for (m=0; m<=l; m++) {
            fprintf( fp, " %c %13.6e {a[%2d][%2d]: g}\n",
                         mod->photo.optical[ilaw].harmhapke.g.a[l][m].state,
                         mod->photo.optical[ilaw].harmhapke.g.a[l][m].val, l, m);
            if (m > 0)
              fprintf( fp, " %c %13.6e {b[%2d][%2d]: g}\n",
                           mod->photo.optical[ilaw].harmhapke.g.b[l][m].state,
                           mod->photo.optical[ilaw].harmhapke.g.b[l][m].val, l, m);
          }

        L = mod->photo.optical[ilaw].harmhapke.theta.nhar;
        fprintf( fp, "%14d {number of harmonics: theta}\n", L);
        for (l=0; l<=L; l++)
          for (m=0; m<=l; m++) {
            fprintf( fp, " %c %13.6e {a[%2d][%2d]: theta}\n",
                         mod->photo.optical[ilaw].harmhapke.theta.a[l][m].state,
                         mod->photo.optical[ilaw].harmhapke.theta.a[l][m].val*R2D, l, m);
            if (m > 0)
              fprintf( fp, " %c %13.6e {b[%2d][%2d]: theta}\n",
                           mod->photo.optical[ilaw].harmhapke.theta.b[l][m].state,
                           mod->photo.optical[ilaw].harmhapke.theta.b[l][m].val*R2D, l, m);
          }
    } else if (mod->photo.opttype[ilaw] == INHOHAPKE) {
        fprintf( fp, "%14s {type}\n", "inhohapke");
        fprintf( fp, " %c %13.6e {global w}\n",
                     mod->photo.optical[ilaw].inhohapke.global.w.state,
                     mod->photo.optical[ilaw].inhohapke.global.w.val);
        fprintf( fp, " %c %13.6e {global h}\n",
                     mod->photo.optical[ilaw].inhohapke.global.h.state,
                     mod->photo.optical[ilaw].inhohapke.global.h.val);
        fprintf( fp, " %c %13.6e {global B0}\n",
                     mod->photo.optical[ilaw].inhohapke.global.B0.state,
                     mod->photo.optical[ilaw].inhohapke.global.B0.val);
        fprintf( fp, " %c %13.6e {global g}\n",
                     mod->photo.optical[ilaw].inhohapke.global.g.state,
                     mod->photo.optical[ilaw].inhohapke.global.g.val);
        fprintf( fp, " %c %13.6e {global theta}\n",
                     mod->photo.optical[ilaw].inhohapke.global.theta.state,
                     mod->photo.optical[ilaw].inhohapke.global.theta.val*R2D);
        fprintf( fp, "%14d {number of components}\n", mod->shape.ncomp);
        for (c=0; c<mod->shape.ncomp; c++) {
          fprintf( fp, "%14d {number of facets in component %d}\n", 
                       mod->shape.comp[c].real.nf, c);
          for (f=0; f<mod->shape.comp[c].real.nf; f++) {
            fprintf( fp, " %c %13.6e",
                         mod->photo.optical[ilaw].inhohapke.local[c][f].w.state,
                         mod->photo.optical[ilaw].inhohapke.local[c][f].w.val);
            fprintf( fp, " %c %13.6e",
                         mod->photo.optical[ilaw].inhohapke.local[c][f].h.state,
                         mod->photo.optical[ilaw].inhohapke.local[c][f].h.val);
            fprintf( fp, " %c %13.6e",
                         mod->photo.optical[ilaw].inhohapke.local[c][f].B0.state,
                         mod->photo.optical[ilaw].inhohapke.local[c][f].B0.val);
            fprintf( fp, " %c %13.6e",
                         mod->photo.optical[ilaw].inhohapke.local[c][f].g.state,
                         mod->photo.optical[ilaw].inhohapke.local[c][f].g.val);
            fprintf( fp, " %c %13.6e {facet[%d][%d]}\n", 
                         mod->photo.optical[ilaw].inhohapke.local[c][f].theta.state,
                         mod->photo.optical[ilaw].inhohapke.local[c][f].theta.val*R2D,
                         c, f);
          }
        }
    } else if (mod->photo.opttype[ilaw] == KAASALAINEN) {
        fprintf( fp, "%14s {type}\n", "kaasalainen");
        fprintf( fp, " %c %12.6f {R}\n", mod->photo.optical[ilaw].kaas.R.state,
                     mod->photo.optical[ilaw].kaas.R.val);
        fprintf( fp, " %c %12.6f {wt}\n", mod->photo.optical[ilaw].kaas.wt.state,
                     mod->photo.optical[ilaw].kaas.wt.val);
        fprintf( fp, " %c %12.6f {A0}\n", mod->photo.optical[ilaw].kaas.A0.state,
                     mod->photo.optical[ilaw].kaas.A0.val);
        fprintf( fp, " %c %12.6f {D}\n", mod->photo.optical[ilaw].kaas.D.state,
                     mod->photo.optical[ilaw].kaas.D.val*R2D);
        fprintf( fp, " %c %12.6f {k}\n", mod->photo.optical[ilaw].kaas.k.state,
                     mod->photo.optical[ilaw].kaas.k.val/R2D);
    } else if (mod->photo.opttype[ilaw] == HARMKAAS) {
        fprintf( fp, "%14s {type}\n", "harmkaas");
  
        L = mod->photo.optical[ilaw].harmkaas.R.nhar;
        fprintf( fp, "%14d {number of harmonics: R}\n", L);
        for (l=0; l<=L; l++)
          for (m=0; m<=l; m++) {
            fprintf( fp, " %c %13.6e {a[%2d][%2d]: R}\n",
                         mod->photo.optical[ilaw].harmkaas.R.a[l][m].state,
                         mod->photo.optical[ilaw].harmkaas.R.a[l][m].val, l, m);
            if (m > 0)
              fprintf( fp, " %c %13.6e {b[%2d][%2d]: R}\n",
                           mod->photo.optical[ilaw].harmkaas.R.b[l][m].state,
                           mod->photo.optical[ilaw].harmkaas.R.b[l][m].val, l, m);
          }

        L = mod->photo.optical[ilaw].harmkaas.wt.nhar;
        fprintf( fp, "%14d {number of harmonics: wt}\n", L);
        for (l=0; l<=L; l++)
          for (m=0; m<=l; m++) {
            fprintf( fp, " %c %13.6e {a[%2d][%2d]: wt}\n",
                         mod->photo.optical[ilaw].harmkaas.wt.a[l][m].state,
                         mod->photo.optical[ilaw].harmkaas.wt.a[l][m].val, l, m);
            if (m > 0)
              fprintf( fp, " %c %13.6e {b[%2d][%2d]: wt}\n",
                           mod->photo.optical[ilaw].harmkaas.wt.b[l][m].state,
                           mod->photo.optical[ilaw].harmkaas.wt.b[l][m].val, l, m);
          }

        L = mod->photo.optical[ilaw].harmkaas.A0.nhar;
        fprintf( fp, "%14d {number of harmonics: A0}\n", L);
        for (l=0; l<=L; l++)
          for (m=0; m<=l; m++) {
            fprintf( fp, " %c %13.6e {a[%2d][%2d]: A0}\n",
                         mod->photo.optical[ilaw].harmkaas.A0.a[l][m].state,
                         mod->photo.optical[ilaw].harmkaas.A0.a[l][m].val, l, m);
            if (m > 0)
              fprintf( fp, " %c %13.6e {b[%2d][%2d]: A0}\n",
                           mod->photo.optical[ilaw].harmkaas.A0.b[l][m].state,
                           mod->photo.optical[ilaw].harmkaas.A0.b[l][m].val, l, m);
          }

        L = mod->photo.optical[ilaw].harmkaas.D.nhar;
        fprintf( fp, "%14d {number of harmonics: D}\n", L);
        for (l=0; l<=L; l++)
          for (m=0; m<=l; m++) {
            fprintf( fp, " %c %13.6e {a[%2d][%2d]: D}\n",
                         mod->photo.optical[ilaw].harmkaas.D.a[l][m].state,
                         mod->photo.optical[ilaw].harmkaas.D.a[l][m].val*R2D, l, m);
            if (m > 0)
              fprintf( fp, " %c %13.6e {b[%2d][%2d]: D}\n",
                           mod->photo.optical[ilaw].harmkaas.D.b[l][m].state,
                           mod->photo.optical[ilaw].harmkaas.D.b[l][m].val*R2D, l, m);
          }

        L = mod->photo.optical[ilaw].harmkaas.k.nhar;
        fprintf( fp, "%14d {number of harmonics: k}\n", L);
        for (l=0; l<=L; l++)
          for (m=0; m<=l; m++) {
            fprintf( fp, " %c %13.6e {a[%2d][%2d]: k}\n",
                         mod->photo.optical[ilaw].harmkaas.k.a[l][m].state,
                         mod->photo.optical[ilaw].harmkaas.k.a[l][m].val/R2D, l, m);
            if (m > 0)
              fprintf( fp, " %c %13.6e {b[%2d][%2d]: k}\n",
                           mod->photo.optical[ilaw].harmkaas.k.b[l][m].state,
                           mod->photo.optical[ilaw].harmkaas.k.b[l][m].val/R2D, l, m);
          }
    } else if (mod->photo.opttype[ilaw] == INHOKAAS) {
        fprintf( fp, "%14s {type}\n", "inhokaas");
        fprintf( fp, " %c %12.6f {global R}\n",
                     mod->photo.optical[ilaw].inhokaas.global.R.state,
                     mod->photo.optical[ilaw].inhokaas.global.R.val);
        fprintf( fp, " %c %12.6f {global wt}\n",
                     mod->photo.optical[ilaw].inhokaas.global.wt.state,
                     mod->photo.optical[ilaw].inhokaas.global.wt.val);
        fprintf( fp, " %c %12.6f {global A0}\n",
                     mod->photo.optical[ilaw].inhokaas.global.A0.state,
                     mod->photo.optical[ilaw].inhokaas.global.A0.val);
        fprintf( fp, " %c %12.6f {global D}\n",
                     mod->photo.optical[ilaw].inhokaas.global.D.state,
                     mod->photo.optical[ilaw].inhokaas.global.D.val*R2D);
        fprintf( fp, " %c %12.6f {global k}\n",
                     mod->photo.optical[ilaw].inhokaas.global.k.state,
                     mod->photo.optical[ilaw].inhokaas.global.k.val/R2D);
        fprintf( fp, "%14d {number of components}\n", mod->shape.ncomp);
        for (c=0; c<mod->shape.ncomp; c++) {
          fprintf( fp, "%14d {number of facets in component %d}\n", 
                       mod->shape.comp[c].real.nf, c);
          for (f=0; f<mod->shape.comp[c].real.nf; f++) {
            fprintf( fp, " %c %12.6f",
                         mod->photo.optical[ilaw].inhokaas.local[c][f].R.state,
                         mod->photo.optical[ilaw].inhokaas.local[c][f].R.val);
            fprintf( fp, " %c %12.6f",
                         mod->photo.optical[ilaw].inhokaas.local[c][f].wt.state,
                         mod->photo.optical[ilaw].inhokaas.local[c][f].wt.val);
            fprintf( fp, " %c %12.6f",
                         mod->photo.optical[ilaw].inhokaas.local[c][f].A0.state,
                         mod->photo.optical[ilaw].inhokaas.local[c][f].A0.val);
            fprintf( fp, " %c %12.6f",
                         mod->photo.optical[ilaw].inhokaas.local[c][f].D.state,
                         mod->photo.optical[ilaw].inhokaas.local[c][f].D.val*R2D);
            fprintf( fp, " %c %12.6f {facet[%d][%d]}\n", 
                         mod->photo.optical[ilaw].inhokaas.local[c][f].k.state,
                         mod->photo.optical[ilaw].inhokaas.local[c][f].k.val/R2D,
                         c, f);
          }
        }
    } else
        bailout("write_mod.c: invalid optical scattering law\n");
  }
}


void write_spin( FILE *fp, struct mod_t *mod)
{
  int i, cd[6], n;
  double twopi, x, angle[3];

  /* Leave model's Euler angles alone (in case there are branch nodes) and
   * work with temporary copies of their values  */
  for (i=0; i<=2; i++)
    angle[i] = mod->spin.angle[i].val;

  /* Deal with spin vectors that are "over the pole":
   *      |ecliptic pole latitude| > 90 deg
   *
   * To do this, shift the second Euler angle (= pi/2 - pole latitude) into the
   * range [0, 2*pi) rad; if it's now > pi, fold it around pi and add pi to 1st
   * Euler angle (= pi/2 + pole longitude) and to 3rd Euler angle (orientation
   * about rotation axis). Allow for slop in last floating-point digit.         */
  twopi = 2*PIE;
  if (angle[1] < -SMALLVAL || angle[1] >= twopi)
    angle[1] -= twopi*floor(angle[1]/twopi);
  if (angle[1] > PIE + SMALLVAL) {
    angle[1] = twopi - angle[1];
    angle[0] += PIE;
    angle[2] += PIE;
  }

  /* Now shift the 1st and 3rd Euler angles into the range [0, 2*pi)rad (again
   * allowing for slop in the last floating-point digit)               */
  if (angle[0] < -SMALLVAL || angle[0] >= twopi)
	  angle[0] -= twopi*floor(angle[0]/twopi);
  if (angle[2] < -SMALLVAL || angle[2] >= twopi)
      angle[2] -= twopi*floor(angle[2]/twopi);

  /*  Write spin state to disk  */
  fprintf( fp, "\n\n{SPIN STATE}\n");
  jd2cal( &cd[0], &cd[1], &cd[2], &cd[3], &cd[4], &cd[5], mod->spin.t0);
  fprintf( fp, "%4d %2d %2d %2d %2d %2d {yyyy mo dd hh mm ss of t0}\n",
               cd[0], cd[1], cd[2], cd[3], cd[4], cd[5]);
  x = angle[0]*R2D - 90.0;
  if (x < 0.0)
    x += 360.0;
  fprintf( fp, " %c %16.10f {angle 0 (deg) lambda=%f}\n", 
               mod->spin.angle[0].state, angle[0]*R2D, x);
  fprintf( fp, " %c %16.10f {angle 1 (deg) beta=%f}\n", 
               mod->spin.angle[1].state, angle[1]*R2D,
               90.0 - angle[1]*R2D);
  fprintf( fp, " %c %16.10f {angle 2 (deg)}\n",
               mod->spin.angle[2].state, angle[2]*R2D);
  for (i=0; i<=1; i++)
    fprintf( fp, " %c %16.10f {spin %d (deg/day)}\n",
                 mod->spin.omega[i].state,
                 mod->spin.omega[i].val*R2D, i);
  fprintf( fp, " %c %16.10f {spin %d (deg/day) P=%f}\n",
               mod->spin.omega[2].state,
               mod->spin.omega[2].val*R2D, 2,
               48.0*PIE/mod->spin.omega[2].val);
  for (i=0; i<=2; i++)
    fprintf( fp, " %c %16.10f {moment of inertia %d}\n",
                 mod->spin.inertia[i].state, mod->spin.inertia[i].val, i);
  for (i=0; i<=2; i++)
    fprintf( fp, " %c %16.10f {spin %d dot (deg/day/day)}\n",
	         mod->spin.omegadot[i].state,
	         mod->spin.omegadot[i].val*R2D, i);
  fprintf( fp, " %c %16.10f {Libration Amplitude (degrees)}\n",
	   mod->spin.lib_amp.state,
           mod->spin.lib_amp.val*R2D);

  fprintf( fp, " %c %16.10f {Libration Frequency (degrees/day)}\n",
	   mod->spin.lib_freq.state,
           mod->spin.lib_freq.val*R2D);

  fprintf( fp, " %c %16.10f {Libration Phase (degrees)}\n",
	   mod->spin.lib_phase.state,
           mod->spin.lib_phase.val*R2D);

  fprintf( fp, "   %16d {number of spin impulses}\n", mod->spin.n_impulse);
  for (n=0; n<mod->spin.n_impulse; n++) {
    jd2cal( &cd[0], &cd[1], &cd[2], &cd[3], &cd[4], &cd[5], mod->spin.t_impulse[n]);
    fprintf( fp, "%4d %2d %2d %2d %2d %2d ", cd[0], cd[1], cd[2], cd[3], cd[4], cd[5]);
    for (i=0; i<=2; i++)
      fprintf( fp, " %c %16.10f",
               mod->spin.impulse[n][i].state, mod->spin.impulse[n][i].val*R2D);
    fprintf( fp, " {spin impulse %d}\n", n);
  }
}


void write_misc( FILE *fp, struct mod_t *mod)
{
  int i, j;
  double max;

  fprintf( fp, "\n{\n");
  fprintf( fp, "volume  = %f km^3\n", mod->shape.volume);
  fprintf( fp, "com     = %f %f %f km\n", mod->shape.com[0],
               mod->shape.com[1], mod->shape.com[2]);
  max = 0.0;
  for (i=0; i<=2; i++)
    for (j=0; j<=2; j++)
      if (fabs(mod->shape.inertia[i][j]) > max)
        max = fabs(mod->shape.inertia[i][j]);
  fprintf( fp, "inertia = %9.6f %9.6f %9.6f\n", mod->shape.inertia[0][0]/max,
               mod->shape.inertia[0][1]/max, mod->shape.inertia[0][2]/max);
  fprintf( fp, "          %9.6f %9.6f %9.6f\n", mod->shape.inertia[1][0]/max,
               mod->shape.inertia[1][1]/max, mod->shape.inertia[1][2]/max);
  fprintf( fp, "          %9.6f %9.6f %9.6f\n", mod->shape.inertia[2][0]/max,
               mod->shape.inertia[2][1]/max, mod->shape.inertia[2][2]/max);
  fprintf( fp, "}\n");
}
