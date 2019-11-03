/***************************************************************************
                                                                split_mod.c

Take a one-component input model that has already been realized, split it
into two pieces along a user-specified dividing plane (adding interpolated
vertices in the plane as needed), close off each piece by creating new
facets in that plane, and write each piece to disk as a mod file.  These
two mod files have the same name as the input mod file but with ".split1"
and ".split2" appended.

The merge_comps routine is called for each piece of the model in order to
add the needed facets in the dividing plane and then write to disk.  If the
dividing plane would split the model into MORE than two pieces, merge_comps
will quit with a warning that one subset of the model (i.e., the piece on
one particular side of the dividing plane) may itself be broken into two or
more spatially disjoint pieces.

At the other extreme, if the dividing plane does not intersect the input
model at all, split_mod will quit with an error message before calling
merge_comps.

Modified 2010 September 1 by CM:
    Initialize variables to avoid compilation warnings

Modified 2010 June 1 by CM:
    Change "scalefactor" parameter from a scalar to a 3-component vector

Written 2010 May 21 by CM
***************************************************************************/

#include "head.h"

#define HAIRWIDTH 1.0e-7


void split_mod( struct par_t *par, struct mod_t *mod)
{
  int ncomp, nv, v, j, nv2, nf, nf2, ns, s, va, vb, f, v2, f2, ns2, vc, vd, fa, fb,
      s2, k, sa, sb, sc, sd;
  int *s_intersects_plane, **s_v, **s_f, *act2;
  double interp, x[3];
  double *r_perp, *r2_perp;
  struct mod_t mod2;
  struct vertices_t *real, *real2;

  /* Initialize variables to avoid compilation warnings  */
  sa = sc = vc = vd = 0;

  /* Exit if there's more than one component  */
  fflush(stdout);
  ncomp = mod->shape.ncomp;
  if (ncomp != 1)
    bailout("split_mod.c: input model must have only one component\n");

  /* Compute how far each vertex is from dividing plane, then avoid tiny model
   * sides (once we've added interpolated vertices in the plane) by shifting
   * vertices very close to the plane until they lie exactly in the plane. Note
   * that r_perp[v], the distance from the dividing plane to vertex v in the
   * direction along the normal to that plane, can be either positive or negative   */
  real = &mod->shape.comp[0].real;
  nv = real->nv;
  r_perp = vector( 0, nv-1);
  for (v=0; v<nv; v++) {
    r_perp[v] = dot( par->split_norm, real->v[v].x) + par->split_const;
    if (fabs(r_perp[v]) < HAIRWIDTH) {
      for (j=0; j<=2; j++)
        real->v[v].x[j] -= (r_perp[v] * par->split_norm[j]);
      r_perp[v] = 0.0;
    }
  }

  /* Count how many vertices and facets model will have once we add vertices
   * along sides that intersect the dividing plane, and flag those sides  */
  nv2 = nv;
  nf = nf2 = real->nf;
  ns = real->ns;
  s_intersects_plane = ivector( 0, ns-1);
  for (s=0; s<ns; s++) {
    va = real->s[s].v[0];
    vb = real->s[s].v[1];
    if ((r_perp[va] < -HAIRWIDTH && r_perp[vb] > HAIRWIDTH) ||
        (r_perp[vb] < -HAIRWIDTH && r_perp[va] > HAIRWIDTH)    ) {

        nv2++;
        nf2 += 2;
        s_intersects_plane[s] = 1;
    } else {
        s_intersects_plane[s] = 0;
    }
  }

  /* Exit if the dividing plane doesn't intersect the model  */
  if (nv2 == nv)
    bailout("split_mod.c: dividing plane does not intersect model\n");

  /* Allocate memory for and begin to initialize the "extended" model's struc-
   * ture - the structure for the version of the input model that has inter-
   * polated vertices added in the dividing plane    */
  mod2.shape.ncomp = 1;
  mod2.shape.comp = (struct comp_t *)
                    calloc( mod2.shape.ncomp, sizeof( struct comp_t));
  for (j=0; j<=2; j++) {
    mod2.shape.comp[0].off[j].state = 'c';
    mod2.shape.comp[0].off[j].val = 0.0;
    mod2.shape.comp[0].rot[j].state = 'c';
    mod2.shape.comp[0].rot[j].val = 0.0;
  }
  mod2.shape.comp[0].type = VERTEX;

  for (j=0; j<=2; j++) {
    mod2.shape.comp[0].desc.ver.scalefactor[j].state
                     = mod->shape.comp[0].desc.ver.scalefactor[j].state;
    mod2.shape.comp[0].desc.ver.scalefactor[j].val = 1.0;
  }

  /* Allocate memory for the extended model's vertices, then copy properties of
   * all existing vertices to the extended model. For now, set each vertex's
   * direction cosines to point radially outward; later, when we call
   * merge_comps for each half of the model, they will be reset to point normal
   * to the surface (i.e., along vertex normals).    */
  mod2.shape.comp[0].desc.ver.nv = nv2;
  mod2.shape.comp[0].desc.ver.v = (struct vertex_t *)
                                  calloc( nv2, sizeof( struct vertex_t));
  for (v=0; v<nv; v++) {
    mod2.shape.comp[0].desc.ver.v[v].r.state = 'f';
    mod2.shape.comp[0].desc.ver.v[v].r.val = 0.0;
    for (j=0; j<=2; j++) {
      mod2.shape.comp[0].desc.ver.v[v].u[j] = real->v[v].x[j];
      mod2.shape.comp[0].desc.ver.v[v].a[j] = real->v[v].x[j];
    }
    normalize( mod2.shape.comp[0].desc.ver.v[v].u);
  }

  /* Allocate memory for extended model's facets, then copy properties of all
   * existing facets to the extended model         */
  mod2.shape.comp[0].desc.ver.nf = nf2;
  mod2.shape.comp[0].desc.ver.f = (struct facet_t *)
                                  calloc( nf2, sizeof( struct facet_t));
  for (f=0; f<nf; f++) {
    for (j=0; j<=2; j++)
      mod2.shape.comp[0].desc.ver.f[f].v[j] = real->f[f].v[j];
    for (j=0; j<=2; j++)
      mod2.shape.comp[0].desc.ver.f[f].s[j] = real->f[f].s[j];
  }

  /* Create a vector that gives distance from dividing plane, measured along
   * the normal to the plane, to each vertex in the extended model; this dis-
   * tance can be either positive or negative. Copy the values previously com-
   * puted for existing vertices, then set the values for the (not yet created)
   * interpolated vertices to zero -- since they will lie exactly in plane.  */
  r2_perp = vector( 0, nv2-1);
  for (v2=0; v2<nv; v2++)
    r2_perp[v2] = r_perp[v2];
  for (v2=nv; v2<nv2; v2++)
    r2_perp[v2] = 0.0;

  /* Create matrices that list the vertices and facets attached to each side in
   * extended model - memory for this information hasn't yet been allocated for
   * extended model's "mod2" structure, since we haven't yet called the
   * setupsides routine - and copy the values for existing sides      */
  ns2 = 3*nv2 - 6;
  s_v = imatrix( 0, ns2-1, 0, 1);
  s_f = imatrix( 0, ns2-1, 0, 1);
  for (s=0; s<ns; s++) {
    for (j=0; j<=1; j++)
      s_v[s][j] = real->s[s].v[j];
    for (j=0; j<=1; j++)
      s_f[s][j] = real->s[s].f[j];
  }

  /* Initialize the number of vertices, facets, and sides
   * that have been defined so far for the extended model  */
  v2 = nv;
  f2 = nf;
  s2 = ns;

  /* Loop through all sides in the input model, look at those that intersect
   * the dividing plane, create new vertices that lie exactly in the plane,
   * and create new facets that contain those vertices                        */
  for (s=0; s<ns; s++)
    if (s_intersects_plane[s]) {

      /* This side intersects the dividing plane: subdivide it by
       * interpolating a new vertex that lies exactly in the plane  */
      va = real->s[s].v[0];
      vb = real->s[s].v[1];
      interp = -r_perp[va] / (r_perp[vb] - r_perp[va]);
      for (j=0; j<=2; j++)
        x[j] = real->v[va].x[j] + interp * (real->v[vb].x[j] - real->v[va].x[j]);

      /* Assign properties of the new vertex  */
      mod2.shape.comp[0].desc.ver.v[v2].r.state = 'f';
      mod2.shape.comp[0].desc.ver.v[v2].r.val = 0.0;
      for (j=0; j<=2; j++) {
        mod2.shape.comp[0].desc.ver.v[v2].u[j] = x[j];
        mod2.shape.comp[0].desc.ver.v[v2].a[j] = x[j];
      }
      normalize( mod2.shape.comp[0].desc.ver.v[v2].u);

      /* Identify the facets attached to the divided side, the third vertex
       * of each of those facets, and the other two sides of each facet      */
      fa = s_f[s][0];
      for (j=0; j<=2; j++)
        if (mod2.shape.comp[0].desc.ver.f[fa].v[j] != va &&
            mod2.shape.comp[0].desc.ver.f[fa].v[j] != vb    )
          vc = mod2.shape.comp[0].desc.ver.f[fa].v[j];
      for (j=0; j<=2; j++) {
        k = mod2.shape.comp[0].desc.ver.f[fa].s[j];
        if ((s_v[k][0] == vb && s_v[k][1] == vc) ||
            (s_v[k][1] == vb && s_v[k][0] == vc)    )
          sa = k;
        else if ((s_v[k][0] == vc && s_v[k][1] == va) ||
                 (s_v[k][1] == vc && s_v[k][0] == va)    )
          sb = k;
      }
      fb = s_f[s][1];
      for (j=0; j<=2; j++)
        if (mod2.shape.comp[0].desc.ver.f[fb].v[j] != va &&
            mod2.shape.comp[0].desc.ver.f[fb].v[j] != vb    )
          vd = mod2.shape.comp[0].desc.ver.f[fb].v[j];
      for (j=0; j<=2; j++) {
        k = mod2.shape.comp[0].desc.ver.f[fb].s[j];
        if ((s_v[k][0] == va && s_v[k][1] == vd) ||
            (s_v[k][1] == va && s_v[k][0] == vd)    )
          sc = k;
        else if ((s_v[k][0] == vd && s_v[k][1] == vb) ||
                 (s_v[k][1] == vd && s_v[k][0] == vb)    )
          sd = k;
      }

      /* Assign vertices and sides to 2 new facets containing the new vertex  */
      if ((mod2.shape.comp[0].desc.ver.f[fa].v[0] == va
                    && mod2.shape.comp[0].desc.ver.f[fa].v[1] == vb) ||
          (mod2.shape.comp[0].desc.ver.f[fa].v[1] == va
                    && mod2.shape.comp[0].desc.ver.f[fa].v[2] == vb) ||
          (mod2.shape.comp[0].desc.ver.f[fa].v[2] == va
                    && mod2.shape.comp[0].desc.ver.f[fa].v[0] == vb)    ) {

          mod2.shape.comp[0].desc.ver.f[f2].v[0] = v2;
          mod2.shape.comp[0].desc.ver.f[f2].v[1] = vb;
          mod2.shape.comp[0].desc.ver.f[f2].v[2] = vc;
          mod2.shape.comp[0].desc.ver.f[f2].s[0] = s2;
          mod2.shape.comp[0].desc.ver.f[f2].s[1] = sa;
          mod2.shape.comp[0].desc.ver.f[f2].s[2] = s2 + 1;
          mod2.shape.comp[0].desc.ver.f[f2+1].v[0] = v2;
          mod2.shape.comp[0].desc.ver.f[f2+1].v[1] = va;
          mod2.shape.comp[0].desc.ver.f[f2+1].v[2] = vd;
          mod2.shape.comp[0].desc.ver.f[f2+1].s[0] = s;
          mod2.shape.comp[0].desc.ver.f[f2+1].s[1] = sc;
          mod2.shape.comp[0].desc.ver.f[f2+1].s[2] = s2 + 2;
      } else {
          mod2.shape.comp[0].desc.ver.f[f2].v[0] = v2;
          mod2.shape.comp[0].desc.ver.f[f2].v[1] = vc;
          mod2.shape.comp[0].desc.ver.f[f2].v[2] = vb;
          mod2.shape.comp[0].desc.ver.f[f2].s[0] = s2 + 1;
          mod2.shape.comp[0].desc.ver.f[f2].s[1] = sa;
          mod2.shape.comp[0].desc.ver.f[f2].s[2] = s2;
          mod2.shape.comp[0].desc.ver.f[f2+1].v[0] = v2;
          mod2.shape.comp[0].desc.ver.f[f2+1].v[1] = vd;
          mod2.shape.comp[0].desc.ver.f[f2+1].v[2] = va;
          mod2.shape.comp[0].desc.ver.f[f2+1].s[0] = s2 + 2;
          mod2.shape.comp[0].desc.ver.f[f2+1].s[1] = sc;
          mod2.shape.comp[0].desc.ver.f[f2+1].s[2] = s;
      }

      /* Update vertices and sides for the 2 existing (now shrunken) facets  */
      for (j=0; j<=2; j++)
        if (mod2.shape.comp[0].desc.ver.f[fa].v[j] == vb)
          mod2.shape.comp[0].desc.ver.f[fa].v[j] = v2;
      for (j=0; j<=2; j++)
        if (mod2.shape.comp[0].desc.ver.f[fa].s[j] == sa)
          mod2.shape.comp[0].desc.ver.f[fa].s[j] = s2 + 1;
      for (j=0; j<=2; j++)
        if (mod2.shape.comp[0].desc.ver.f[fb].v[j] == va)
          mod2.shape.comp[0].desc.ver.f[fb].v[j] = v2;
      for (j=0; j<=2; j++)
        if (mod2.shape.comp[0].desc.ver.f[fb].s[j] == s)
          mod2.shape.comp[0].desc.ver.f[fb].s[j] = s2;
        else if (mod2.shape.comp[0].desc.ver.f[fb].s[j] == sc)
          mod2.shape.comp[0].desc.ver.f[fb].s[j] = s2 + 2;

      /*  Update vertices and facets attached to 3 existing sides  */
      if (s_v[s][0] == vb)
        s_v[s][0] = v2;
      else
        s_v[s][1] = v2;
      if (s_f[s][0] == fb)
        s_f[s][0] = f2 + 1;
      else
        s_f[s][1] = f2 + 1;
      if (s_f[sa][0] == fa)
        s_f[sa][0] = f2;
      else
        s_f[sa][1] = f2;
      if (s_f[sc][0] == fb)
        s_f[sc][0] = f2 + 1;
      else
        s_f[sc][1] = f2 + 1;

      /* Assign attached vertices and facets to 3 new sides that have the new
       * vertex at one end                     */
      s_v[s2][0] = v2;
      s_v[s2][1] = vb;
      s_f[s2][0] = fb;
      s_f[s2][1] = f2;
      s_v[s2+1][0] = v2;
      s_v[s2+1][1] = vc;
      s_f[s2+1][0] = f2;
      s_f[s2+1][1] = fa;
      s_v[s2+2][0] = v2;
      s_v[s2+2][1] = vd;
      s_f[s2+2][0] = f2 + 1;
      s_f[s2+2][1] = fb;

      /* Update # of vertices,facets,sides defined so far for extended model */
      v2++;
      f2 += 2;
      s2 += 3;
    }

  /* Complete the extended model structure and then realize the model  */
  mod2.shape.comp[0].real = mod2.shape.comp[0].desc.ver;
  setupsides( &mod2.shape.comp[0].real);
  setupvertices( &mod2.shape.comp[0].real);
  realize_mod( par, &mod2);

  /* Assign photometric and spin portions of input model's structure to
   * extended model's structure    */
  mod2.photo = mod->photo;
  mod2.spin = mod->spin;

  /* Record which vertices of extended model are active (since we're about to
   * change that information)          */
  real2 = &mod2.shape.comp[0].real;
  act2 = ivector( 0, nv2-1);
  for (v2=0; v2<nv2; v2++)
    act2[v2] = real2->v[v2].act;

  /* Take extended model and flag all vertices on the "negative" side of
   * dividing plane - on the side opposite the normal to the plane - as in-
   * active. Then send extended model to merge_comps routine to turn the
   * "positive" side of model (incl.vertices lying in the plane) into a topo-
   * logically closed model; done by adding facets in the plane as needed.
   * merge_comps writes this closed model to disk as mod file whose name is the
   * input model's name with ".split1" appended.    */
  for (v2=0; v2<nv2; v2++)
    if (r2_perp[v2] < 0.0)
      real2->v[v2].act = 0;
  sprintf(mod2.name, "%s.split1", mod->name);
  printf("#\n");
  printf("# closing the model subset on the positive side of the dividing plane...\n");
  fflush(stdout);
  merge_comps( par, &mod2);

  /* Reverse above process: produce a closed model for the "negative" side of
   * dividing plane (incl. plane itself) and write it to disk as mod file whose
   * name is the input model's name with ".split2" appended     */
  for (v2=0; v2<nv2; v2++)
    if (r2_perp[v2] > 0.0)
      real2->v[v2].act = 0;
    else
      real2->v[v2].act = act2[v2];
  sprintf(mod2.name, "%s.split2", mod->name);
  printf("#\n");
  printf("# closing the model subset on the negative side of the dividing plane...\n");
  fflush(stdout);
  merge_comps( par, &mod2);

  /* Close up shop  */
  free_vector( r_perp, 0, nv-1);
  free_ivector( s_intersects_plane, 0, ns-1);
  free_vector( r2_perp, 0, nv2-1);
  free_imatrix( s_v, 0, ns2-1, 0, 1);
  free_imatrix( s_f, 0, ns2-1, 0, 1);
  free_ivector( act2, 0, nv2-1);
}

#undef HAIRWIDTH
