/***************************************************************************
                                                             convex_hull.c

Takes a struct mod_t model which has already been realized as a polyhedral
vertex model by routine realize_mod, and outputs a new mod file describing
the model's 3D convex hull.

Note that the original model needn't be a vertex model, since convex_hull
deals only with the vertex realization of this model.

Modified 2010 October 22 by CM:
    Get vertex deviations relative to the reference ellipsoid by exactly
        solving a sixth-degree polynomial equation rather than by using
        Powell's method to search for the shortest distance to the
        ellipsoid: see header comments to wf2fac.c for more information

Modified 2010 June 1 by CM:
    Change "scalefactor" parameter from a scalar to a 3-component vector

Modified 2009 July 21 by CM:
    Remove initialization of facet "act" flags from hull_moments routine,
        since realize_mod now does this

Modified 2007 August 10 by CM:
    Eliminate unused variables

Modified 2007 January 8 by PT and CM:
    Fix bug in preceding change (scalefactor).  Instead of trying to get
        vertex deviations, base displacements, and direction cosines
        directly from the input mod file -- an approach that doesn't work
        for multiple-component models with different linear offsets,
        rotational offsets, and scale factors -- use code stolen from
        standalone program wf2fac.c to compute the hull's best-fit
        ellipsoid and determine the vertex deviations and so on.

Modified 2006 October 1 by CM:
    Add "scalefactor" to vertex shape structure

Modified 2005 January 25 by CM:
    Take care of uninitialized variables to avoid compilation warnings

Modified 2004 June 6 by CM:
    Added code to handle roundoff problems due to near-coplanar vertices

Written 2004 March 7 by CM, based on the algorithm described in
    Appendix B of Kaasalainen & Torppa 2001 (Icarus 153, 24-36), but
    omitting the step in which coplanar hull vertices are eliminated.
    (This step isn't really necessary for our purposes; furthermore,
    when I tried writing the relevant code anyway, with some small
    angular tolerance used for judging coplanarity, I ran into problems
    with hull vertices which were considered coplanar/redundant from
    one direction but not from a different direction.)
***************************************************************************/

#include "head.h"

#define TINY_DOTPROD 1.0e-15

void hull_moments( struct par_t *par, struct mod_t *hull);
void best_ellipse(void);
void ellipse_points(void);
double ellipse_root( fcomplex poly6coeffs[7]);

static double **x, **a, **u, *r, k[3];
static int n_hull_verts;

void convex_hull( struct par_t *par, struct mod_t *mod)
{
  int c, nv, nf, i, j, k, n, i_a, i_b, i_c, i_d, n_a, n_b, n_c,
      n_hull_facets, k_b, k_c, last_coplanar_i_b, bad_connection,
      n_new_hull_verts, restart;
  int *mod_comp, *mod_vert, *hull_vert, **is_connected, **forceconnect,
      *mod1_vert, **connect_verts, *first_connection, *n_connections,
      **normal_sense;
  double zmax, norm_0[3], norm_b[3], cosangle, cosanglemin, r_ab[3],
         r_ac[3], r_ad[3], cross_acb[3], dot_d_acb, r_bc[3],
         cross_abc[3], last_dot_d_acb;
  double **vcoord;
  struct mod_t hull;

  printf("# finding convex hull vertices\n");
  fflush(stdout);

  /* Initialize variables to avoid compilation warnings  */
  i_a = i_b = 0;

  /* Define several important structures, scalars, vectors, and matrices:
   *
   *  mod          = vertex realization of original model
   *  mod1         = realization of original model but with all vertices
   *  				 from all components placed in a single list
   *  hull         = convex hull model
   *  nv           = total number of vertices in all components of mod
   *                 (which also is the number of vertices in mod1)
   *  n_hull_verts = number of vertices in hull
   *  vcoord[i][j] = jth coordinate of mod1 vertex i (1<=i<=nv  and  1<=j<=3)
   *  mod_comp[i]  = mod component number (0-based) for mod1 vertex i
   *  mod_vert[i]  = mod vertex number (0-based) within mod component
   *                 mod_comp[i] for mod1 vertex i
   *  hull_vert[i] = hull vertex number for mod1 vertex i (if this vertex in
   *  	  	  	  	 fact lies on the convex hull, else 0)
   *  is_connected[i][i'] = 1 if mod1 vertices i and i' lie on convex hull and
   *  				 are connected to each other; 0 if they are not connected
   *  				 to each other; -1 if status has not yet been determined
   *  				            (1 <= i,i' <= nv)
   *  forceconnect[i][i'] = 1 if mod1 vertices i and i' have explicitly been
   *  				 connected to each other to avoid ambiguities due to near-
   *  				 coplanar vertices on convex hull; 0 if instead they have
   *  				 explicitly been disconnected; -1 otherwise
   *  mod1_vert[k] = mod1 vertex number of hull vertex k (1<=k<=n_hull_verts)
   *  connect_verts[k][n] = hull vertex number of nth hull vertex connected to
   *  				 hull vertex k (1 <= n <= n_connections[k])
   *  first_connection[k] = mod1 vertex number of first vertex used as a
   *                 connection (base line) to hull vertex k
   *  n_connections[k]    = number of hull vertices connected to hull vertex k
   *  normal_sense[k][k'] = 1 if *second* facet attached to the side (edge)
   *  				 between hull vertices k and k' should have its normal de-
   *  				 fined in the sense such that we travel from k to k'; 0 if
   *  				 instead from k' to k.  It follows that
   *  				 	normal_sense[k][k'] != normal_sense[k'][k].
   *  				 	       (1 <= k,k' <= n_hull_verts) */

  nv = 0;
  for (c=0; c<mod->shape.ncomp; c++)
    nv += mod->shape.comp[c].real.nv;

  n_hull_verts = 0;

  vcoord = matrix( 1, nv, 1, 3);
  mod_comp = ivector( 1, nv);
  mod_vert = ivector( 1, nv);
  hull_vert = ivector( 1, nv);
  is_connected = imatrix( 1, nv, 1, nv);
  forceconnect = imatrix( 1, nv, 1, nv);
  mod1_vert = ivector( 1, nv);
  connect_verts = imatrix( 1, nv, 1, nv);
  first_connection = ivector( 1, nv);
  n_connections = ivector( 1, nv);
  normal_sense = imatrix( 1, nv, 1, nv);

  for (i=1; i<=nv; i++) {
    hull_vert[i] = 0;
    mod1_vert[i] = 0;
    first_connection[i] = 0;
    for (k=1; k<=nv; k++) {
      is_connected[i][k] = -1;
      forceconnect[i][k] = -1;
    }
  }

  /* Put all vertex coordinates for all model components into a single array */
  i = 0;
  for (c=0; c<mod->shape.ncomp; c++) {
    for (k=0; k<mod->shape.comp[c].real.nv; k++) {
      i++;
      for (j=1; j<=3; j++)
        vcoord[i][j] = mod->shape.comp[c].real.v[k].x[j-1];
      mod_comp[i] = c;
      mod_vert[i] = k;
    }
  }

  /* Initialize n_a, the 1-based number within hull of vertex a, the current
   * "center" vertex. For each center vertex we will search for all neighboring
   * vertices b which also lie on the convex hull and which are directly
   * connected to vertex a.                     */
  n_a = 0;

  /* Find 1st vertex on convex hull: the one with largest z coordinate.  This
   * will be our first center (vertex a).  */
  zmax = -1.0e20;
  for (i=1; i<=nv; i++) {
    if (vcoord[i][3] > zmax) {
      zmax = vcoord[i][3];
      i_a = i;
    }
  }
  n_hull_verts++;
  hull_vert[i_a] = n_hull_verts;
  mod1_vert[n_hull_verts] = i_a;

  /* Find 1st base line vertex b: vertex which is the greatest angular distance
   * from origin as viewed from center vertex a. Vertex b is also on the convex
   * hull, and is directly connected to vertex a.
   *       i_b is the 1-based number within mod1 of vertex b.          */
  for (j=1; j<=3; j++)
    norm_0[j-1] = -vcoord[i_a][j];
  normalize( norm_0);
  cosanglemin = 1.0e20;
  for (i=1; i<=nv; i++) {
    if (i != i_a) {
      for (j=1; j<=3; j++)
        norm_b[j-1] = vcoord[i][j] - vcoord[i_a][j];
      normalize( norm_b);
      cosangle = dot( norm_0, norm_b);
      if (cosangle < cosanglemin) {
        cosanglemin = cosangle;
        i_b = i;
      }
    }
  }
  first_connection[1] = i_b;

  /* Loop through all center vertices a known to lie on the convex hull,
   * searching for all other vertices b which are on the hull and which are
   * connected to vertex a.           */
  do {

    /* Increment n_a, the 1-based number within hull of center vertex a;
     * also update i_a, the 1-based number within mod1 of vertex a         */
    i_a = mod1_vert[++n_a];
    if (n_a % 100 == 1) {
      printf("#     -- working on hull vertex %d\n", n_a-1);
      fflush(stdout);
    }

    /* Initialize number of base line vertices b found for center vertex a    */
    n_connections[n_a] = 0;

    /* As 1st base line vertex b for new center vertex a, we can choose ANY
     * vertex which lies on the convex hull and is directly connected to new
     * center - so choose the vertex which was the center back when the current
     * center was 1st id'd as a connected neighbor and was added to hull.*/
    i_b = first_connection[n_a];
    n_new_hull_verts = 0;
    last_coplanar_i_b = 0;

    /* Loop through for all possible base lines - all vertices b which lie on
     * the convex hull and are connected to center vertex a.                 */
    do {

      /* Add vertex b to convex hull if it's not already there, and add it to
       * the list of connections to center vertex a.   */
      is_connected[i_a][i_b] = 1;
      if (hull_vert[i_b] == 0) {
        hull_vert[i_b] = ++n_hull_verts;
        mod1_vert[n_hull_verts] = i_b;
        first_connection[n_hull_verts] = i_a;
        n_new_hull_verts++;
      }
      connect_verts[n_a][++n_connections[n_a]] = hull_vert[i_b];

      /* Get the base line: the displacement vector from a to b  */
      for (j=1; j<=3; j++)
        r_ab[j-1] = vcoord[i_b][j] - vcoord[i_a][j];

      /* Find a reference vertex c which will form a plane with a and b  */
      i_c = 1;
      while (i_c == i_a || i_c == i_b
                        || is_connected[i_c][i_a] == 0 || is_connected[i_a][i_c] == 0)
        i_c++;

      /* Get displacement vector from a to c and its cross product with r_ab
       * (proportional to the normal to plane acb)            */
      for (j=1; j<=3; j++)
        r_ac[j-1] = vcoord[i_c][j] - vcoord[i_a][j];
      cross( cross_acb, r_ac, r_ab);

      /* Check whether other vertices d are above plane abc (i.e., are dis-
       * placed from a along the positive normal to this plane); if so, vertex
       *  d becomes the new vertex c     */
      i_d = i_c + 1;
      last_dot_d_acb = 1.0e20;
      while (i_d <= nv) {
        if (i_d != i_a && i_d != i_b && is_connected[i_a][i_d] != 0
                                     && is_connected[i_d][i_a] != 0) {
          for (j=1; j<=3; j++)
            r_ad[j-1] = vcoord[i_d][j] - vcoord[i_a][j];
          dot_d_acb = dot( r_ad, cross_acb);

          /* Due to roundoff error (near-coplanar vertices) we can't just test
           * "if (dot_d_acb > 0.0)"       */
          if (dot_d_acb >= TINY_DOTPROD ||
              (dot_d_acb > -TINY_DOTPROD && forceconnect[i_a][i_c] != 1 &&
               (forceconnect[i_a][i_d] == 1 || is_connected[i_c][i_a] != 1) )) {
            last_dot_d_acb = dot_d_acb;
            i_c = i_d;
            for (j=0; j<=2; j++)
              r_ac[j] = r_ad[j];
            cross( cross_acb, r_ac, r_ab);
          }
        }
        i_d++;
      }

      /* All vertices have been checked: vertex c is on the convex hull.
       * Use it as the new vertex b.  */
      i_b = i_c;

      /* If choice between this vertex and the next best one was down in the
       * roundoff noise (near-coplanar vertices), save its mod1 index in case
       * we have to disconnect it later on           */
      if (fabs(last_dot_d_acb) < TINY_DOTPROD)
        last_coplanar_i_b = i_b;

      /* In the absence of roundoff error, we shouldn't repeat any vertex b
       * until we arrive back at the 1st one. If this isn't the case, try to
       * fix the problem by disconnecting a near-coplanar baseline vertex which
       * was assigned earlier. If this near-coplanar vertex has not yet been
       * used as center vertex a, we can just erase the connections to current
       * center vertex a and continue from there; otherwise we must restart the
       * hull vertex search from scratch.                 */
      bad_connection = (is_connected[i_a][i_b] == 1) && (i_b != first_connection[n_a]);
      if (bad_connection) {
        if (last_coplanar_i_b == 0)
          bailout("infinite vertex loop in convex_hull.c\n");
        restart = (is_connected[last_coplanar_i_b][i_a] == 1);
        if (restart) {

            /* Restart vertex search from scratch (except that we know which
             * mod1 vertices will be hull vertices 1 and 2)       */
            if (forceconnect[i_a][last_coplanar_i_b] == -1
                    && forceconnect[last_coplanar_i_b][i_a] == -1) {
                forceconnect[i_a][last_coplanar_i_b] = forceconnect[last_coplanar_i_b][i_a] = 0;
                printf("# restarting at hull vertex %d\n", n_a-1);
                fflush(stdout);
            } else {
                bailout("infinite vertex loop in convex_hull.c\n");
            }
            i_a = mod1_vert[1];
            i_b = first_connection[1];
            for (i=1; i<=nv; i++) {
              hull_vert[i] = 0;
              mod1_vert[i] = 0;
              first_connection[i] = 0;
              for (k=1; k<=nv; k++)
                is_connected[i][k] = (forceconnect[i][k] == 0) ? 0 : -1;
            }
            n_hull_verts = 1;
            hull_vert[i_a] = 1;
            mod1_vert[1] = i_a;
            first_connection[1] = i_b;
            n_a = 1;
            n_connections[1] = 0;
        } else {

            /* Just redo the center vertex a which we're currently working on  */
            is_connected[i_a][last_coplanar_i_b] = is_connected[last_coplanar_i_b][i_a] = 0;
            for (n=1; n<=n_connections[n_a]; n++) {
              i_b = mod1_vert[connect_verts[n_a][n]];
              is_connected[i_a][i_b] = -1;
            }
            n_connections[n_a] = 0;
            for (n_b=n_hull_verts-n_new_hull_verts+1; n_b<=n_hull_verts; n_b++) {
              i_b = mod1_vert[n_b];
              mod1_vert[n_b] = 0;
              hull_vert[i_b] = 0;
              first_connection[n_b] = 0;
            }
            n_hull_verts -= n_new_hull_verts;
            i_b = first_connection[n_a];
        }
        n_new_hull_verts = 0;
        last_coplanar_i_b = 0;
      }

      /* If new vertex b is NOT the same as 1st base line vertex b, use it as
       * a new base line, and loop back to find a new vertex c on convex hull*/
    } while (i_b != first_connection[n_a] || bad_connection);

    /* We have found all base line vertices b which lie on the convex hull and
     * are directly connected to center vertex a. Mark all other mod1 vertices
     * as not connected to vertex a.               */
    for (i_b=1; i_b<=nv; i_b++)
      if (is_connected[i_a][i_b] == -1)
        is_connected[i_a][i_b] = 0;

    /* Check all connections to vertex a: In the absence of roundoff error, we
     * should never find that two adjacent connecting vertices b are definitely
     * NOT connected to each other, because this implies an incomplete facet.
     * If we encounter such a problem, due to near-coplanar vertices, we must
     * explicitly connect the two unconnected vertices and then restart the
     * hull vertex search from scratch.                        */
    k_b = 1;
    do {
      i_b = mod1_vert[connect_verts[n_a][k_b]];
      k_c = (k_b % n_connections[n_a]) + 1;
      i_c = mod1_vert[connect_verts[n_a][k_c]];
      bad_connection = (is_connected[i_b][i_a] == 1 && is_connected[i_c][i_a] == 1 &&
                        is_connected[i_b][i_c] == 0 && is_connected[i_c][i_b] == 0    );
      if (bad_connection) {
        if (forceconnect[i_b][i_c] == -1 && forceconnect[i_c][i_b] == -1) {
            forceconnect[i_b][i_c] = forceconnect[i_c][i_b] = 1;
            printf("# found incomplete facet: mod vertices %d %d %d\n",
                   i_a-1, i_b-1, i_c-1);
            printf("# connecting mod vertex %d to %d\n", i_b-1, i_c-1);
            printf("# restarting at hull vertex %d\n", n_a-1);
            fflush(stdout);
        } else {
            printf("# incomplete facet can't be fixed: mod vertices %d %d %d\n",
                   i_a-1, i_b-1, i_c-1);
            fflush(stdout);
            bailout("incomplete facet in convex_hull.c\n");
        }
        i_a = mod1_vert[1];
        i_b = first_connection[1];
        for (i=1; i<=nv; i++) {
          hull_vert[i] = 0;
          mod1_vert[i] = 0;
          first_connection[i] = 0;
          for (k=1; k<=nv; k++)
            is_connected[i][k] = (forceconnect[i][k] == 0) ? 0 : -1;
        }
        n_hull_verts = 1;
        hull_vert[i_a] = 1;
        mod1_vert[1] = i_a;
        first_connection[1] = i_b;
        n_a = 0;
      }
      k_b++;
    } while (k_b <= n_connections[n_a] && !bad_connection);

    /* If there is any other vertex which is known to lie on convex hull and
     * which has NOT yet been used as a center, make that the new vertex a
     * and loop back to find all base line vertices b which are directly
     * connected to it.                 */
  } while (n_a < n_hull_verts);

  /* Found all hull vertices: Mark any mod1 vertices which were never looked
   * at as being disconnected from the hull         */
  for (i=1; i<=nv; i++)
    if (hull_vert[i] == 0)
      for (k=1; k<=nv; k++)
        is_connected[i][k] = 0;

  /* Check that if mod1 vertex i is (un)connected to mod1 vertex k then k is
   * also (un)connected to i                              */
  for (i=1; i<nv; i++)
    for (k=i+1; k<=nv; k++)
      if (is_connected[i][k] != is_connected[k][i]) {
        if (is_connected[i][k] == 1)
          printf("# mod vertex %d is connected to %d but not vice versa\n", i-1, k-1);
        else
          printf("# mod vertex %d is connected to %d but not vice versa\n", k-1, i-1);
        fflush(stdout);
        bailout("one-way connection in convex_hull.c\n");
      }

  /* Display the number of hull vertices and hull facets  */
  n_hull_facets = 2*n_hull_verts - 4;
  printf("# convex hull has %d vertices and %d facets\n", n_hull_verts, n_hull_facets);
  fflush(stdout);

  /* Start to fill in the convex hull model structure  */
  printf("# creating convex hull model structure\n");
  fflush(stdout);

  strcpy( hull.name, par->convex_file);
  hull.shape.ncomp = 1;
  hull.shape.comp = (struct comp_t *)
                    calloc( hull.shape.ncomp, sizeof( struct comp_t));
  for (j=0; j<=2; j++) {
    hull.shape.comp[0].off[j].state = 'c';
    hull.shape.comp[0].off[j].val = 0.0;
    hull.shape.comp[0].rot[j].state = 'c';
    hull.shape.comp[0].rot[j].val = 0.0;
  }
  hull.shape.comp[0].type = VERTEX;
  hull.shape.comp[0].desc.ver.nv = n_hull_verts;
  hull.shape.comp[0].desc.ver.v = (struct vertex_t *)
                                  calloc( n_hull_verts, sizeof( struct vertex_t));
  for (j=0; j<=2; j++) {
    hull.shape.comp[0].desc.ver.scalefactor[j].state = (j == 0) ? 'c' : '=';
    for (c=0; c<mod->shape.ncomp; c++)
      if (mod->shape.comp[c].real.scalefactor[j].state == 'f')
        hull.shape.comp[0].desc.ver.scalefactor[j].state = 'f';
    hull.shape.comp[0].desc.ver.scalefactor[j].val = 1.0;
  }

  /* Use code stolen from standalone program wf2fac.c to compute the hull's
   * best-fit ellipsoid & to use that ellipsoid to define a base displacement,
   * direction cosines, and vertex deviation for each hull vertex          */
  printf("# defining convex hull vertices w.r.t. best-fit ellipsoid\n");
  fflush(stdout);

  x = matrix( 0, n_hull_verts-1, 0, 2);     /* vertex coordinates */
  a = matrix( 0, n_hull_verts-1, 0, 2);     /* base displacements */
  u = matrix( 0, n_hull_verts-1, 0, 2);     /* direction cosines  */
  r = vector( 0, n_hull_verts-1);           /* vertex deviations  */
  for (n_a=1; n_a<=n_hull_verts; n_a++) {
    i_a = mod1_vert[n_a];
    for (j=1; j<=3; j++)
      x[n_a-1][j-1] = vcoord[i_a][j];
  }
  best_ellipse();
  ellipse_points();
  for (n_a=1; n_a<=n_hull_verts; n_a++) {
    i_a = mod1_vert[n_a];
    c = mod_comp[i_a];
    k = mod_vert[i_a];
    hull.shape.comp[0].desc.ver.v[n_a-1].r.state = mod->shape.comp[c].real.v[k].r.state;
    hull.shape.comp[0].desc.ver.v[n_a-1].r.val = r[n_a-1];
    for (j=0; j<=2; j++) {
      hull.shape.comp[0].desc.ver.v[n_a-1].u[j] = u[n_a-1][j];
      hull.shape.comp[0].desc.ver.v[n_a-1].a[j] = a[n_a-1][j];
    }
  }
  free_matrix( x, 0, n_hull_verts-1, 0, 2);
  free_matrix( a, 0, n_hull_verts-1, 0, 2);
  free_matrix( u, 0, n_hull_verts-1, 0, 2);
  free_vector( r, 0, n_hull_verts-1);

  /* Start identifying hull facets, assigning 3 vertices (corners) and a normal
   * vector to each facet. The normal is defined in right-hand sense of travel
   * from vertex 0 to 1 to 2.
   * 1st, find a facet which is attached to the 1st (closest) hull vertex and
   * which is not viewed edge-on, then assign the normal to this facet so that
   * it points towards us (convex).                  */
  printf("# identifying convex hull facets\n");
  fflush(stdout);

  hull.shape.comp[0].desc.ver.nf = n_hull_facets;
  hull.shape.comp[0].desc.ver.f = (struct facet_t *)
                                  calloc( n_hull_facets, sizeof( struct facet_t));

  i_a = mod1_vert[1];
  do {
    n_b = connect_verts[1][1];
    i_b = mod1_vert[n_b];
    n_c = connect_verts[1][2];
    i_c = mod1_vert[n_c];
    for (j=1; j<=3; j++) {
      r_ab[j-1] = vcoord[i_b][j] - vcoord[i_a][j];
      r_bc[j-1] = vcoord[i_c][j] - vcoord[i_b][j];
    }
    cross( cross_abc, r_ab, r_bc);
    normalize( cross_abc);
    if (cross_abc[2] > 0) {
        hull.shape.comp[0].desc.ver.f[0].v[0] = 0;
        hull.shape.comp[0].desc.ver.f[0].v[1] = 1;
        hull.shape.comp[0].desc.ver.f[0].v[2] = 2;
        for (j=0; j<=2; j++)
          hull.shape.comp[0].desc.ver.f[0].n[j] = cross_abc[j];
        normal_sense[1][n_b] = normal_sense[n_b][n_c] = normal_sense[n_c][1] = 0;
        normal_sense[n_b][1] = normal_sense[n_c][n_b] = normal_sense[1][n_c] = 1;
    } else if (cross_abc[2] < 0) {
        hull.shape.comp[0].desc.ver.f[0].v[0] = 0;
        hull.shape.comp[0].desc.ver.f[0].v[1] = 2;
        hull.shape.comp[0].desc.ver.f[0].v[2] = 1;
        for (j=0; j<=2; j++)
          hull.shape.comp[0].desc.ver.f[0].n[j] = -cross_abc[j];
        normal_sense[1][n_b] = normal_sense[n_b][n_c] = normal_sense[n_c][1] = 1;
        normal_sense[n_b][1] = normal_sense[n_c][n_b] = normal_sense[1][n_c] = 0;
    } else {
        /* This facet is seen edge-on: Cyclically permute the connections to
         * the closest vertex so that we can use a different facet      */
        for (k_b=1; k_b<=n_connections[1]-1; k_b++)
          connect_verts[1][k_b] = connect_verts[1][k_b+1];
        connect_verts[1][n_connections[1]] = n_b;
    }
  } while (cross_abc[2] == 0);
  /* Now identify the rest of the facets: order each triple of facet vertices
   * in the same sense as was done for the adjacent facet previously identified
   * - so that you travel along the the common side in the opposite direction
   * as was done for the adjacent facet.                               */
  nf = 0;
  for (n_a=1; n_a<=n_hull_verts; n_a++) {
    i_a = mod1_vert[n_a];
    for (k_b=1; k_b<=n_connections[n_a]; k_b++) {
      n_b = connect_verts[n_a][k_b];
      i_b = mod1_vert[n_b];
      k_c = (k_b % n_connections[n_a]) + 1;
      n_c = connect_verts[n_a][k_c];
      if (n_b > n_a && n_c > n_a) {
        if (nf > 0) {
          i_c = mod1_vert[n_c];
          for (j=1; j<=3; j++) {
            r_ab[j-1] = vcoord[i_b][j] - vcoord[i_a][j];
            r_bc[j-1] = vcoord[i_c][j] - vcoord[i_b][j];
          }
          cross( cross_abc, r_ab, r_bc);
          normalize( cross_abc);
          if (normal_sense[n_a][n_b]) {
              hull.shape.comp[0].desc.ver.f[nf].v[0] = n_a - 1;
              hull.shape.comp[0].desc.ver.f[nf].v[1] = n_b - 1;
              hull.shape.comp[0].desc.ver.f[nf].v[2] = n_c - 1;
              for (j=0; j<=2; j++)
                hull.shape.comp[0].desc.ver.f[nf].n[j] = cross_abc[j];
              normal_sense[n_b][n_c] = normal_sense[n_c][n_a] = 0;
              normal_sense[n_c][n_b] = normal_sense[n_a][n_c] = 1;
          } else {
              hull.shape.comp[0].desc.ver.f[nf].v[0] = n_a - 1;
              hull.shape.comp[0].desc.ver.f[nf].v[1] = n_c - 1;
              hull.shape.comp[0].desc.ver.f[nf].v[2] = n_b - 1;
              for (j=0; j<=2; j++)
                hull.shape.comp[0].desc.ver.f[nf].n[j] = -cross_abc[j];
              normal_sense[n_b][n_c] = normal_sense[n_c][n_a] = 1;
              normal_sense[n_c][n_b] = normal_sense[n_a][n_c] = 0;
          }
        }
        nf++;
      }
    }
  }

  if (nf != n_hull_facets) {
    printf("ERROR: Should have %d hull facets but found %d instead\n",
           n_hull_facets, nf);
    bailout("convex_hull.c\n");
  }
  printf("# found all facets for convex hull model\n");
  fflush(stdout);

  /* Assign spin and and photometric portions of original model structure to
   * the new convex hull structure              */
  printf("# assigning spin and photometric hull parameters\n");
  fflush(stdout);
  hull.spin = mod->spin;
  hull.photo = mod->photo;

  /* Complete the hull structure and get the hull's moments  */
  printf("# computing hull's moments\n");
  fflush(stdout);
  hull_moments( par, &hull);
  printf("# finished computing moments\n");
  fflush(stdout);

  /* Write the hull model to disk  */
  write_mod( par, &hull);

  /* Clean up storage space  */
  free_matrix( vcoord, 1, nv, 1, 3);
  free_ivector( mod_comp, 1, nv);
  free_ivector( mod_vert, 1, nv);
  free_ivector( hull_vert, 1, nv);
  free_imatrix( is_connected, 1, nv, 1, nv);
  free_imatrix( forceconnect, 1, nv, 1, nv);
  free_ivector( mod1_vert, 1, nv);
  free_imatrix( connect_verts, 1, nv, 1, nv);
  free_ivector( first_connection, 1, nv);
  free_ivector( n_connections, 1, nv);
  free_imatrix( normal_sense, 1, nv, 1, nv);
}


void hull_moments( struct par_t *par, struct mod_t *hull)
{

  /* Finish setting up convex hull model realization, steps normally carried
   * out within the read_mod routine.  Then call realize_mod to compute hull's
   * 0th, 1st, and 2nd-order moments.         */
  hull->shape.comp[0].real = hull->shape.comp[0].desc.ver;
  setupsides( &hull->shape.comp[0].real);
  setupvertices( &hull->shape.comp[0].real);
  realize_mod( par, hull);
}


void best_ellipse(void)
{
  double **m, *b, x2[4], d;
  int i, j, n, *indx;

  m = matrix( 1, 3, 1, 3);
  b = vector( 1, 3);
  indx = ivector( 1, 3);

  for (i=1; i<=3; i++) {
    for (j=1; j<=3; j++)
      m[i][j] = 0.0;
    b[i] = 0.0;
  }

  for (n=0; n<n_hull_verts; n++) {
    for (i=1; i<=3; i++)
      b[i] += (x2[i] = x[n][i-1]*x[n][i-1]);
    for (i=1; i<=3; i++)
      for (j=1; j<=3; j++)
        m[i][j] += x2[i]*x2[j];
  }

  ludcmp( m, 3, indx, &d);
  lubksb( m, 3, indx, b);
  for (i=0; i<=2; i++)
    k[i] = b[i+1];
  printf("# base ellipsoid diameters = %f %f %f km\n",
         2/sqrt(k[0]), 2/sqrt(k[1]), 2/sqrt(k[2]));
}


void ellipse_points(void)
{
  int i;
  double a2, b2, c2, a4, b4, c4, fact0_1, fact1_1, fact1_2, fact1_3, fact1_4,
         term1_1, fact2_1, fact2_2, fact2_3, term2_1, fact3_1, fact3_2, fact3_3,
         term3_1, term4_1, u0, v0, w0, alpha, nrmmag;
  fcomplex poly6coeffs[7];

  /* Compute semimajor axes squared and to the fourth power  */
  a2 = 1/k[0];
  b2 = 1/k[1];
  c2 = 1/k[2];
  a4 = a2*a2;
  b4 = b2*b2;
  c4 = c2*c2;

  /* Speed things up by computing some other factors and terms in advance  */

  fact0_1 = a4*b4*c4;
  fact1_1 = 2*a2*b2*c2;
  fact1_2 = a2*(b2 + c2);
  fact1_3 = b2*(a2 + c2);
  fact1_4 = c2*(a2 + b2);
  term1_1 = -(a2*b2 + a2*c2 + b2*c2);
  fact2_1 = (b4 + 4*b2*c2 + c4)*a4;
  fact2_2 = (a4 + 4*a2*c2 + c4)*b4;
  fact2_3 = (a4 + 4*a2*b2 + b4)*c4;
  term2_1 = -(a4*b4 + a4*c4 + b4*c4);
  fact3_1 = a4*(b2 + c2);
  fact3_2 = b4*(a2 + c2);
  fact3_3 = c4*(a2 + b2);
  term3_1 = 4*a2*b2*c2;
  term4_1 = 4*(a2*b2 + a2*c2 + b2*c2);

  /* Initialize complex coefficients for the 6th-degree polynomial whose roots
   * must be found to solve for shortest distance from each vertex to
   * ellipsoid; the 2 highest-order coefficients are same for all vertices   */
  poly6coeffs[0] = Complex( 0.0, 0.0);
  poly6coeffs[1] = Complex( 0.0, 0.0);
  poly6coeffs[2] = Complex( 0.0, 0.0);
  poly6coeffs[3] = Complex( 0.0, 0.0);
  poly6coeffs[4] = Complex( 0.0, 0.0);
  poly6coeffs[5] = Complex( 2*(a2 + b2 + c2), 0.0);
  poly6coeffs[6] = Complex( 1.0, 0.0);

  /* Loop through all vertices  */
  for (i=0; i<n_hull_verts; i++) {

    /* Compute normalized versions of squared vertex coordinates  */
    u0 = x[i][0]*x[i][0]/a2;
    v0 = x[i][1]*x[i][1]/b2;
    w0 = x[i][2]*x[i][2]/c2;

    /* Update the polynomial coefficients for this vertex  */
    poly6coeffs[0].r = (1 - u0 - v0 - w0)*fact0_1;
    poly6coeffs[1].r = fact1_1*((1 - u0)*fact1_2 + (1 - v0)*fact1_3
                                                 + (1 - w0)*fact1_4 + term1_1);
    poly6coeffs[2].r = (1 - u0)*fact2_1 + (1 - v0)*fact2_2
                                        + (1 - w0)*fact2_3 + term2_1;
    poly6coeffs[3].r = 2*((1 - u0)*fact3_1 + (1 - v0)*fact3_2
                                           + (1 - w0)*fact3_3 + term3_1);
    poly6coeffs[4].r = (1 - u0)*a4 + (1 - v0)*b4 + (1 - w0)*c4 + term4_1;

    /* Find the largest real root of the polynomial  */
    alpha = ellipse_root( poly6coeffs);

    /* Use this root to compute the point on the ellipsoid that is closest to
     * this vertex, the outward unit normal to ellipsoid at that point, and
     * the distance from that point to this vertex: these will be vertex's base
     * displacement, direction cosines, and vertex deviation      */
    a[i][0] = a2*x[i][0]/(alpha + a2);
    a[i][1] = b2*x[i][1]/(alpha + b2);
    a[i][2] = c2*x[i][2]/(alpha + c2);

    u[i][0] = a[i][0]/a2;
    u[i][1] = a[i][1]/b2;
    u[i][2] = a[i][2]/c2;
    nrmmag = normalize( u[i]);

    r[i] = alpha*nrmmag;
  }
}


/* Solve a 6th-degree polynomial equation and return largest real root: this
 * corresponds to the shortest distance from a given point to an ellipsoid  */

double ellipse_root( fcomplex poly6coeffs[7])
{
  int foundroot, j, k;
  double maxroot;
  fcomplex poly6roots[7];

  maxroot = -HUGENUMBER;

  zroots( poly6coeffs, 6, poly6roots, 1);
  foundroot = 0;
  for (j=1; j<=6; j++)
    if (poly6roots[j].i == 0.0) {
      maxroot = MAX( poly6roots[j].r, maxroot);
      foundroot = 1;
    }
  if (!foundroot) {
    fprintf( stderr, "Sixth-degree polynomial has no real roots for distance to ellipsoid:\n");
    for (k=1; k<=6; k++)
      fprintf( stderr, "    root %d: (Re, Im) = (%13.6e, %13.6e)\n",
               k, poly6roots[k].r, poly6roots[k].i);
    bailout("convex_hull.c\n");
  }
  return maxroot;
}

#undef TINY_DOTPROD
