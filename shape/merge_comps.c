/*****************************************************************************************
                                                                            merge_comps.c

Take a multiple-component input model that has already been realized, merge it into a
one-component vertex model, and write the merged model to disk as a wavefront .OBJ file.

All active vertices in the input model's vertex realization -- that is, all vertices that
lie on the model's surface -- are incorporated into the merged model.  Very small facets
are then eliminated from the model, so the final number of vertices may be smaller than
the number of active input vertices.

The facet list for the merged model starts with all facets in input component 1 that are
"intact" -- that is, that lie on the surface and are not crossed by component boundaries,
so that they have three active vertices at their corners.  They are listed in the same
order as they were listed in the input realization.  Then the intact facets from input
component 2 are listed, and so on.  The end of the list consists of facets that were not
part of the input model's realization but had to be constructed in the gaps near component
boundaries.

The vertex list for the merged model starts with all vertices that are part of the intact
facets of input component 1 (see above), listed in the same order that they were listed in
the input realization; then come the vertices that are part of the intact facets of input
component 2, and so on.  Last come the vertices (if any) that are not part of any intact
facet.  So, for example, if a two-component ellipsoid model is merged, the output vertices
will start at the northernmost active vertex of the former component 1 and spiral their
way toward its south pole, then jump to the northernmost active vertex of the former
component 2 and spiral their way toward its south pole, and then (if any vertices are
left) jump around randomly within the region along the former component boundary.

The code is quite lengthy due to the large amount of tedious bookkeeping that is required
-- keeping track of which facets are attached to which vertices, mapping vertex numbers in
the merged model to component numbers and vertex numbers in the input model realization,
etc.  Due to the many storage arrays that are needed to hold this information, the code
isn't at all modular: each routine would have to have all of these arrays passed as
arguments, or else the arrays would have to have global scope, thus defeating the purpose
of having modules.

Instead this file is broken into four large blocks, each one headed by a comment block:

    Initialization block: start off the merged model with the intact portions of the input
                          model's realization

    Merging block:        the hard part -- create new facets that join the intact portions
                          of the input realization, forming a closed polyhedral solid

    Optimization block:   Redefine facets in order to remove sharp facet boundaries and
                          very narrow facets, and eliminate very small facets

    Output block:         Create and realize a standard mod_t structure for the merged
                          model, then write the shape specification to disk as a
                          wavefront .OBJ file

Modified 2015 December 3 by CM:
    In the merging block, strengthen the geometric criterion that discourages the creation
        of an interior facet: when connecting a side to vertex v, if neither of the side's
        two end-vertices v1 and v2 is already connected to v, demand that neither v1 nor
        v2 comes from the same input component as v.  Previously the connection was
        allowed unless BOTH v1 and v2 came from the same input component as v.

Modified 2015 November 21 by CM:
    In the optimization block, add an extra check to ensure that in redefining a pair of
        facets we don't connect two vertices that already have a direct connection, and
        initialize the "f_very_small" vector so it's defined for all facets

Modified 2012 February 27 by CM:
    In the initialization block, remove sides from the initial merged model if they don't
        have any adjoining facets in the initial merged model; this can happen even if
        both end vertices belong to the initial merged model
    In the initialization block, compute and display the number of vertices in each
        contiguous subset of the initial merged model

Modified 2010 September 1 by CM:
    Initialize variables to avoid compilation warnings

Modified 2010 June 1 by CM:
    Change "scalefactor" parameter from a scalar to a 3-component
        vector

Modified 2010 May 20 by CM:
    Permit this routine to be called by the split action

Modified 2009 December 8 by CM:
    In the merger block, if the number of facets for the merged model is
        less than expected (given the number of vertices) by a multiple
        of four, give a warning (before quitting) that the input model
        may be broken into two or more spatially disjoint pieces

Modified 2009 November 6 by CM:
    In the merger block, don't allow a new facet to be created from three
        vertices from the same input component unless two of the three
        facet sides already exist; this discourages the algorithm from
        creating a facet INSIDE the model where a component boundary makes
        a hairpin turn and passes close to itself

Modified 2009 August 10 by CM:
    When redefining facets in the "optimization" block, check that the
        two vertices that would become connected to each other aren't
        already connected to each other
    Bug fix: in the merger block, when checking that a new side wouldn't
        leave any "stranded" facets, use the brute force approach of
        checking every permutation of every combination of attached
        vertices.  The "check_combinations" and "check_permutations"
        routines that do this are adapted from algorithms listed online at
        http://www.cs.utexas.edu/users/djimenez/utsa/cs3343/lecture25.html

Written 2009 August 2 by CM
***************************************************************************/

#include "head.h"

#define MAXATTACH 100

double facetgeom( double x1[3], double x2[3], double x3[3],
                  double normal[3], double *base, double *height, int *baseindex);
int stranded_facet( int n1, int n2, int *attachvec, int n_attached, int **v_connected);
int check_combinations( int v[], int start, int n, int k, int kmax,
                        int n1, int n2, int *attachvec, int **v_connected);
int check_permutations( int v[], int n, int i,
                        int n1, int n2, int *attachvec, int **v_connected);
void swap( int v[], int i, int j);


void merge_comps( struct par_t *par, struct mod_t *mod)
{
  int ncomp, nv_act, nv_tot, nf_tot, ns_tot, c, nv, nf, ns, v, n, keep_going,
      n_goodfacets, j, f, n1, n2, n3, n_goodsides, k, s, nv_merged, nf_merged, ns_merged,
      nv_in_model, nf_in_model, ns_in_model, m, s2, f2, v2, c2, nv2, n_subsets,
      nv_in_all_subsets, new_subset, nv_in_subset, m1, c1, v1, m2, n_finished_sides, cmin,
      vmin, valid_connection, f1, s1, v3, i, s1_is_new, s2_is_new, m3, s3, s4, w1, w2, w3,
      c_attached, v_attached, n_attached, c3, c4, m4, v4, baseindex1, baseindex2,
      baseindex3, baseindex4, redefine_facets, f3, f4, m5, m6, s5, s6, s7, s8, s9,
      nv_final, nf_final, af1, af2, af3, af4, n_missingfacets, n_pieces;
  int *nv_0, *nf_0, *ns_0, *v_in_model, *f_in_model, *s_in_model, *v_naf, **v_af, *v_nas,
      **v_as, **v_av, **f_v, **f_s, **s_v, *s_nf, **s_f, **v_old, *v_new, *f_new, *s_new,
      **v_connected, *attachvec, *subsetvec, *added_v_attached, *f_very_small, *v_good,
      *f_good, *s_good, *v_final, *f_final;
  double distmin, distsum, norm1[3], norm2[3], norm3[3], norm4[3], dot12, dot34, minheight,
         base1, base2, base3, base4, height1, height2, height3, height4;
  double **v_distance;
  struct mod_t mergedmod;
  struct vertices_t *real, *real2;

  /*  Initialize variables to avoid compilation warnings  */

  s1 = s2 = s3 = s4 = s5 = s6 = s7 = s8 = s9 = m3 = m4 = 0;

  /*  Exit if there's only one component (unless this is the split action)  */

  ncomp = mod->shape.ncomp;
  if (ncomp == 1 && par->action != SPLIT)
    bailout("merge_comps.c: nothing to merge for a one-component model\n");


  /*************************************************************************************
    Initialization block:

    Determine which vertices of the input model's realization will be in the merged
    model and which ones will be used to initialize the merged model, then perform
    the initialization by copying the intact portions of the input realization
    (i.e., facets that aren't "broken" at component boundaries) to the merged model.
    Also create arrays that contain the mappings in both directions between vertex
    numbers in the new model and (component, vertex) pairs in the old model, and
    similarly for facet numbers and side numbers.
  **************************************************************************************/


  /*  Count active vertices, total vertices, total facets, and total sides in the
      input model.  Also create the vector nv_0, which permits us to convert from
      a vertex number v within a given input component c to a vertex number n for
      all input vertices in all components numbered in a single sequential list:
      n = nv_0[c] + v.  Do the same for facets (nf_0) and sides (ns_0).            */

  nv_act = nv_tot = nf_tot = ns_tot = 0;
  nv_0 = ivector( 0, ncomp-1);
  nf_0 = ivector( 0, ncomp-1);
  ns_0 = ivector( 0, ncomp-1);
  for (c=0; c<ncomp; c++) {
    real = &mod->shape.comp[c].real;
    nv  = real->nv;
    nf  = real->nf;
    ns  = real->ns;
    nv_0[c] = nv_tot;   /* number of vertices in components with c' < c */
    nf_0[c] = nf_tot;   /* number of facets   in components with c' < c */
    ns_0[c] = ns_tot;   /* number of sides    in components with c' < c */
    nv_tot += nv;
    nf_tot += nf;
    ns_tot += ns;
    for (v=0; v<nv; v++)
      if (real->v[v].act)
        nv_act++;
  }

  /*  Initialize flags that identify the vertices, facets, and sides we'll
      start with in constructing the merged model.  All active vertices
      (i.e., external vertices) are put into the initial merged model.
      All facets with three active vertices at their corners are put into
      the initial merged model.  To be put into the initial merged model,
      a side must have two active vertices at its ends AND at least one of
      its two attached facets must be in the initial merged model.          */

  v_in_model = ivector( 0, nv_tot-1);
  f_in_model = ivector( 0, nf_tot-1);
  s_in_model = ivector( 0, ns_tot-1);
  for (c=0; c<ncomp; c++) {
    real = &mod->shape.comp[c].real;
    nv  = real->nv;
    for (v=0; v<nv; v++) {
      n = nv_0[c] + v;
      v_in_model[n] = real->v[v].act;
    }
    nf  = real->nf;
    for (f=0; f<nf; f++) {
      v1 = real->f[f].v[0];
      v2 = real->f[f].v[1];
      v3 = real->f[f].v[2];
      n = nf_0[c] + f;
      if (real->v[v1].act && real->v[v2].act && real->v[v3].act)
        f_in_model[n] = 1;
      else
        f_in_model[n] = 0;
    }
    ns  = real->ns;
    for (s=0; s<ns; s++) {
      f1 = real->s[s].f[0];
      f2 = real->s[s].f[1];
      n = ns_0[c] + s;
      if (real->s[s].act && (f_in_model[ nf_0[c] + f1 ] || f_in_model[ nf_0[c] + f2 ]))
        s_in_model[n] = 1;
      else
        s_in_model[n] = 0;
    }
  }

  /*  Iteratively remove vertices, facets, and sides from this starting list
      until we're left with nothing but complete facets.  That is, remove
      any vertex that has fewer than two sides connecting it to other
      starting vertices and that isn't part of at least one facet with
      starting vertices at all three corners, then remove all facets and
      sides attached to the vertex being removed.  Sides are also removed
      if both of their adjoining facets are removed.                          */

  do {
      keep_going = 0;
      for (c=0; c<ncomp; c++) {
        real = &mod->shape.comp[c].real;
        nv  = real->nv;
        for (v=0; v<nv; v++) {
          n = nv_0[c] + v;
          if (v_in_model[n]) {
            n_goodfacets = 0;
            for (j=0; j<real->v[v].naf; j++) {
              f = real->v[v].af[j];
              if (f_in_model[ nf_0[c] + f ])
                n_goodfacets++;
            }
            n_goodsides = 0;
            for (k=0; k<real->v[v].nas; k++) {
              s = real->v[v].as[k];
              if (s_in_model[ ns_0[c] + s ])
                n_goodsides++;
            }
            if (n_goodfacets < 1 || n_goodsides < 2) {

              /*  Remove this vertex from the initial merged model  */

              v_in_model[n] = 0;

              /*  Remove all facets attached to the vertex we just removed  */

              for (j=0; j<real->v[v].naf; j++) {
                f = real->v[v].af[j];
                f_in_model[ nf_0[c] + f ] = 0;

                /*  Loop through all three sides of the facet we just removed
                    from the initial merged model; remove a side from the model
                    if both of its adjoining facets have been removed.           */

                for (k=0; k<=2; k++) {
                  s = real->f[f].s[k];
                  if (real->s[s].f[0] == f)
                    f2 = real->s[s].f[1];
                  else
                    f2 = real->s[s].f[0];
                  if (!f_in_model[ nf_0[c] + f2 ])
                    s_in_model[ ns_0[c] + s ] = 0;
                }
              }

              /*  Remove all sides attached to the vertex we just removed  */

              for (k=0; k<real->v[v].nas; k++) {
                s = real->v[v].as[k];
                s_in_model[ ns_0[c] + s ] = 0;
              }

              /*  Set a flag that says we've removed a vertex so we'll
                  need to loop through all vertices at least once more  */

              keep_going = 1;
            }
          }
        }
      }
  } while (keep_going);

  /*  Set the number of vertices, facets, and sides that the merged model
      could potentially have (if it incorporates all active vertices)
      before we then eliminate any very small facets                       */

  nv_merged = nv_act;
  nf_merged = 2*nv_merged - 4;
  ns_merged = 3*nv_merged - 6;

  /*  Create storage arrays that will hold the properties of the new vertices,
      facets, and sides; these arrays mostly mimic a standard vertices_t structure
      but it would be unwieldy to use those long structure names in the code below  */

  v_naf = ivector( 0, nv_merged-1);                  /* number of facets   per vertex */
  v_af = imatrix( 0, nv_merged-1, 0, MAXATTACH-1);   /* attached  facets   per vertex */
  v_nas = ivector( 0, nv_merged-1);                  /* number of sides    per vertex */
  v_as = imatrix( 0, nv_merged-1, 0, MAXATTACH-1);   /* attached  sides    per vertex */
  v_av = imatrix( 0, nv_merged-1, 0, MAXATTACH-1);   /* attached  vertices per vertex */
  f_v = imatrix( 0, nf_merged-1, 0, 2);              /* corner    vertices per facet  */
  f_s = imatrix( 0, nf_merged-1, 0, 2);              /* bounding  sides    per facet  */
  s_v = imatrix( 0, ns_merged-1, 0, 1);              /* bounding  vertices per side   */
  s_nf = ivector( 0, ns_merged-1);                   /* number of facets   per side   */
  s_f = imatrix( 0, ns_merged-1, 0, 1);              /* attached  facets   per side   */

  /*  Set up vectors that will map from vertex, facet, and side numbers in the
      input model realization to the corresponding numbers in the merged model,
      plus an array that maps from vertex number in the merged model to
      (component, vertex) pairs in the input model realization                   */

  v_new = ivector( 0, nv_tot-1);
  f_new = ivector( 0, nf_tot-1);
  s_new = ivector( 0, ns_tot-1);

  v_old = imatrix( 0, nv_merged-1, 0, 1);

  /*  Create these mappings for the initial version of the merged model.
      nv_in_model is the number of vertices currently in the merged model
      -- it will be incremented as we add vertices later -- and similarly
      for nf_in_model and ns_in_model.                                     */

  nv_in_model = 0;
  for (c=0; c<ncomp; c++) {
    real = &mod->shape.comp[c].real;
    nv  = real->nv;
    for (v=0; v<nv; v++) {
      n = nv_0[c] + v;
      if (v_in_model[n]) {
          v_old[nv_in_model][0] = c;
          v_old[nv_in_model][1] = v;
          v_new[n] = nv_in_model;
          nv_in_model++;
      } else {
          v_new[n] = -1;
      }
    }
  }

  nf_in_model = 0;
  for (c=0; c<ncomp; c++) {
    real = &mod->shape.comp[c].real;
    nf  = real->nf;
    for (f=0; f<nf; f++) {
      n = nf_0[c] + f;
      if (f_in_model[n]) {
          f_new[n] = nf_in_model;
          nf_in_model++;
      } else {
          f_new[n] = -1;
      }
    }
  }

  ns_in_model = 0;
  for (c=0; c<ncomp; c++) {
    real = &mod->shape.comp[c].real;
    ns  = real->ns;
    for (s=0; s<ns; s++) {
      n = ns_0[c] + s;
      if (s_in_model[n]) {
          s_new[n] = ns_in_model;
          ns_in_model++;
      } else {
          s_new[n] = -1;
      }
    }
  }

  /*  Copy the starting vertices' intact connections to the appropriate arrays  */

  m = 0;
  for (c=0; c<ncomp; c++) {
    real = &mod->shape.comp[c].real;
    nv  = real->nv;
    for (v=0; v<nv; v++) {
      n = nv_0[c] + v;
      if (v_in_model[n]) {
        v_naf[m] = 0;
        for (j=0; j<real->v[v].naf; j++) {
          f = real->v[v].af[j];
          if (f_in_model[ nf_0[c] + f ]) {
            if (v_naf[m] == MAXATTACH)
              bailout("merge_comps.c: need to increase MAXATTACH\n");
            f2 = f_new[ nf_0[c] + f ];
            v_af[m][v_naf[m]] = f2;
            v_naf[m]++;
          }
        }

        v_nas[m] = 0;
        for (k=0; k<real->v[v].nas; k++) {
          s = real->v[v].as[k];
          if (s_in_model[ ns_0[c] + s ]) {
            if (v_nas[m] == MAXATTACH)
              bailout("merge_comps.c: need to increase MAXATTACH\n");
            s2 = s_new[ ns_0[c] + s ];
            v_as[m][v_nas[m]] = s2;
            n1 = nv_0[c] + real->s[s].v[0];
            n2 = nv_0[c] + real->s[s].v[1];
            m2 = (n == n1) ? v_new[n2] : v_new[n1];
            v_av[m][v_nas[m]] = m2;
            v_nas[m]++;
          }
        }

        m++;
      }
    }
  }

  /*  Copy the starting facets' vertices and sides to the appropriate arrays  */

  m = 0;
  for (c=0; c<ncomp; c++) {
    real = &mod->shape.comp[c].real;
    nf  = real->nf;
    for (f=0; f<nf; f++)
      if (f_in_model[ nf_0[c] + f ]) {
        for (j=0; j<=2; j++)
          f_v[m][j] = v_new[ nv_0[c] + real->f[f].v[j] ];
        for (j=0; j<=2; j++)
          f_s[m][j] = s_new[ ns_0[c] + real->f[f].s[j] ];
        m++;
      }
  }

  /*  Copy the starting sides' vertices and attached facets to the appropriate arrays;
      if only one formerly attached facet is still intact, make that the first
      attached facet for the new side and flag the second attached facet with a -1      */

  m = 0;
  for (c=0; c<ncomp; c++) {
    real = &mod->shape.comp[c].real;
    ns  = real->ns;
    for (s=0; s<ns; s++)
      if (s_in_model[ ns_0[c] + s ]) {
        for (j=0; j<=1; j++)
          s_v[m][j] = v_new[ nv_0[c] + real->s[s].v[j] ];
        f1 = f_new[ nf_0[c] + real->s[s].f[0] ];
        f2 = f_new[ nf_0[c] + real->s[s].f[1] ];
        if (f1 == -1) {
            s_f[m][0] = f2;
            s_f[m][1] = f1;
            s_nf[m] = 1;
        } else {
            s_f[m][0] = f1;
            s_f[m][1] = f2;
            s_nf[m] = (f2 == -1) ? 1 : 2;
        }
        m++;
      }
  }

  /*  For convenience, create an array that tells us whether or not
      any two vertices within the input model realization are
      connected (yet) in the merged model                            */

  v_connected = imatrix( 0, nv_tot-1, 0, nv_tot-1);
  for (n=0; n<nv_tot; n++)
    for (n2=0; n2<nv_tot; n2++)
      v_connected[n][n2] = 0;

  for (n=0; n<nv_tot; n++) {
    m = v_new[n];
    if (m != -1)
      for (k=0; k<v_nas[m]; k++) {
        m2 = v_av[m][k];
        c2 = v_old[m2][0];
        v2 = v_old[m2][1];
        n2 = nv_0[c2] + v2;
        v_connected[n][n2] = 1;
      }
  }

  /*  Create an array that tells us the distance between any two
      active vertices in the input model realization              */

  v_distance = matrix( 0, nv_tot-1, 0, nv_tot-1);
  for (n=0; n<nv_tot; n++)
    for (n2=0; n2<nv_tot; n2++)
      v_distance[n][n2] = HUGENUMBER;

  for (c=0; c<ncomp; c++) {
    real = &mod->shape.comp[c].real;
    nv  = real->nv;
    for (v=0; v<nv; v++)
      if (real->v[v].act) {
        n = nv_0[c] + v;
        for (c2=c; c2<ncomp; c2++) {
          real2  = &mod->shape.comp[c2].real;
          nv2  = real2->nv;
          for (v2=0; v2<nv2; v2++)
            if ((c2 > c || v2 > v) && real2->v[v2].act) {
              n2 = nv_0[c2] + v2;
              v_distance[n][n2] = v_distance[n2][n]
                                = distance( real->v[v].x, real2->v[v2].x);
            }
        }
      }
  }

  /*  Create a vector that we'll use later for work space  */

  attachvec = ivector( 0, MAXATTACH-1);

  /*  Send status to the screen  */

  if (par->action != SPLIT) {

    /*  Display basic information on the initial merged model  */

    printf("#\n");
    printf("# merging %d-component model into a 1-component vertex model\n", ncomp);
    printf("#\n");
    printf("# input  model realization has  %d surface vertices\n", nv_act);
    printf("# merged model initialized with %d of these vertices,\n", nv_in_model);
    printf("#              forming %d complete facets with %d sides;\n",
           nf_in_model, ns_in_model);

    /*  Figure out how many vertices are included in each contiguous subset
        of the initial merged model and print that information to the screen  */

    printf("#              there are (");
    subsetvec = ivector( 0, nv_tot-1);
    for (n=0; n<nv_tot; n++)
      subsetvec[n] = -1;
    added_v_attached = ivector( 0, nv_tot-1);
    for (n=0; n<nv_tot; n++)
      added_v_attached[n] = 0;
    n_subsets = 0;
    nv_in_all_subsets = 0;
    do {

        /*  Go through this while loop once for each contiguous model subset  */

        new_subset = 1;
        nv_in_subset = 0;
        do {

            /*  Go through this while loop repeatedly for a given contiguous model
                subset; on each pass, add in vertices that are attached to vertices
                previously identified as belonging to this subset.  Stop when no
                more vertices are added on a pass.                                   */

            keep_going = 0;
            for (n=0; n<nv_tot; n++)

              /*  Consider vertex n if it hasn't already had
                  its attached vertices added to some subset  */

              if (!added_v_attached[n]) {
                m = v_new[n];
                if (m == -1) {

                    /*  Vertex n isn't part of the initial merged model, so mark
                        its attached vertices as having been added to some subset
                        -- since in fact it doesn't have any attached vertices     */

                    added_v_attached[n] = 1;
                } else {

                    /*  If this subset doesn't yet include any vertices,
                        and vertex n hasn't yet been assigned to any subset,
                        assign it to be the first vertex in this subset       */

                    if (new_subset && subsetvec[n] == -1) {
                      subsetvec[n] = n_subsets;
                      nv_in_subset++;
                      new_subset = 0;
                    }

                    /*  If vertex n belongs to this subset, check its attached vertices,
                        add them to this subset if they haven't already been added, and
                        mark vertex n as having had its attached vertices added           */

                    if (subsetvec[n] == n_subsets) {
                      for (k=0; k<v_nas[m]; k++) {
                        m2 = v_av[m][k];
                        c2 = v_old[m2][0];
                        v2 = v_old[m2][1];
                        n2 = nv_0[c2] + v2;
                        if (subsetvec[n2] == -1) {
                          subsetvec[n2] = subsetvec[n];
                          nv_in_subset++;
                          keep_going = 1;
                        }
                      }
                      added_v_attached[n] = 1;
                    }
                }
              }
        } while (keep_going);

        /*  Display information about this subset  */

        if (nv_in_subset > 1) {
          if (n_subsets > 0)
            printf(" ");
          printf("%d", nv_in_subset);
        }
        n_subsets++;
        nv_in_all_subsets += nv_in_subset;
    } while (nv_in_all_subsets < nv_in_model);
    printf(") vertices in %d contiguous model subsets\n", n_subsets);
    printf("#\n");
    fflush(stdout);
    free_ivector( subsetvec, 0, nv_tot-1);
    free_ivector( added_v_attached, 0, nv_tot-1);
  }


  /*************************************************************************************
    Merging block:

    Loop through each side s of each facet f of the new model, and if side s has only
    one attached facet so far, search for an active vertex v to be connected to the two
    vertices v1 and v2 at the ends of side s to form a new facet.  Vertex v must meet
    six criteria: it cannot be "finished" in the sense that all of its attached sides
    already have two attached facets; if it is already connected to v1, the connecting
    side s1 must have only one attached facet so far, and similarly for v2 and
    connecting side s2; if neither s1 nor s2 already exists, neither v1 nor v2 can come
    from the same input component as v; if either s1 or s2 already exists, the vertex
    order for the proposed new facet must be consistent with the vertex order that has
    already been established along sides s and s1 or s2; if s1 already exists, the
    proposed new side s2 cannot leave v1 surrounded by a filled surface but with a
    "stranded" facet jutting out from it in the third dimension, and similarly for s2,
    s1 and v1; and, if these five criteria are met, v must be the active vertex that
    minimizes the new facet's perimeter (i.e., that minimizes the sum of distances v-v1
    and v-v2).

    Once vertex v has been identified for side s, update the lists of sides, attached
    vertices, facet corners, etc.
  **************************************************************************************/


  /*  Go through each side of each facet  */

  for (f=0; f<nf_merged; f++) {
    for (k=0; k<=2; k++) {
      s = f_s[f][k];

      /*  If side s has only one attached facet, we need to find
          some vertex m to attach to side s to make a new facet   */

      if (s_nf[s] == 1) {

        /*  Side s runs from vertex m1 to m2 (in the numbering of the merged model)  */

        m1 = s_v[s][0];
        c1 = v_old[m1][0];
        v1 = v_old[m1][1];
        n1 = nv_0[c1] + v1;
        m2 = s_v[s][1];
        c2 = v_old[m2][0];
        v2 = v_old[m2][1];
        n2 = nv_0[c2] + v2;

        /*  Loop through all active vertices in the input model realization,
            testing whether or not each one would produce a geometrically
            valid connection to side s and, if so, looking for the vertex
            that produces a new facet with the smallest perimeter length      */

        distmin = HUGENUMBER;
        cmin = -1;
        vmin = -1;
        for (c=0; c<ncomp; c++) {
          real = &mod->shape.comp[c].real;
          nv  = real->nv;
          for (v=0; v<nv; v++) {
            n = nv_0[c] + v;
            if (real->v[v].act) {
              m = v_new[n];

              /*  This is an active vertex whose number in the merged model is m.
                  If in fact it isn't yet included in the merged model (m == -1)
                  then a connection between it and side s would certainly be valid;
                  if it's already included, though, we'll have to check the validity.  */

              s1 = s2 = -1;
              valid_connection = 1;
              if (m != -1) {

                /*  Check whether or not a connection to vertex m would be valid.
                    First we'll make sure that vertex m isn't already "finished"
                    in the sense of being completely surrounded by a surface of
                    facets (i.e., every side attached to m has two attached facets).
                    Second, check that if side s1 connecting vertex m1 to m already
                    exists, it has only one attached facet, and similarly if side s2
                    connecting vertex m2 to m already exists.  Third, if neither s1
                    nor s2 already exists, check that neither m1 nor m2 comes from the
                    same input component as m; this discourages the creation of a
                    facet INSIDE the model where a component boundary makes a hairpin
                    turn and passes close to itself.  Fourth, if either s1 or s2
                    already exists, make sure that the vertex order for proposed new
                    facet m1-m2-m (which determines the sense of the facet normal) is
                    consistent with the vertex order that has already been established
                    on side s and on s1 or s2.                                          */

                n_finished_sides = 0;
                for (j=0; j<v_nas[m]; j++) {
                  if (s_nf[v_as[m][j]] == 2)
                    n_finished_sides++;

                  if (v_av[m][j] == m1)
                    s1 = v_as[m][j];
                  else if (v_av[m][j] == m2)
                    s2 = v_as[m][j];
                }
                if (n_finished_sides == v_nas[m]) {
                    valid_connection = 0;
                } else if ((s1 != -1 && s_nf[s1] == 2) || (s2 != -1 && s_nf[s2] == 2)) {
                    valid_connection = 0;
                } else if (s1 == -1 && s2 == -1 && (c == c1 || c == c2)) {
                    valid_connection = 0;
                } else if (s1 != -1 || s2 != -1) {
                    f2 = s_f[s][0];
                    w1 = f_v[f2][0];
                    w2 = f_v[f2][1];
                    w3 = f_v[f2][2];
                    if ((w1 == m1 && w2 == m2) || (w2 == m1 && w3 == m2) || (w3 == m1 && w1 == m2)) {

                        /*  The vertex order for the new facet would go from vertex m1 to m
                            along s1 and from m to m2 along s2, so the order for the other
                            facet attached to s1 (if any) should go from m to m1 and the order
                            for the other facet attached to s2 (if any) should go from m2 to m  */

                        if (s1 != -1) {
                          f2 = s_f[s1][0];
                          w1 = f_v[f2][0];
                          w2 = f_v[f2][1];
                          w3 = f_v[f2][2];
                          if ((w1 == m1 && w2 == m) || (w2 == m1 && w3 == m) || (w3 == m1 && w1 == m))
                            valid_connection = 0;
                        }
                        if (s2 != -1) {
                          f2 = s_f[s2][0];
                          w1 = f_v[f2][0];
                          w2 = f_v[f2][1];
                          w3 = f_v[f2][2];
                          if ((w1 == m && w2 == m2) || (w2 == m && w3 == m2) || (w3 == m && w1 == m2))
                            valid_connection = 0;
                        }

                    } else {

                        /*  Reversed vertex order from the preceding block  */

                        if (s1 != -1) {
                          f2 = s_f[s1][0];
                          w1 = f_v[f2][0];
                          w2 = f_v[f2][1];
                          w3 = f_v[f2][2];
                          if ((w1 == m && w2 == m1) || (w2 == m && w3 == m1) || (w3 == m && w1 == m1))
                            valid_connection = 0;
                        }
                        if (s2 != -1) {
                          f2 = s_f[s2][0];
                          w1 = f_v[f2][0];
                          w2 = f_v[f2][1];
                          w3 = f_v[f2][2];
                          if ((w1 == m2 && w2 == m) || (w2 == m2 && w3 == m) || (w3 == m2 && w1 == m))
                            valid_connection = 0;
                        }
                    }
                }

                /*  Our fifth validity criterion: If side s1 already exists, check that by
                    creating new side s2 we wouldn't be "stranding" a facet that's already
                    attached to vertex m1.  That is, we don't want to have m1 surrounded by a
                    filled surface of attached facets PLUS have another attached facet
                    sticking out from m1 in the third dimension.

                    The code below checks whether or not any sides attached to m1 (other than s
                    and s1) have only one attached facet.  If so, we need to worry, so we check
                    whether, starting from vertex m, we can follow the connections between those
                    vertices that are attached to m1 and make our way around m1 to get to m2.
                    If we can do this, we are forbidden to "complete the circuit" by connecting
                    m2 to m via side 2.  The check is made by routine stranded_facet, which takes
                    the brute force approach of checking every permutation of every combination
                    of attached vertices that could lie between m and m2 along such a circuit.     */

                if (valid_connection && s1 != -1 && v_nas[m1] >= 5) {
                  keep_going = 0;
                  n_attached = 0;
                  for (j=0; j<v_nas[m1]; j++) {
                    c_attached = v_old[v_av[m1][j]][0];
                    v_attached = v_old[v_av[m1][j]][1];
                    if (v_av[m1][j] != m && v_av[m1][j] != m2) {
                      attachvec[n_attached] = nv_0[c_attached] + v_attached;
                      n_attached++;
                      if (s_nf[v_as[m1][j]] == 1)
                        keep_going = 1;
                    }
                  }
                  if (keep_going && stranded_facet( n, n2, attachvec, n_attached, v_connected))
                    valid_connection = 0;
                }

                /*  Make the same check (don't strand a facet) for vertex m2  */

                if (valid_connection && s2 != -1 && v_nas[m2] >= 5) {
                  keep_going = 0;
                  n_attached = 0;
                  for (j=0; j<v_nas[m2]; j++) {
                    c_attached = v_old[v_av[m2][j]][0];
                    v_attached = v_old[v_av[m2][j]][1];
                    if (v_av[m2][j] != m && v_av[m2][j] != m1) {
                      attachvec[n_attached] = nv_0[c_attached] + v_attached;
                      n_attached++;
                      if (s_nf[v_as[m2][j]] == 1)
                        keep_going = 1;
                    }
                  }
                  if (keep_going && stranded_facet( n, n1, attachvec, n_attached, v_connected))
                    valid_connection = 0;
                }
              }

              /*  If all of the above criteria were met, a connection from side s
                  to vertex m would be geometrically valid, so now we can check if
                  this vertex m would minimize the perimeter of facet m1-m2-m       */

              if (valid_connection) {
                distsum = v_distance[n][n1] + v_distance[n][n2];
                if (distsum < distmin) {
                  distmin = distsum;
                  cmin = c;
                  vmin = v;
                }
              }
            }
          }    /* end vertex loop */
        }      /* end component loop */

        /*  We've found our connection  */

        n = nv_0[cmin] + vmin;
        m = v_new[n];

        /*  If this vertex isn't already in the merged model,
            add it and initialize its attached sides and vertices  */

        if (m == -1) {
          m = v_new[n] = nv_in_model;
          v_in_model[n] = 1;
          v_old[m][0] = cmin;
          v_old[m][1] = vmin;
          v_naf[m] = v_nas[m] = 0;
          nv_in_model++;
        }

        /*  Attach the new facet to each of vertices m, m1, and m2  */

        v_af[m][v_naf[m]] = v_af[m1][v_naf[m1]] = v_af[m2][v_naf[m2]] = nf_in_model;
        v_naf[m]++;
        v_naf[m1]++;
        v_naf[m2]++;

        /*  If side s1 already exists, just get its number; if it's new, add it
            to the new model, attach it to vertices m and m1, flag it as connected
            to m and m1, assign m and m1 as its bounding vertices, and initialize
            its attached facets                                                     */

        if (v_connected[n][n1]) {
            s1_is_new = 0;
            for (j=0; j<v_nas[m]; j++)
              if (v_av[m][j] == m1)
                s1 = v_as[m][j];
        } else {
            s1_is_new = 1;
            s1 = ns_in_model;
            if (ns_in_model == ns_merged)
              bailout("found too many connections between vertices\n");
            v_as[m][v_nas[m]] = s1;
            v_av[m][v_nas[m]] = m1;
            v_nas[m]++;
            v_as[m1][v_nas[m1]] = s1;
            v_av[m1][v_nas[m1]] = m;
            v_nas[m1]++;
            v_connected[n][n1] = v_connected[n1][n] = 1;
            s_v[s1][0] = m1;
            s_v[s1][1] = m;
            s_f[s1][0] = s_f[s1][1] = -1;
            s_nf[s1] = 0;
            ns_in_model++;
        }

        /*  Do the same for side s2  */

        if (v_connected[n][n2]) {
            s2_is_new = 0;
            for (j=0; j<v_nas[m]; j++)
              if (v_av[m][j] == m2)
                s2 = v_as[m][j];
        } else {
            s2_is_new = 1;
            s2 = ns_in_model;
            if (ns_in_model == ns_merged)
              bailout("found too many connections between vertices\n");
            v_as[m][v_nas[m]] = s2;
            v_av[m][v_nas[m]] = m2;
            v_nas[m]++;
            v_as[m2][v_nas[m2]] = s2;
            v_av[m2][v_nas[m2]] = m;
            v_nas[m2]++;
            v_connected[n][n2] = v_connected[n2][n] = 1;
            s_v[s2][0] = m2;
            s_v[s2][1] = m;
            s_f[s2][0] = s_f[s2][1] = -1;
            s_nf[s2] = 0;
            ns_in_model++;
        }

        /*  Attach the new facet to each of sides s, s1, and s2  */

        s_f[s][s_nf[s]] = s_f[s1][s_nf[s1]] = s_f[s2][s_nf[s2]] = nf_in_model;
        s_nf[s]++;
        s_nf[s1]++;
        s_nf[s2]++;

        /*  Add the new facet to the merged model by assigning its vertices and
            sides, being careful to get the vertex order right by looking at the
            vertex order for the other facet that's attached to side s            */

        f2 = s_f[s][0];
        w1 = f_v[f2][0];
        w2 = f_v[f2][1];
        w3 = f_v[f2][2];
        if ((w1 == m1 && w2 == m2) || (w2 == m1 && w3 == m2) || (w3 == m1 && w1 == m2)) {
            f_v[nf_in_model][0] = m2;
            f_v[nf_in_model][1] = m1;
            f_v[nf_in_model][2] = m;
            f_s[nf_in_model][0] = s;
            f_s[nf_in_model][1] = s1;
            f_s[nf_in_model][2] = s2;
        } else {
            f_v[nf_in_model][0] = m1;
            f_v[nf_in_model][1] = m2;
            f_v[nf_in_model][2] = m;
            f_s[nf_in_model][0] = s;
            f_s[nf_in_model][1] = s2;
            f_s[nf_in_model][2] = s1;
        }
        nf_in_model++;

        /*  If a new side s1 was created above, check to see if it left a triangular
            "space" on the other side of it, and if so, catalogue that space as a
            new facet.  To check for this, look for a vertex m3 that's not the same
            as m2 and that's attached both to m and to m1 by sides s3 and s4 that
            have one attached facet each.                                             */

        if (s1_is_new) {
          n3 = 0;
          keep_going = 1;
          while (n3 < nv_tot && keep_going) {
            if (v_connected[n][n3] && v_connected[n1][n3] && n3 != n2) {
              m3 = v_new[n3];
              for (j=0; j<v_nas[m3]; j++)
                if (v_av[m3][j] == m)
                  s3 = v_as[m3][j];
                else if (v_av[m3][j] == m1)
                  s4 = v_as[m3][j];
              if (s_nf[s3] == 1 && s_nf[s4] == 1)
                keep_going = 0;
            }
            if (keep_going)
              n3++;
          }
          if (n3 != nv_tot) {
            v_af[m][v_naf[m]] = v_af[m1][v_naf[m1]] = v_af[m3][v_naf[m3]] = nf_in_model;
            v_naf[m]++;
            v_naf[m1]++;
            v_naf[m3]++;
            s_f[s1][s_nf[s1]] = s_f[s3][s_nf[s3]] = s_f[s4][s_nf[s4]] = nf_in_model;
            s_nf[s1]++;
            s_nf[s3]++;
            s_nf[s4]++;
            f2 = s_f[s1][0];
            w1 = f_v[f2][0];
            w2 = f_v[f2][1];
            w3 = f_v[f2][2];
            if ((w1 == m1 && w2 == m) || (w2 == m1 && w3 == m) || (w3 == m1 && w1 == m)) {
                f_v[nf_in_model][0] = m;
                f_v[nf_in_model][1] = m1;
                f_v[nf_in_model][2] = m3;
                f_s[nf_in_model][0] = s1;
                f_s[nf_in_model][1] = s4;
                f_s[nf_in_model][2] = s3;
            } else {
                f_v[nf_in_model][0] = m1;
                f_v[nf_in_model][1] = m;
                f_v[nf_in_model][2] = m3;
                f_s[nf_in_model][0] = s1;
                f_s[nf_in_model][1] = s3;
                f_s[nf_in_model][2] = s4;
            }
            nf_in_model++;
          }
        }

        /*  Do the same thing for side s2 if it was newly created above  */

        if (s2_is_new) {
          n3 = 0;
          keep_going = 1;
          while (n3 < nv_tot && keep_going) {
            if (v_connected[n][n3] && v_connected[n2][n3] && n3 != n1) {
              m3 = v_new[n3];
              for (j=0; j<v_nas[m3]; j++)
                if (v_av[m3][j] == m)
                  s3 = v_as[m3][j];
                else if (v_av[m3][j] == m2)
                  s4 = v_as[m3][j];
              if (s_nf[s3] == 1 && s_nf[s4] == 1)
                keep_going = 0;
            }
            if (keep_going)
              n3++;
          }
          if (n3 != nv_tot) {
            v_af[m][v_naf[m]] = v_af[m2][v_naf[m2]] = v_af[m3][v_naf[m3]] = nf_in_model;
            v_naf[m]++;
            v_naf[m2]++;
            v_naf[m3]++;
            s_f[s2][s_nf[s2]] = s_f[s3][s_nf[s3]] = s_f[s4][s_nf[s4]] = nf_in_model;
            s_nf[s2]++;
            s_nf[s3]++;
            s_nf[s4]++;
            f2 = s_f[s2][0];
            w1 = f_v[f2][0];
            w2 = f_v[f2][1];
            w3 = f_v[f2][2];
            if ((w1 == m2 && w2 == m) || (w2 == m2 && w3 == m) || (w3 == m2 && w1 == m)) {
                f_v[nf_in_model][0] = m;
                f_v[nf_in_model][1] = m2;
                f_v[nf_in_model][2] = m3;
                f_s[nf_in_model][0] = s2;
                f_s[nf_in_model][1] = s4;
                f_s[nf_in_model][2] = s3;
            } else {
                f_v[nf_in_model][0] = m2;
                f_v[nf_in_model][1] = m;
                f_v[nf_in_model][2] = m3;
                f_s[nf_in_model][0] = s2;
                f_s[nf_in_model][1] = s3;
                f_s[nf_in_model][2] = s4;
            }
            nf_in_model++;
          }
        }

      }   /* end "if facet side has only one attached facet" */
    }     /* end k-loop over facet sides                     */
  }       /* end f-loop over facets                          */


  /*  Send status to the screen  */

  printf("#\n");
  if (par->action == SPLIT)
    printf("# closure completed: model has %d vertices, %d facets, %d sides\n",
           nv_in_model, nf_in_model, ns_in_model);
  else
    printf("# merger completed: model has %d vertices, %d facets, %d sides\n",
           nv_in_model, nf_in_model, ns_in_model);
  fflush(stdout);

  if ((nf_in_model != 2*nv_in_model - 4) || (ns_in_model != 3*nv_in_model - 6)) {
      fprintf( stderr, "# should be %d facets and %d sides for %d vertices\n",
                       2*nv_in_model - 4, 3*nv_in_model - 6, nv_in_model);
      n_missingfacets = (2*nv_in_model - 4) - nf_in_model;
      if (n_missingfacets > 0 && n_missingfacets % 4 == 0) {
        n_pieces = 1 + n_missingfacets/4;
        if (par->action == SPLIT)
           fprintf( stderr,
                    "WARNING: this model subset may be broken into %d spatially disjoint pieces\n",
                    n_pieces);
        else
          fprintf( stderr,
                   "WARNING: input model may be broken into %d spatially disjoint pieces\n",
                   n_pieces);
      }
      bailout("merge_comps.c\n");
  } else if (nf_in_model < 4) {
      bailout("too few facets found, surface is no longer closed\n");
  }


  /*************************************************************************************
    Optimization block:

    The merged model is complete, but near the component boundaries we may have some
    sharp "creases" (adjacent facets that are highly tilted relative to each other) and
    some facets that are extremely narrow or small, so we now fix these problems.
  **************************************************************************************/


  /*  Find the smallest facet side in the input model realization and then set the
      facet size threshold for the merged model to be a fixed fraction of this length;
      then initialize a vector that will flag whether or not a given facet in the
      merged model is narrower, or is smaller in both dimensions, than this threshold   */

  minheight = HUGENUMBER;
  for (c=0; c<ncomp; c++) {
    real = &mod->shape.comp[c].real;
    nv  = real->nv;
    for (v=0; v<nv-1; v++) {
      n = nv_0[c] + v;
      for (v2=v+1; v2<nv; v2++) {
        n2 = nv_0[c] + v2;
        minheight = MIN( minheight, v_distance[n][n2]);
      }
    }
  }
  minheight *= 0.1;

  f_very_small = ivector( 0, nf_in_model-1);
  for (f=0; f<nf_in_model; f++)
    f_very_small[f] = 0;

  /*  Look at each side to see if we can redefine the two attached facets
      so as to make them significantly more coplanar or else to turn a
      very narrow facet attached to a normal one into two normal ones      */

  for (s=0; s<ns_in_model; s++) {

    /*  s is the side common to facets f1 and f2; it runs from vertex
        m1 to m2 for f1 and from m2 to m1 for f2.  The third vertex in
        f1 is m3 and the third vertex in f2 is m4.                      */

    f1 = s_f[s][0];
    if ((f_v[f1][0] == s_v[s][0] && f_v[f1][1] == s_v[s][1])
           || (f_v[f1][0] == s_v[s][1] && f_v[f1][1] == s_v[s][0])) {
        m1 = f_v[f1][0];
        m2 = f_v[f1][1];
        m3 = f_v[f1][2];
    } else if ((f_v[f1][1] == s_v[s][0] && f_v[f1][2] == s_v[s][1])
               || (f_v[f1][1] == s_v[s][1] && f_v[f1][2] == s_v[s][0])) {
        m1 = f_v[f1][1];
        m2 = f_v[f1][2];
        m3 = f_v[f1][0];
    } else {
        m1 = f_v[f1][2];
        m2 = f_v[f1][0];
        m3 = f_v[f1][1];
    }
    f2 = s_f[s][1];
    for (j=0; j<=2; j++)
      if (f_v[f2][j] != m1 && f_v[f2][j] != m2 && f_v[f2][j] != m3)
        m4 = f_v[f2][j];

    /*  Find the other four sides of facets f1 and f2: f1 runs from
        side s to s1 to s2 and f2 runs from side s to s3 to s4       */

    for (j=0; j<=2; j++) {
      k = f_s[f1][j];
      if ((s_v[k][0] == m2 && s_v[k][1] == m3) || (s_v[k][1] == m2 && s_v[k][0] == m3))
        s1 = k;
      else if ((s_v[k][0] == m3 && s_v[k][1] == m1) || (s_v[k][1] == m3 && s_v[k][0] == m1))
        s2 = k;
    }
    for (j=0; j<=2; j++) {
      k = f_s[f2][j];
      if ((s_v[k][0] == m1 && s_v[k][1] == m4) || (s_v[k][1] == m1 && s_v[k][0] == m4))
        s3 = k;
      else if ((s_v[k][0] == m4 && s_v[k][1] == m2) || (s_v[k][1] == m4 && s_v[k][0] == m2))
        s4 = k;
    }

    /*  Find the four other facets attached to those four sides:
        facet af1 is attached to side s1, etc.                    */

    af1 = (s_f[s1][0] == f1) ? s_f[s1][1] : s_f[s1][0];
    af2 = (s_f[s2][0] == f1) ? s_f[s2][1] : s_f[s2][0];
    af3 = (s_f[s3][0] == f2) ? s_f[s3][1] : s_f[s3][0];
    af4 = (s_f[s4][0] == f2) ? s_f[s4][1] : s_f[s4][0];

    /*  For each of facets f1 and f2, compute the unit normal, the base
        (maximum side length), and the height (perpendicular to the base);
        then compute these quantities as they would be if we redefined the
        two facets such that their common side s ran from vertex m3 to m4   */

    c1 = v_old[m1][0];
    v1 = v_old[m1][1];
    c2 = v_old[m2][0];
    v2 = v_old[m2][1];
    c3 = v_old[m3][0];
    v3 = v_old[m3][1];
    c4 = v_old[m4][0];
    v4 = v_old[m4][1];

    facetgeom( mod->shape.comp[c1].real.v[v1].x,
               mod->shape.comp[c2].real.v[v2].x,
               mod->shape.comp[c3].real.v[v3].x, norm1, &base1, &height1, &baseindex1);
    facetgeom( mod->shape.comp[c2].real.v[v2].x,
               mod->shape.comp[c1].real.v[v1].x,
               mod->shape.comp[c4].real.v[v4].x, norm2, &base2, &height2, &baseindex2);
    facetgeom( mod->shape.comp[c4].real.v[v4].x,
               mod->shape.comp[c3].real.v[v3].x,
               mod->shape.comp[c1].real.v[v1].x, norm3, &base3, &height3, &baseindex3);
    facetgeom( mod->shape.comp[c3].real.v[v3].x,
               mod->shape.comp[c4].real.v[v4].x,
               mod->shape.comp[c2].real.v[v2].x, norm4, &base4, &height4, &baseindex4);

    /*  Compute the dot product of the two facet normals as the facets currently
        exist and as they would be if we redefined them.

        NOTE: In theory, since redefining f1 and f2 would also change the angles
              between those two facets and the four adjoining facets af1 through
              af4, we could consider those four dot products as well -- say, by
              taking the mean of the five dot products.  In this case we would
              do the revision iteratively until the loop through all sides
              produced no new revisions.  However, since the goal here is to
              leave the model mostly alone and just smooth out seriously tilted
              facet pairs near the boundaries of the input model's components,
              we'll just consider the angle between f1 and f2 and will make just
              one pass through the model's sides.                                 */

    dot12 = dot( norm1, norm2);
    dot34 = dot( norm3, norm4);

    /*  Carry out initial tests to decide whether or not to redefine facets f1 and f2:

        If af2 and af3 are the same facet, m3 and m4 already have a direct connection,
        so we can't even consider giving them a second connection by redefining f1 and
        f2 -- and similarly for af4 and af1.

        Otherwise, redefine the facets if this would significantly increase the dot
        product of the two facet normals (i.e., make the facets more coplanar), OR if
        one of the facets is extremely narrow perpendicular to side s and the revision
        wouldn't significantly decrease the dot product.  (If a facet is very small
        in BOTH dimensions, leave it alone for now: we'll eliminate it later.)           */

    if (af2 == af3 || af4 == af1)
      redefine_facets = 0;
    else if (dot34 > dot12 + 0.1)
      redefine_facets = 1;
    else if (baseindex1 == 0 && (height1 < minheight || height2 < minheight)
                             && base1 > minheight
                             && height3 > minheight && base3 > minheight
                             && height4 > minheight && base4 > minheight
                             && dot34 > dot12 - 0.01)
      redefine_facets = 1;
    else
      redefine_facets = 0;

    /*  If the tests we just made suggest that we should redefine facets f1 and f2,
        we still have one more test to make: for a sufficiently complex vertex mesh,
        vertices m3 and m4 may already have a direct connection even if we've just
        verified that attached facets af2 != af3 and af4 != af1.  Thus we now do a
        thorough check to make sure that none of the vertices attached to m3 is m4.   */

    if (redefine_facets)
      for (j=0; j<v_nas[m3]; j++)
        if (v_av[m3][j] == m4)
          redefine_facets = 0;

    /*  If all of the above tests imply we should redefine facets f1 and f2, do so  */

    if (redefine_facets) {

        /*  Redefine the facets: change common side s to run from m3 to m4, and
            reconnect vertices m1, m2, m3, and m4 such that facet f1 runs from
            m4 to m3 to m1 and facet f2 runs from m3 to m4 to m2.  This change
            has no effect on the four other facets that adjoin f1 and f2.        */

        /*  Redefine side s to go from m3 to m4 rather than from m1 to m2  */

        s_v[s][0] = m3;
        s_v[s][1] = m4;

        /*  Reassign facet vertices and sides for f1 and f2  */

        f_v[f1][0] = m4;
        f_v[f1][1] = m3;
        f_v[f1][2] = m1;
        f_s[f1][0] = s;
        f_s[f1][1] = s2;
        f_s[f1][2] = s3;
        f_v[f2][0] = m3;
        f_v[f2][1] = m4;
        f_v[f2][2] = m2;
        f_s[f2][0] = s;
        f_s[f2][1] = s4;
        f_s[f2][2] = s1;

        /*  Reassign facets attached to sides s1 and s3 (nothing changes for s2 and s4)  */

        if (s_f[s1][0] == f1)
          s_f[s1][0] = f2;
        else
          s_f[s1][1] = f2;

        if (s_f[s3][0] == f2)
          s_f[s3][0] = f1;
        else
          s_f[s3][1] = f1;

        /*  Reassign sides attached to the four vertices  */

        v_as[m3][v_nas[m3]] = s;
        v_av[m3][v_nas[m3]] = m4;
        v_nas[m3]++;
        v_as[m4][v_nas[m4]] = s;
        v_av[m4][v_nas[m4]] = m3;
        v_nas[m4]++;
        j = 0;
        while (v_as[m1][j] != s)
          j++;
        for (k=j+1; k<v_nas[m1]; k++) {
          v_as[m1][k-1] = v_as[m1][k];
          v_av[m1][k-1] = v_av[m1][k];
        }
        v_nas[m1]--;
        j = 0;
        while (v_as[m2][j] != s)
          j++;
        for (k=j+1; k<v_nas[m2]; k++) {
          v_as[m2][k-1] = v_as[m2][k];
          v_av[m2][k-1] = v_av[m2][k];
        }
        v_nas[m2]--;

        /*  Reassign facets attached to the four vertices  */

        v_af[m3][v_naf[m3]] = f2;
        v_naf[m3]++;
        v_af[m4][v_naf[m4]] = f1;
        v_naf[m4]++;
        j = 0;
        while (v_af[m1][j] != f2)
          j++;
        for (k=j+1; k<v_naf[m1]; k++)
          v_af[m1][k-1] = v_af[m1][k];
        v_naf[m1]--;
        j = 0;
        while (v_af[m2][j] != f1)
          j++;
        for (k=j+1; k<v_naf[m2]; k++)
          v_af[m2][k-1] = v_af[m2][k];
        v_naf[m2]--;

        /*  Flag facets that are currently so small in both dimensions that they
            should be eliminated at the next stage; this flag may change for a
            given facet as we loop over all sides, since each facet is inspected
            more than once and may be redefined (also more than once) in the process  */

        if (base3 < minheight)
          f_very_small[f1] = 1;
        else
          f_very_small[f1] = 0;

        if (base4 < minheight)
          f_very_small[f2] = 1;
        else
          f_very_small[f2] = 0;

    } else {

        /*  Flag very small facets (see comments above)  */

        if (base1 < minheight)
          f_very_small[f1] = 1;
        else
          f_very_small[f1] = 0;

        if (base2 < minheight)
          f_very_small[f2] = 1;
        else
          f_very_small[f2] = 0;

    }
  }

  /*  As the last step in the process, go through all facets and eliminate the ones
      that are very small in both dimensions.  Each such elimination will also entail
      eliminating two vertices, six sides, and three other facets that share a side
      with the very small one.  We may need to make several passes through the list
      of facets, since each facet elimination changes the corner vertices -- and thus
      the size -- of several facets that share one vertex with the eliminated facet.

      Note that as we remove vertices, facets, and sides, we will NOT reduce the
      values of nv_in_model, nf_in_model, and ns_in_model: memory for several vectors
      is allocated using these values, so we need to keep them the same in order
      to deallocate the memory later.  Instead, we will later define a new variable
      nv_final to denote the number of post-small-facet-elimination vertices, and
      similarly will use nf_final for the final count of facets.                       */

  /*  Create and initialize some vectors of flags that indicate
      whether or not we'll be keeping a given vertex/facet/side
      in the model once we've eliminated very small facets       */

  v_good = ivector( 0, nv_in_model-1);
  f_good = ivector( 0, nf_in_model-1);
  s_good = ivector( 0, ns_in_model-1);
  for (m=0; m<nv_in_model; m++)
    v_good[m] = 1;
  for (f=0; f<nf_in_model; f++)
    f_good[f] = 1;
  for (s=0; s<ns_in_model; s++)
    s_good[s] = 1;

  /*  As mentioned above, the facet elimination process must be
      performed iteratively, so we now enter the iteration loop  */

  do {

      keep_going = 0;

      /*  Loop through all facets and eliminate the ones that were
          flagged earlier as too small; this will also entail
          eliminating vertices, sides, and other facets             */

      for (f=0; f<nf_in_model; f++) {
        if (f_very_small[f] && f_good[f]) {

          /*  f is a very small facet that hasn't yet been eliminated
              from the model, so we'll eliminate it now                */

          /*  Get the three facet vertices: m1, m2, and m3 (in order)  */

          m1 = f_v[f][0];
          m2 = f_v[f][1];
          m3 = f_v[f][2];

          /*  Get the three facet sides: s1 runs from m1 to m2;
              s2 runs from m2 to m3; and s3 runs from m3 to m1   */

          for (k=0; k<=2; k++) {
            s = f_s[f][k];
            if ((s_v[s][0] == m1 && s_v[s][1] == m2) || (s_v[s][1] == m1 && s_v[s][0] == m2))
              s1 = s;
            else if ((s_v[s][0] == m2 && s_v[s][1] == m3) || (s_v[s][1] == m2 && s_v[s][0] == m3))
              s2 = s;
            else
              s3 = s;
          }

          /*  Get the three adjoining facets: f1 is on the other side of s1, etc.  */

          f1 = (s_f[s1][0] == f) ? s_f[s1][1] : s_f[s1][0];
          f2 = (s_f[s2][0] == f) ? s_f[s2][1] : s_f[s2][0];
          f3 = (s_f[s3][0] == f) ? s_f[s3][1] : s_f[s3][0];

          /*  Get the three vertices at the far ends of the adjoining facets:
              m4 is on the far end of f1 (opposite s1), m5 is on the far end of
              f2 (opposite s2), and m6 is on the far end of f3 (opposite s3).
              If f1 and f2 adjoin each other, m4 and m5 are the same vertex; if
              f2 and f3 adjoin each other, m5 and m6 are the same vertex; if
              f3 and f1 adjoin each other, m6 and m4 are the same vertex.        */
 
          m4 = (s_v[s4][0] == m1) ? s_v[s4][1] : s_v[s4][0];
          m5 = (s_v[s6][0] == m2) ? s_v[s6][1] : s_v[s6][0];
          m6 = (s_v[s8][0] == m3) ? s_v[s8][1] : s_v[s8][0];

          /*  Get the other sides of the three adjoining facets: f1 runs from
              s1 to s4 to s5; f2 runs from s2 to s6 to s7; and f3 runs from
              s3 to s8 to s9.  If f1 and f2 adjoin each other, s5 and s6 are the
              same side; if f2 and f3 adjoin each other, s6 and s8 are the same
              side; if f3 and f1 adjoin each other, s9 and s4 are the same side.  */

          for (k=0; k<=2; k++) {
            s = f_s[f1][k];
            if (s_v[s][0] != m2 && s_v[s][1] != m2)
              s4 = s;
            else if (s_v[s][0] != m1 && s_v[s][1] != m1)
              s5 = s;
          }
          for (k=0; k<=2; k++) {
            s = f_s[f2][k];
            if (s_v[s][0] != m3 && s_v[s][1] != m3)
              s6 = s;
            else if (s_v[s][0] != m2 && s_v[s][1] != m2)
              s7 = s;
          }
          for (k=0; k<=2; k++) {
            s = f_s[f3][k];
            if (s_v[s][0] != m1 && s_v[s][1] != m1)
              s8 = s;
            else if (s_v[s][0] != m3 && s_v[s][1] != m3)
              s9 = s;
          }

          /*  Eliminate two vertices, m2 and m3  */

          v_good[m2] = v_good[m3] = 0;

          /*  Eliminate the small facet plus the three adjoining facets  */
         
          f_good[f] = f_good[f1] = f_good[f2] = f_good[f3] = 0;

          /*  Eliminate the three facet sides plus one other side per adjoining facet  */

          s_good[s1] = s_good[s2] = s_good[s3] = 0;
          s_good[s4] = s_good[s6] = s_good[s8] = 0;

          /*  Now that the adjoining facets have been eliminated, remaining sides
              s5, s7, and s9 have one attached facet each that has changed, and
              each of those attached facets has a bounding side that has changed.
              In assigning the new values, account for the fact that, for example,
              s5 and s6 might be the same side (if f1 and f2 adjoin each other).

              Since we need the current values of the "s_f" array (facets attached
              to sides) in order to update the "f_s" array (facets' bounding sides),
              the code block below does not touch s_f until all changes to f_s have
              been completed.                                                         */

          if (s5 == s6) {

              f4 = (s_f[s4][0] == f1) ? s_f[s4][1] : s_f[s4][0];
              for (k=0; k<=2; k++)
                if (f_s[f4][k] == s4)
                  f_s[f4][k] = s7;

              f4 = (s_f[s8][0] == f3) ? s_f[s8][1] : s_f[s8][0];
              for (k=0; k<=2; k++)
                if (f_s[f4][k] == s8)
                  f_s[f4][k] = s9;

              f4 = (s_f[s4][0] == f1) ? s_f[s4][1] : s_f[s4][0];
              for (j=0; j<=1; j++)
                if (s_f[s7][j] == f2)
                  s_f[s7][j] = f4;

              f4 = (s_f[s8][0] == f3) ? s_f[s8][1] : s_f[s8][0];
              for (j=0; j<=1; j++)
                if (s_f[s9][j] == f3)
                  s_f[s9][j] = f4;

          } else if (s7 == s8) {

              f4 = (s_f[s6][0] == f2) ? s_f[s6][1] : s_f[s6][0];
              for (k=0; k<=2; k++)
                if (f_s[f4][k] == s6)
                  f_s[f4][k] = s9;

              f4 = (s_f[s4][0] == f1) ? s_f[s4][1] : s_f[s4][0];
              for (k=0; k<=2; k++)
                if (f_s[f4][k] == s4)
                  f_s[f4][k] = s5;

              f4 = (s_f[s6][0] == f2) ? s_f[s6][1] : s_f[s6][0];
              for (j=0; j<=1; j++)
                if (s_f[s9][j] == f3)
                  s_f[s9][j] = f4;

              f4 = (s_f[s4][0] == f1) ? s_f[s4][1] : s_f[s4][0];
              for (j=0; j<=1; j++)
                if (s_f[s5][j] == f1)
                  s_f[s5][j] = f4;

          } else if (s9 == s4) {

              f4 = (s_f[s8][0] == f3) ? s_f[s8][1] : s_f[s8][0];
              for (k=0; k<=2; k++)
                if (f_s[f4][k] == s8)
                  f_s[f4][k] = s5;

              f4 = (s_f[s6][0] == f2) ? s_f[s6][1] : s_f[s6][0];
              for (k=0; k<=2; k++)
                if (f_s[f4][k] == s6)
                  f_s[f4][k] = s7;

              f4 = (s_f[s8][0] == f3) ? s_f[s8][1] : s_f[s8][0];
              for (j=0; j<=1; j++)
                if (s_f[s5][j] == f1)
                  s_f[s5][j] = f4;

              f4 = (s_f[s6][0] == f2) ? s_f[s6][1] : s_f[s6][0];
              for (j=0; j<=1; j++)
                if (s_f[s7][j] == f2)
                  s_f[s7][j] = f4;

          } else {

              f4 = (s_f[s4][0] == f1) ? s_f[s4][1] : s_f[s4][0];
              for (k=0; k<=2; k++)
                if (f_s[f4][k] == s4)
                  f_s[f4][k] = s5;

              f4 = (s_f[s6][0] == f2) ? s_f[s6][1] : s_f[s6][0];
              for (k=0; k<=2; k++)
                if (f_s[f4][k] == s6)
                  f_s[f4][k] = s7;

              f4 = (s_f[s8][0] == f3) ? s_f[s8][1] : s_f[s8][0];
              for (k=0; k<=2; k++)
                if (f_s[f4][k] == s8)
                  f_s[f4][k] = s9;

              f4 = (s_f[s4][0] == f1) ? s_f[s4][1] : s_f[s4][0];
              for (j=0; j<=1; j++)
                if (s_f[s5][j] == f1)
                  s_f[s5][j] = f4;

              f4 = (s_f[s6][0] == f2) ? s_f[s6][1] : s_f[s6][0];
              for (j=0; j<=1; j++)
                if (s_f[s7][j] == f2)
                  s_f[s7][j] = f4;

              f4 = (s_f[s8][0] == f3) ? s_f[s8][1] : s_f[s8][0];
              for (j=0; j<=1; j++)
                if (s_f[s9][j] == f3)
                  s_f[s9][j] = f4;

          }

          /*  Remove facets f, f1, f2, and f3 from the lists of facets
              attached to vertices m1, m4, m5, and m6 (don't bother with m2 and m3)  */

          j = 0;
          while (j < v_naf[m1]) {
            if (v_af[m1][j] == f || v_af[m1][j] == f1 || v_af[m1][j] == f3) {
                for (k=j; k<v_naf[m1]-1; k++)
                  v_af[m1][k] = v_af[m1][k+1];
                v_naf[m1]--;
            } else {
                j++;
            }
          }
          j = 0;
          while (j < v_naf[m4]) {
            if (v_af[m4][j] == f1) {
                for (k=j; k<v_naf[m4]-1; k++)
                  v_af[m4][k] = v_af[m4][k+1];
                v_naf[m4]--;
            } else {
                j++;
            }
          }
          j = 0;
          while (j < v_naf[m5]) {
            if (v_af[m5][j] == f2) {
                for (k=j; k<v_naf[m5]-1; k++)
                  v_af[m5][k] = v_af[m5][k+1];
                v_naf[m5]--;
            } else {
                j++;
            }
          }
          j = 0;
          while (j < v_naf[m6]) {
            if (v_af[m6][j] == f3) {
                for (k=j; k<v_naf[m6]-1; k++)
                  v_af[m6][k] = v_af[m6][k+1];
                v_naf[m6]--;
            } else {
                j++;
            }
          }

          /*  Take the good facets that are attached to m2 and m3, attach them to m1,
              and go through their vertex triples switching vertex number m2 (or m3) to m1.

              Since this vertex change will change the size of each of these facets,
              reevaluate whether or not they are small enough to eliminate from the model.
              If a facet that wasn't previously slated for elimination must now be
              eliminated, and if we already passed it in this iteration, set a flag
              denoting that we'll need to go through another iteration.                      */

          for (j=0; j<v_naf[m2]; j++) {
            f4 = v_af[m2][j];
            if (f_good[f4]) {
              if (v_naf[m1] == MAXATTACH)
                bailout("merge_comps.c: need to increase MAXATTACH\n");
              v_af[m1][v_naf[m1]] = f4;
              v_naf[m1]++;
              for (k=0; k<=2; k++)
                if (f_v[f4][k] == m2)
                  f_v[f4][k] = m1;
              c1 = v_old[ f_v[f4][0] ][0];
              v1 = v_old[ f_v[f4][0] ][1];
              c2 = v_old[ f_v[f4][1] ][0];
              v2 = v_old[ f_v[f4][1] ][1];
              c3 = v_old[ f_v[f4][2] ][0];
              v3 = v_old[ f_v[f4][2] ][1];
              facetgeom( mod->shape.comp[c1].real.v[v1].x,
                         mod->shape.comp[c2].real.v[v2].x,
                         mod->shape.comp[c3].real.v[v3].x, norm4, &base4, &height4, &baseindex4);
              if (base4 < minheight) {
                  if (f_very_small[f4] == 0 && f4 < f)
                    keep_going = 1;
                  f_very_small[f4] = 1;
              } else {
                  f_very_small[f4] = 0;
              }
            }
          }
          for (j=0; j<v_naf[m3]; j++) {
            f4 = v_af[m3][j];
            if (f_good[f4]) {
              if (v_naf[m1] == MAXATTACH)
                bailout("merge_comps.c: need to increase MAXATTACH\n");
              v_af[m1][v_naf[m1]] = f4;
              v_naf[m1]++;
              for (k=0; k<=2; k++)
                if (f_v[f4][k] == m3)
                  f_v[f4][k] = m1;
              c1 = v_old[ f_v[f4][0] ][0];
              v1 = v_old[ f_v[f4][0] ][1];
              c2 = v_old[ f_v[f4][1] ][0];
              v2 = v_old[ f_v[f4][1] ][1];
              c3 = v_old[ f_v[f4][2] ][0];
              v3 = v_old[ f_v[f4][2] ][1];
              facetgeom( mod->shape.comp[c1].real.v[v1].x,
                         mod->shape.comp[c2].real.v[v2].x,
                         mod->shape.comp[c3].real.v[v3].x, norm4, &base4, &height4, &baseindex4);
              if (base4 < minheight) {
                  if (f_very_small[f4] == 0 && f4 < f)
                    keep_going = 1;
                  f_very_small[f4] = 1;
              } else {
                  f_very_small[f4] = 0;
              }
            }
          }

          /*  Remove sides s1, s2, s3, s4, s6, and s8 from the lists of sides
              attached to vertices m1, m4, m5, and m6 (don't bother with m2 and m3)  */

          j = 0;
          while (j < v_nas[m1]) {
            if (v_as[m1][j] == s1 || v_as[m1][j] == s3 || v_as[m1][j] == s4) {
                for (k=j; k<v_nas[m1]-1; k++) {
                  v_as[m1][k] = v_as[m1][k+1];
                  v_av[m1][k] = v_av[m1][k+1];
                }
                v_nas[m1]--;
            } else {
                j++;
            }
          }
          j = 0;
          while (j < v_nas[m4]) {
            if (v_as[m4][j] == s4) {
                for (k=j; k<v_nas[m4]-1; k++) {
                  v_as[m4][k] = v_as[m4][k+1];
                  v_av[m4][k] = v_av[m4][k+1];
                }
                v_nas[m4]--;
            } else {
                j++;
            }
          }
          j = 0;
          while (j < v_nas[m5]) {
            if (v_as[m5][j] == s6) {
                for (k=j; k<v_nas[m5]-1; k++) {
                  v_as[m5][k] = v_as[m5][k+1];
                  v_av[m5][k] = v_av[m5][k+1];
                }
                v_nas[m5]--;
            } else {
                j++;
            }
          }
          j = 0;
          while (j < v_nas[m6]) {
            if (v_as[m6][j] == s8) {
                for (k=j; k<v_nas[m6]-1; k++) {
                  v_as[m6][k] = v_as[m6][k+1];
                  v_av[m6][k] = v_av[m6][k+1];
                }
                v_nas[m6]--;
            } else {
                j++;
            }
          }

          /*  Take the good sides that are attached to m2 and m3 and attach them to m1,
              change one of the side endpoints from m2 (or m3) to m1, and change the
              attached vertex for the other end of the side from m2 (or m3) to m1       */

          for (j=0; j<v_nas[m2]; j++) {
            s = v_as[m2][j];
            if (s_good[s]) {
              if (v_nas[m1] == MAXATTACH)
                bailout("merge_comps.c: need to increase MAXATTACH\n");
              m = v_av[m2][j];
              v_as[m1][v_nas[m1]] = s;
              v_av[m1][v_nas[m1]] = m;
              v_nas[m1]++;
              for (k=0; k<=1; k++)
                if (s_v[s][k] == m2)
                  s_v[s][k] = m1;
              for (k=0; k<v_nas[m]; k++)
                if (v_av[m][k] == m2)
                  v_av[m][k] = m1;
            }
          }
          for (j=0; j<v_nas[m3]; j++) {
            s = v_as[m3][j];
            if (s_good[s]) {
              if (v_nas[m1] == MAXATTACH)
                bailout("merge_comps.c: need to increase MAXATTACH\n");
              m = v_av[m3][j];
              v_as[m1][v_nas[m1]] = s;
              v_av[m1][v_nas[m1]] = m;
              v_nas[m1]++;
              for (k=0; k<=1; k++)
                if (s_v[s][k] == m3)
                  s_v[s][k] = m1;
              for (k=0; k<v_nas[m]; k++)
                if (v_av[m][k] == m3)
                  v_av[m][k] = m1;
            }
          }

          /*  We're done with this facet  */

        }
      }    /* end f-loop over facets */

      /* end of iteration */

  } while (keep_going);


  /*  All small facets have now been eliminated from the merged model  */

  /*  Create mappings from vertex and facet numbers in the merged model prior to
      eliminating small facets to the vertex and facet numbers after elimination  */

  v_final = ivector( 0, nv_in_model-1);
  f_final = ivector( 0, nf_in_model-1);

  nv_final = 0;
  for (m=0; m<nv_in_model; m++)
    if (v_good[m]) {
        v_final[m] = nv_final;
        nv_final++;
    } else {
        v_final[m] = -1;
    }

  nf_final = 0;
  for (f=0; f<nf_in_model; f++)
    if (f_good[f]) {
        f_final[f] = nf_final;
        nf_final++;
    } else {
        f_final[f] = -1;
    }

  /*  Send status to the screen  */

  if (nf_final < nf_in_model) {
    printf("# eliminating %d very small facets\n", nf_in_model - nf_final);
    printf("#\n");
    printf("# final version of merged model has %d vertices and %d facets\n",
           nv_final, nf_final);
    fflush(stdout);
    if (nf_final != 2*nv_final - 4) {
        fprintf( stderr, "Should have %d facets for %d vertices, not %d facets\n",
                 2*nv_final - 4, nv_final, nf_final);
        bailout("invalid model following elimination of small facets\n");
    } else if (nf_final < 4) {
        bailout("too many small facets eliminated, surface is no longer closed\n");
    }
  }
  printf("#\n");


  /*************************************************************************************
    Output block:

    Now that all vertices and facets have been identified, allocate memory for the
    merged model and fill in the necessary values, leaving it to the setupsides and
    setupvertices routines (normally called within read_mod) to finish the job.  Then
    realize the merged model (to get its moments) and write it to disk.
  **************************************************************************************/


  /*  Allocate memory for and initialize the merged model's structure  */

  sprintf(mergedmod.name, "%s", mod->name);   /* write_wf will append ".wf" to the name */
  mergedmod.shape.ncomp = 1;
  mergedmod.shape.comp = (struct comp_t *)
                         calloc( mergedmod.shape.ncomp, sizeof( struct comp_t));
  for (i=0; i<=2; i++) {
    mergedmod.shape.comp[0].off[i].state = 'c';
    mergedmod.shape.comp[0].off[i].val = 0.0;
    mergedmod.shape.comp[0].rot[i].state = 'c';
    mergedmod.shape.comp[0].rot[i].val = 0.0;
  }
  mergedmod.shape.comp[0].type = VERTEX;

  for (i=0; i<=2; i++) {
    mergedmod.shape.comp[0].desc.ver.scalefactor[i].state = (i == 0) ? 'c' : '=';
    for (c=0; c<ncomp; c++)
      if (mod->shape.comp[c].real.scalefactor[i].state == 'f')
        mergedmod.shape.comp[0].desc.ver.scalefactor[i].state = 'f';
    mergedmod.shape.comp[0].desc.ver.scalefactor[i].val = 1.0;
  }

  /*  Assign vertex properties; for now, set each vertex's
      direction cosines to point radially outward           */

  mergedmod.shape.comp[0].desc.ver.nv = nv_final;
  mergedmod.shape.comp[0].desc.ver.v = (struct vertex_t *)
                                       calloc( nv_final, sizeof( struct vertex_t));
  for (m=0; m<nv_in_model; m++) {
    if (v_good[m]) {
      n = v_final[m];
      mergedmod.shape.comp[0].desc.ver.v[n].r.state = 'f';
      mergedmod.shape.comp[0].desc.ver.v[n].r.val = 0.0;
      c = v_old[m][0];
      v = v_old[m][1];
      real = &mod->shape.comp[c].real;
      for (i=0; i<=2; i++) {
        mergedmod.shape.comp[0].desc.ver.v[n].u[i] = real->v[v].x[i];
        mergedmod.shape.comp[0].desc.ver.v[n].a[i] = real->v[v].x[i];
      }
      normalize( mergedmod.shape.comp[0].desc.ver.v[n].u);
    }
  }

  /*  Assign facet properties  */

  mergedmod.shape.comp[0].desc.ver.nf = nf_final;
  mergedmod.shape.comp[0].desc.ver.f = (struct facet_t *)
                                       calloc( nf_final, sizeof( struct facet_t));
  for (f=0; f<nf_in_model; f++)
    if (f_good[f]) {
      n = f_final[f];
      for (j=0; j<=2; j++) {
        m = f_v[f][j];
        mergedmod.shape.comp[0].desc.ver.f[n].v[j] = v_final[m];
      }
    }

  /*  Assign the photometric and spin portions of the original
      model structure to the merged model's structure           */

  mergedmod.photo = mod->photo;
  mergedmod.spin = mod->spin;

  /*  Complete the merged model structure and then realize the model  */

  mergedmod.shape.comp[0].real = mergedmod.shape.comp[0].desc.ver;
  setupsides( &mergedmod.shape.comp[0].real);
  setupvertices( &mergedmod.shape.comp[0].real);
  realize_mod( par, &mergedmod);

  /*  The vertex normals were computed when we realized the model, so now we can
      redefine each vertex's direction cosines to point along the vertex normal.  */

  for (n=0; n<nv_final; n++)
    for (i=0; i<=2; i++)
      mergedmod.shape.comp[0].desc.ver.v[n].u[i] = 
                            mergedmod.shape.comp[0].desc.ver.v[n].n[i];

  /*  Write a wavefront .OBJ description of the merged model's shape to disk
      (unless this is the split action, in which case write the entire model)  */

  if (par->action == SPLIT)
    write_mod( par, &mergedmod);
  else
    write_wf( &mergedmod);

  /*  Clean up storage space  */

  free_ivector( nv_0, 0, ncomp-1);
  free_ivector( nf_0, 0, ncomp-1);
  free_ivector( ns_0, 0, ncomp-1);
  free_ivector( v_in_model, 0, nv_tot-1);
  free_ivector( f_in_model, 0, nf_tot-1);
  free_ivector( s_in_model, 0, ns_tot-1);
  free_ivector( v_naf, 0, nv_merged-1);
  free_imatrix( v_af, 0, nv_merged-1, 0, MAXATTACH-1);
  free_ivector( v_nas, 0, nv_merged-1);
  free_imatrix( v_as, 0, nv_merged-1, 0, MAXATTACH-1);
  free_imatrix( v_av, 0, nv_merged-1, 0, MAXATTACH-1);
  free_imatrix( f_v, 0, nf_merged-1, 0, 2);
  free_imatrix( f_s, 0, nf_merged-1, 0, 2);
  free_imatrix( s_v, 0, ns_merged-1, 0, 1);
  free_ivector( s_nf, 0, ns_merged-1);
  free_imatrix( s_f, 0, ns_merged-1, 0, 1);
  free_ivector( v_new, 0, nv_tot-1);
  free_ivector( f_new, 0, nf_tot-1);
  free_ivector( s_new, 0, ns_tot-1);
  free_imatrix( v_old, 0, nv_merged-1, 0, 1);
  free_imatrix( v_connected, 0, nv_tot-1, 0, nv_tot-1);
  free_matrix( v_distance, 0, nv_tot-1, 0, nv_tot-1);
  free_ivector( attachvec, 0, MAXATTACH-1);
  free_ivector( f_very_small, 0, nf_in_model-1);
  free_ivector( v_good, 0, nv_in_model-1);
  free_ivector( f_good, 0, nf_in_model-1);
  free_ivector( s_good, 0, ns_in_model-1);
  free_ivector( v_final, 0, nv_in_model-1);
  free_ivector( f_final, 0, nf_in_model-1);

}


/*  Given the displacement vectors x1, x2, and x3 for a facet's three corners,
    ordered in the right-hand sense, compute the facet's unit normal vector,
    its base (maximum side length), its height (perpendicular to the base), and
    the index of the side that is the base (0 if x1-x2, 1 if x2-x3, 2 if x3-x1).
    Return the facet area.                                                        */

double facetgeom( double x1[3], double x2[3], double x3[3],
                  double normal[3], double *base, double *height, int *baseindex)
{
  int i;
  double a[3], b[3], area, dist1, dist2, dist3;

  for (i=0; i<=2; i++) {
    a[i] = x2[i] - x1[i];
    b[i] = x3[i] - x2[i];
  }
  area = 0.5*cross( normal, a, b);
  normalize( normal);

  dist1 = distance( x1, x2);
  dist2 = distance( x2, x3);
  dist3 = distance( x3, x1);

  *base = dist1;
  *baseindex = 0;
  if (dist2 > *base) {
    *base = dist2;
    *baseindex = 1;
  }
  if (dist3 > *base) {
    *base = dist3;
    *baseindex = 2;
  }

  if (*base != 0.0)
    *height = 2*area/(*base);
  else
    *height = 0.0;

  return area;
}


/*  Given n_attached vertex numbers in the attachvec vector, two other vertex
    numbers n1 and n2, and an array of flags v_connected that tells whether
    or not a given pair of vertices is connected, check every permutation of
    every combination of vertex numbers within attachvec to see if there
    exists an unbroken set of connections taking us from n1 to n2.  (This
    would imply that it is topologically invalid to create a new side
    connecting n1 to n2, as that would "strand" a third vertex n3 that is
    attached to both n and n2.)  Return 1 if such a path exists, 0 otherwise.  */

int stranded_facet( int n1, int n2, int *attachvec, int n_attached, int **v_connected)
{
  int bad_combination, k, v[MAXATTACH];

  bad_combination = 0;
  k = 1;
  do {

      /*  Check all permutations of all combinations of the n_attached vertices
          within attachvec, taken k vertices at a time without replacement       */

      bad_combination = check_combinations( v, 0, n_attached, 0, k,
                                            n1, n2, attachvec, v_connected);
      k++;
  } while (k <= n_attached && !bad_combination);

  return bad_combination;
}


/*  Given n vertex numbers in the attachvec vector, two other vertex numbers
    n1 and n2, and an array of flags v_connected that tells whether or not a
    given pair of vertices is connected, check every permutation of every
    combination of vertex numbers within attachvec taken kmax vertices at a
    time without replacement to see if there exists an unbroken set of
    connections taking us from n1 to n2.  Return 1 if such a path exists,
    0 otherwise.

    The combinations are generated through recursive calls to this routine;
    with combinations of integers 0 through n-1 taken kmax at a time without
    replacement being constructed in the v vector.  The initial call should
    have start = 0 and k = 0.                                                 */

int check_combinations( int v[], int start, int n, int k, int kmax,
                        int n1, int n2, int *attachvec, int **v_connected)
{
  int bad_permutation, i;

  bad_permutation = 0;

  /*  k here counts through positions in the kmax-element v.
      If k = kmax, then the v is complete and we can use it.  */

  if (k == kmax) {

    /*  Check every permutation of this combination of vertices.  If a bad
        permutation exists, set a flag (bad_permutation = 1) and return it
        through all the recursive calls to this routine and finally back to
        the stranded_facet routine that initially called this routine.       */

    bad_permutation = check_permutations( v, kmax, 0, n1, n2, attachvec, v_connected);
    return bad_permutation;
  }

  /*  For this k'th element of the v, try all start..(n-1)
      elements in that position                             */

  for (i=start; i<n; i++) {

    v[k] = i;

    /*  Recursively generate combinations of integers from i..(n-1)  */

    if (check_combinations( v, i+1, n, k+1, kmax, n1, n2, attachvec, v_connected))
      bad_permutation = 1;
  }

  return bad_permutation;
}


/*  Given that elements 0 through n-1 of the v vector contain a particular
    combination of integers 0 through n-1, recursively generate all
    permutations of these integers.  (The initial call should have i = 0.)
    For each permutation, look at the corresponding series of vertex
    numbers from the attachvec vector plus two other vertex number n1 and
    n2 to see whether or not an unbroken series of connections leading
    from n1 to n2 exists; the v_connected array is a set of flags that
    tells whether or not a given pair of vertices is connected.
    Return 1 if such a path exists, 0 otherwise.                            */

int check_permutations( int v[], int n, int i,
                        int n1, int n2, int *attachvec, int **v_connected)
{
  int bad_permutation, j;

  bad_permutation = 0;

  /*  If we are at the end of the array, we have one permutation we can use  */

  if (i == n) {

      /*  Check whether or not we can traverse an unbroken path from vertex n1
          to the v[0]-th vertex listed in attachvec to the v[1]-th vertex
          listed in attachvec . . . to the v[n-1]-th vertex listed in attachvec
          to vertex n2.  If we can, a bad permutation exists, so set a flag
          (bad_permutation = 1) and return it through all the recursive calls
          to this routine and finally back to the check_combinations routine
          that initially called this routine.                                    */
          
      if (v_connected[n1][attachvec[v[0]]] && v_connected[attachvec[v[n-1]]][n2]) {
          bad_permutation = 1;
          for (j=0; j<n-1; j++)
            if (!v_connected[attachvec[v[j]]][attachvec[v[j+1]]])
              bad_permutation = 0;
      } else {
          bad_permutation = 0;
      }

  } else {

      /*  Recursively explore the permutations starting
          at index i going through index n-1             */

      for (j=i; j<n; j++) {

        /*  Try the array with i and j switched  */

        swap( v, i, j);
        if (check_permutations( v, n, i+1, n1, n2, attachvec, v_connected))
          bad_permutation = 1;

        /*  Swap them back the way they were  */

        swap( v, i, j);
      }
  }

  return bad_permutation;
}


/*  Routine to swap array elements  */

void swap( int v[], int i, int j)
{
  int t;

  t = v[i];
  v[i] = v[j];
  v[j] = t;
}

#undef MAXATTACH
