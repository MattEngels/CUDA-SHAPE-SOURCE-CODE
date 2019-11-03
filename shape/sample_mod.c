/***************************************************************************
                                                               sample_mod.c

Takes an arbitrary (realized) model and produces a "wavefront" format file
with the desired number of vertices.  This output model begins with six
vertices, two each on the x, y, and z axes.  The algorithm then "samples"
the input model by iteratively subdividing the output model's longest side
and projecting that midpoint onto the input model's surface.  The
projection is along the direction from the origin to the midpoint of the
divided side.

The output model is written to a file whose name is the input mod file's
name with either ".samp.wf" appended (if the "sample" action is being used)
or else ".wf" appended (if the "wavefront" action is being used).

Modified 2010 September 1 by CM:
    Add "facetnorm" argument to the rayfacint routine
    Initialize variables to avoid compilation warnings

Modified 2009 August 9 by CM:
    Major revision: rather than always projecting the midpoint of the
        model's longest side radially outward to find the location of a
        new vertex -- an approach that fails for odd-shaped objects whose
        surface can be tangential to the radial direction -- project along
        two other directions as well: the normalized sum of the outward
        normals to the two facets adjoining the longest side; and, if the
        side's orientation isn't too close to radial, the outward direction
        that's perpendicular to the side in the plane defined by the side
        and the radial direction.  For each new vertex, use the projection
        that yields adjacent facets tilted relative to each other as little
        as possible, while also requiring the new vertex to be no closer to
        any previously placed vertex than 1% of the longest side's length.
    As part of implementing the above change, the code now keeps track of
        model sides and not just vertices and facets
    An additional change is that in searching for each new vertex's
        position, trial projections are rejected if they intersect any of
        the sampled model's existing facets; this check slows things down
        but should prevent the final model from having intersecting facets
        (i.e., from being topologically invalid)

Modified 2009 August 3 by CM:
    This routine will no longer be called for the "wavefront" action for
        multiple-component models, so there's no need to adjust the
        output filename depending on the action
    Increase minimum vertex spacing from 1 mm to 1 cm
    Add "tol" argument when calling "rayfacint" routine

Modified 2009 July 1 by CM:
    Check that vertices aren't too close to each other, and stop adding
        vertices once this situation is reached

Modified 2008 July 8 by CM:
    Improve comments
    Allow for the possibility that this routine is called when using the
        "wavefront" action rather than the "sample" action: adjust the
        output filename accordingly
    For each new sampled vertex, initialize vertex displacement to a
        large negative value rather than to a small positive value;
        then, if it's still negative after going through all input facets,
        reset it to a small positive value and print a warning

Modified 2005 January 25 by CM:
    Remove the defunct "CLOSEST" code block: according to Scott, he had
        trouble with it getting into limit cycles

Modified 2004 June 27 by CM:
    Instead of overwriting the input mod file, write to an output file
        whose name is the input mod file's name with ".samp.wf" appended
***************************************************************************/

#include "head.h"

#define UTOL 1.0e-2
#define RTOL 1.0e-14
#define ONE_CENTIMETER 1.0e-5
#define MAXPASSES 100

void initialize_model( struct mod_t *mod, struct vertices_t *wf, int ver_samps,
                       int *nv, int *nf, int *ns);
double longest_side( struct vertices_t *wf, int ns, int *f1, int *f2);
void get_directions( double a[3], double n1[3], double n2[3], double sidevec[3],
                     double u[3][3], int *ntry);
int furthest_intersection( struct mod_t *mod, double a[3], double u[3],
                           struct vertices_t *wf, int nv, int f1, int f2,
                           double x[3], double *mindist, double *mindot);
double get_mindot( struct vertices_t *wf, int f1, int f2, double x[3]);
double get_normal( double normal[3], double x1[3], double x2[3], double x3[3]);
void add_vertex( struct vertices_t *wf, int f1, int f2, int *v, int *nf, int *ns);
int reduce_angles( struct vertices_t *wf);


void sample_mod( struct par_t *par, struct mod_t *mod)
{
  FILE *fp;
  char name[80];
  int v, nf, ns, i, f1, f2, v1, v2, j, j_largest, keep_going, found_intersection,
      ntry, npasses, revised_connection;
  double mindist, mindot, u[3][3], x[3][3], sidevec[3], sidelength, largest_mindot;
  struct vertices_t wf;

  /* Initialize sampled model with 6 vertices at outermost intersection points
   * between the input model and the coordinate axes  */
  initialize_model( mod, &wf, (*par).ver_samps, &v, &nf, &ns);

  /* Add one vertex at a time to the sampled model  */
  keep_going = 1;
  do {
      /* Find two facets f1 and f2 that adjoin the longest side (edge) in the
       * sampled model as it exists so far. Routine longest_side reorders these
       * 2 facets' bounding vertices so that the longest side runs from vertex
       * 0 to vertex 1 for facet f1 and from vertex 1 to vertex 0 for facet f2*/
      sidelength = longest_side( &wf, ns, &f1, &f2);

      /* Set the base displacement for the newest vertex to the midpoint of
       * longest side, and define a vector that points along this side    */
      v1 = wf.f[f1].v[0];
      v2 = wf.f[f1].v[1];
      for (i=0; i<=2; i++) {
        wf.v[v].a[i] = (wf.v[v1].x[i] + wf.v[v2].x[i]) / 2;
        sidevec[i] = wf.v[v2].x[i] - wf.v[v1].x[i];
      }
      normalize( sidevec);

      /* Get the sets of direction cosines that we'll try  */
      get_directions( wf.v[v].a, wf.f[f1].n, wf.f[f2].n, sidevec, u, &ntry);

      /* Try the various sets of direction cosines, looking for most distant
       * intersection point along that ray and computing that point's dis-
       * placement, its minimum distance from any of the previously defined
       * vertices in sampled model, and minimum dot product of outward normals
       * to adjacent facets that are affected by placement of this new vertex*/
      found_intersection = 0;
      largest_mindot = -1.0;
      j_largest = -1;
      for (j=0; j<ntry; j++)
        if (furthest_intersection( mod, wf.v[v].a, u[j], &wf, v, f1, f2,
                                   x[j], &mindist, &mindot)) {
          found_intersection = 1;

          /* Insist the new vertex be no closer to any previously placed vertex
           * than 1% of longest side's length; once this is met, look for the
           * intersection point (i.e., the trial vertex position) for which the
           * sharpest angle between adjacent facets is as small as possible */
          if (mindist > 0.01*sidelength && mindot > largest_mindot) {
            largest_mindot = mindot;
            j_largest = j;
          }
        }

      /* Check if we were successful -- that is, that we found at least one
       * intersection point that met our minimum distance criterion    */
      if (!found_intersection) {
          fprintf( stderr, "WARNING in sample_mod: no intersection,"
                           " stopping at %d vertices\n", v);
          fflush(stderr);
          keep_going = 0;
      } else if (j_largest == -1) {
          fprintf( stderr, "WARNING in sample_mod: vertices too close to each other,"
                           " stopping at %d vertices\n", v);
          fflush(stderr);
          keep_going = 0;
      } else {

          /* All is well: assign total displacement vector for new vertex  */
          for (i=0; i<=2; i++)
            wf.v[v].x[i] = x[j_largest][i];

          /* Add the new vertex, 2 new facets, 3 new sides, and take care of
           * associated bookkeeping chores            */
          add_vertex( &wf, f1, f2, &v, &nf, &ns);
      }

  } while (v < wf.nv && keep_going);

  /* Sampled model is complete, but before writing it to disk, check to see if
   * we can revise vertex connections for adjacent pairs of facets so as to
   * eliminate sharp angles       */
  npasses = 0;
  do {
      revised_connection = reduce_angles( &wf);
      npasses++;
  } while (revised_connection && npasses < MAXPASSES);

  /* Write the sampled model to disk as a wavefront file  */
  sprintf( name, "%s.samp.wf", mod->name);
  FOPEN( fp, name, "w");
  for (i=0; i<v; i++)
    fprintf( fp, "v %e %e %e\n", wf.v[i].x[0], wf.v[i].x[1], wf.v[i].x[2]);
  for (i=0; i<nf; i++)
    fprintf( fp, "f %d %d %d\n", wf.f[i].v[0] + 1, wf.f[i].v[1] + 1,
                 wf.f[i].v[2] + 1);
  fclose( fp);
}

/* Set up initial version of sampled model, with 6 vertices placed at outermost
 * intersection points between input model's surface and coordinate axes    */

void initialize_model( struct mod_t *mod, struct vertices_t *wf, int ver_samps,
                       int *nv, int *nf, int *ns)
{
  int v, i, f;
  double mindist, mindot;

  wf->nv = ver_samps;
  wf->nf = 2*wf->nv - 4;
  wf->ns = 3*wf->nv - 6;
  wf->v = (struct vertex_t *) calloc( wf->nv, sizeof( struct vertex_t));
  wf->f = (struct facet_t *) calloc( wf->nf, sizeof( struct facet_t));
  wf->s = (struct side_t *) calloc( wf->ns, sizeof( struct side_t));
  
  /* Assign direction cosines for vertices of the initial 6-vertex sampled
   * model, and zero out the base displacements  */
  *nv = 6;
  wf->v[0].u[0] =  0.0; wf->v[0].u[1] =  0.0; wf->v[0].u[2] =  1.0;
  wf->v[1].u[0] =  1.0; wf->v[1].u[1] =  0.0; wf->v[1].u[2] =  0.0;
  wf->v[2].u[0] =  0.0; wf->v[2].u[1] =  1.0; wf->v[2].u[2] =  0.0;
  wf->v[3].u[0] = -1.0; wf->v[3].u[1] =  0.0; wf->v[3].u[2] =  0.0;
  wf->v[4].u[0] =  0.0; wf->v[4].u[1] = -1.0; wf->v[4].u[2] =  0.0;
  wf->v[5].u[0] =  0.0; wf->v[5].u[1] =  0.0; wf->v[5].u[2] = -1.0;
  for (v=0; v<6; v++)
    for (i=0; i<=2; i++)
      wf->v[v].a[i] = 0.0;

  /* Assign the bounding vertices for the corresponding eight facets  */
  *nf = 8;
  wf->f[0].v[0] = 0; wf->f[0].v[1] = 1; wf->f[0].v[2] = 2;
  wf->f[1].v[0] = 0; wf->f[1].v[1] = 2; wf->f[1].v[2] = 3;
  wf->f[2].v[0] = 0; wf->f[2].v[1] = 3; wf->f[2].v[2] = 4;
  wf->f[3].v[0] = 0; wf->f[3].v[1] = 4; wf->f[3].v[2] = 1;
  wf->f[4].v[0] = 1; wf->f[4].v[1] = 5; wf->f[4].v[2] = 2;
  wf->f[5].v[0] = 2; wf->f[5].v[1] = 5; wf->f[5].v[2] = 3;
  wf->f[6].v[0] = 3; wf->f[6].v[1] = 5; wf->f[6].v[2] = 4;
  wf->f[7].v[0] = 4; wf->f[7].v[1] = 5; wf->f[7].v[2] = 1;
  
  /* Assign bounding vertices and attached facets for corresponding 12 sides*/
  *ns = 12;
  wf->s[ 0].v[0] = 0; wf->s[ 0].v[1] = 1; wf->s[ 0].f[0] = 0; wf->s[ 0].f[1] = 3;
  wf->s[ 1].v[0] = 0; wf->s[ 1].v[1] = 2; wf->s[ 1].f[0] = 1; wf->s[ 1].f[1] = 0;
  wf->s[ 2].v[0] = 0; wf->s[ 2].v[1] = 3; wf->s[ 2].f[0] = 2; wf->s[ 2].f[1] = 1;
  wf->s[ 3].v[0] = 0; wf->s[ 3].v[1] = 4; wf->s[ 3].f[0] = 3; wf->s[ 3].f[1] = 2;
  wf->s[ 4].v[0] = 1; wf->s[ 4].v[1] = 2; wf->s[ 4].f[0] = 0; wf->s[ 4].f[1] = 4;
  wf->s[ 5].v[0] = 2; wf->s[ 5].v[1] = 3; wf->s[ 5].f[0] = 1; wf->s[ 5].f[1] = 5;
  wf->s[ 6].v[0] = 3; wf->s[ 6].v[1] = 4; wf->s[ 6].f[0] = 2; wf->s[ 6].f[1] = 6;
  wf->s[ 7].v[0] = 4; wf->s[ 7].v[1] = 1; wf->s[ 7].f[0] = 3; wf->s[ 7].f[1] = 7;
  wf->s[ 8].v[0] = 1; wf->s[ 8].v[1] = 5; wf->s[ 8].f[0] = 4; wf->s[ 8].f[1] = 7;
  wf->s[ 9].v[0] = 2; wf->s[ 9].v[1] = 5; wf->s[ 9].f[0] = 5; wf->s[ 9].f[1] = 4;
  wf->s[10].v[0] = 3; wf->s[10].v[1] = 5; wf->s[10].f[0] = 6; wf->s[10].f[1] = 5;
  wf->s[11].v[0] = 4; wf->s[11].v[1] = 5; wf->s[11].f[0] = 7; wf->s[11].f[1] = 6;

  /* Assign the bounding sides for the eight facets  */
  wf->f[0].s[0] =  0; wf->f[0].s[1] =  4; wf->f[0].s[2] =  1;
  wf->f[1].s[0] =  1; wf->f[1].s[1] =  5; wf->f[1].s[2] =  2;
  wf->f[2].s[0] =  2; wf->f[2].s[1] =  6; wf->f[2].s[2] =  3;
  wf->f[3].s[0] =  3; wf->f[3].s[1] =  7; wf->f[3].s[2] =  0;
  wf->f[4].s[0] =  4; wf->f[4].s[1] =  8; wf->f[4].s[2] =  9;
  wf->f[5].s[0] =  5; wf->f[5].s[1] =  9; wf->f[5].s[2] = 10;
  wf->f[6].s[0] =  6; wf->f[6].s[1] = 10; wf->f[6].s[2] = 11;
  wf->f[7].s[0] =  7; wf->f[7].s[1] = 11; wf->f[7].s[2] =  8;

  /* For each of the 6 vertices, find intersection point (along corresponding
   * direction cosines) with each facet in input model's vertex realization,
   * and choose the intersection point most distant from origin to be the total
   * displacement vector for this vertex             */
  for (v=0; v<6; v++)
    if (!furthest_intersection( mod, wf->v[v].a, wf->v[v].u, wf, v, -1, -1,
                                wf->v[v].x, &mindist, &mindot)) {
      fprintf( stderr, "No intersection for vertex %d\n", v);
      bailout("sample_mod.c: can't initialize sampled model\n");
    }

  /* Compute the outward unit normals for the eight facets  */
  for (f=0; f<8; f++)
    facnrm( *wf, f); 
}


/* Routine longest_side finds the longest side (edge) of the input vertex
 * realization wf, which has nf facets; its return value is the length of that
 * side. The facet numbers of the 2 adjoining facets are passed back as parame-
 * ters f1 and f2. The bounding vertices of these 2 facets are reordered so the
 * longest side runs from vertex 0 to vertex 1 of facet f1 and from vertex 1 to
 * vertex 0 of facet f2.                       */

double longest_side( struct vertices_t *wf, int ns, int *f1, int *f2)
{
  int s, v1, v2, v3, v4, smax, j;
  double maxlength, sidelength;

  /* Initialize variables to avoid compilation warnings  */
   smax = v3 = v4 = 0;

  /* Loop through all model sides, computing each length to find longest side*/
  maxlength = 0.0;
  for (s=0; s<ns; s++) {
    v1 = wf->s[s].v[0];
    v2 = wf->s[s].v[1];
    sidelength = distance( wf->v[v1].x, wf->v[v2].x);
    if (sidelength > maxlength) {
      maxlength = sidelength;
      smax = s;
    }
  }

  /* Get the 2 facets f1 and f2 that are attached to longest side  */
  *f1 = wf->s[smax].f[0];
  *f2 = wf->s[smax].f[1];

  /* Get the 2 vertices v1 and v2 attached to longest side: for facet f1 the
   * vertex order is v1 to v2 to v3 (where v3 is the vertex opposite the
   * longest side) or some cyclical permutation, while for facet f2 the vertex
   * order is v2 to v1 to v4 or some permutation    */
  v1 = wf->s[smax].v[0];
  v2 = wf->s[smax].v[1];
  if ((wf->f[(*f1)].v[0] == v2 && wf->f[(*f1)].v[1] == v1) ||
      (wf->f[(*f1)].v[1] == v2 && wf->f[(*f1)].v[2] == v1) ||
      (wf->f[(*f1)].v[2] == v2 && wf->f[(*f1)].v[0] == v1)    ) {
    v1 = wf->s[smax].v[1];
    v2 = wf->s[smax].v[0];
  }

  /* Get vertices v3 and v4 (see above)  */
  for (j=0; j<=2; j++)
    if (wf->f[(*f1)].v[j] != v1 && wf->f[(*f1)].v[j] != v2)
      v3 = wf->f[(*f1)].v[j];
  for (j=0; j<=2; j++)
    if (wf->f[(*f2)].v[j] != v1 && wf->f[(*f2)].v[j] != v2)
      v4 = wf->f[(*f2)].v[j];

  /* Cyclically permute the corner vertices of facets f1 and f2 so the longest
   * side runs from vertex 0 to vertex 1 for facet f1 and from vertex 1 to
   * vertex 0 for facet f2          */
  wf->f[(*f1)].v[0] = v1;
  wf->f[(*f1)].v[1] = v2;
  wf->f[(*f1)].v[2] = v3;
  wf->f[(*f2)].v[0] = v2;
  wf->f[(*f2)].v[1] = v1;
  wf->f[(*f2)].v[2] = v4;
  
  return maxlength;
}


/* Given a base displacement a and a ray pointing away from that base point
 * along a specified set of direction cosines u, find the intersection point
 * with each facet in the input model's vertex realization, and choose the
 * intersection point that is furthest from the base point (taking direction
 * into account, not just the absolute value of the distance). Pass back the
 * displacement of the intersection point.
 * If facet numbers f1 and f2 are nonnegative, they are the 2 facets in the
 * sampled model that adjoin the side that is being subdivided (i.e., base dis-
 * placement a is the midpoint of the side common to f1 and f2). In this case,
 * also pass back minimum distance from intersection point to any vertex in the
 * sampled model, and minimum dot product between outward normals to 8 pairs of
 * adjacent facets - the 4 facets that adjoin f1 and f2 and the 4 facets that
 * would be created from f1 and f2 if a new vertex were placed at this inter-
 * section point. Return 0 if no intersection is found, 1 otherwise.       */

int furthest_intersection( struct mod_t *mod, double a[3], double u[3],
                           struct vertices_t *wf, int nv, int f1, int f2,
                           double x[3], double *mindist, double *mindot)
{
  int ncomp, c, nf, f, fs, valid_connection, i, v;
  double rmax, r, s, t, rs;
  struct vertices_t *real;

  /* Check every component of the input model  */
  rmax = -HUGENUMBER;
  ncomp = mod->shape.ncomp;
  for (c=0; c<ncomp; c++) {
    real = &mod->shape.comp[c].real;
    nf = real->nf;

    /* Check every facet of this component's vertex realization, searching for
     * the intersection most distant along specified direction cosines      */
    for (f=0; f<nf; f++)
      if (rayfacint( &r, &s, &t, u, a, 
                     real->v[ real->f[f].v[0] ].x,
                     real->v[ real->f[f].v[1] ].x,
                     real->v[ real->f[f].v[2] ].x,
                     real->f[f].n, 0.0))
        if (r > rmax) {
          /* Check vector from base displacement to this intersection doesn't
           * cross any facets of the sampled model  */
          valid_connection = 1;
          fs = 0;
          while (valid_connection && fs < wf->nf) {
            if (rayfacint( &rs, &s, &t, u, a, 
                           wf->v[ wf->f[fs].v[0] ].x,
                           wf->v[ wf->f[fs].v[1] ].x,
                           wf->v[ wf->f[fs].v[2] ].x,
                           wf->f[fs].n, 0.0))
              if ((rs > RTOL && rs < r - RTOL) || (rs < -RTOL && rs > r + RTOL))
                valid_connection = 0;
            fs++;
          }
          if (valid_connection)
            rmax = r;
        }
  }
  /* If no intersection was found, return with dummy values  */
  if (rmax == -HUGENUMBER) {
    rmax = ONE_CENTIMETER;
    for (i=0; i<=2; i++)
      x[i] = a[i] + rmax*u[i];
    *mindist = *mindot = -HUGENUMBER;
    return 0;
  }

  /* Assign the total displacement vector for this intersection point  */
  for (i=0; i<=2; i++)
    x[i] = a[i] + rmax*u[i];

  if (f1 >= 0 && f2 >= 0) {

      /* Find minimum distance from intersection point to any vertex found so
       * far for the sampled model       */
      *mindist = HUGENUMBER;
      for (v=0; v<nv; v++)
        *mindist = MIN( *mindist, distance( x, wf->v[v].x));

      /* Find minimum dot product between adjoining facets that would be
       * created/affected if a new vertex were created at this the
       * intersection point                               */
      *mindot = get_mindot( wf, f1, f2, x);

  } else {
      /* Pass back dummy values if we're just initializing the model  */
      *mindist = *mindot = -HUGENUMBER;
  }
  return 1;
}


/* Given 2 adjoining facets f1 and f2 whose common side s is going to be bi-
 * sected to form a new vertex, find the sharpest angle between adjoining
 * facets - that is, the minimum dot product between outward facet normals
 * that would result from creating this new vertex at position x          */

double get_mindot( struct vertices_t *wf, int f1, int f2, double x[3])
{
  int i, v1, v2, v3, v4, k, n, s, s1, s2, s3, s4, f;
  double norm1[3], norm2[3], norm3[3], norm4[3], norm5[3], norm6[3],
         norm7[3], norm8[3], mindot;

  /* Initialize variables to avoid compilation warnings  */
   s1 = s2 = s3 = s4 = 0;

  /* Get vertices v1 and v2 that define the side we're bisecting, vertex v3
   * that is the 3rd vertex of facet f1, and vertex v4 that is the 3rd
   * vertex of facet f2                             */
  v1 = wf->f[f1].v[0];
  v2 = wf->f[f1].v[1];
  v3 = wf->f[f1].v[2];
  v4 = wf->f[f2].v[2];

  /* Look at facet f1 and find the side s that we're bisecting, side s1 that is
   * opposite v1, and side s2 that is opposite v2  */
  for (n=0; n<=2; n++) {
    k = wf->f[f1].s[n];
    if ((wf->s[k].v[0] == v1 && wf->s[k].v[1] == v2) ||
        (wf->s[k].v[1] == v1 && wf->s[k].v[0] == v2)    )
      s = k;
    else if (wf->s[k].v[0] != v1 && wf->s[k].v[1] != v1)
      s1 = k;
    else
      s2 = k;
  }

  /* Look at facet f2 and find side s3 that is opposite v2 and side s4 that is
   * opposite v1                        */
  for (n=0; n<=2; n++) {
    k = wf->f[f2].s[n];
    if (wf->s[k].v[0] != v2 && wf->s[k].v[1] != v2)
      s3 = k;
    else if (wf->s[k].v[0] != v1 && wf->s[k].v[1] != v1)
      s4 = k;
  }

  /* Get normals to the 4 facets (other than f1 and f2) attached to sides s1,
   * s2, s3, and s4              */
  f = (wf->s[s1].f[0] == f1) ? wf->s[s1].f[1] : wf->s[s1].f[0];
  for (i=0; i<=2; i++)
    norm1[i] = wf->f[f].n[i];
  f = (wf->s[s2].f[0] == f1) ? wf->s[s2].f[1] : wf->s[s2].f[0];
  for (i=0; i<=2; i++)
    norm2[i] = wf->f[f].n[i];
  f = (wf->s[s3].f[0] == f2) ? wf->s[s3].f[1] : wf->s[s3].f[0];
  for (i=0; i<=2; i++)
    norm3[i] = wf->f[f].n[i];
  f = (wf->s[s4].f[0] == f2) ? wf->s[s4].f[1] : wf->s[s4].f[0];
  for (i=0; i<=2; i++)
    norm4[i] = wf->f[f].n[i];

  /* Get normals to the 4 facets that would be made from f1 and f2 were we to
   * move the midpoint of side s to the chosen intersection point  */
  get_normal( norm5,             x, wf->v[v2].x, wf->v[v3].x);
  get_normal( norm6, wf->v[v1].x,             x, wf->v[v3].x);
  get_normal( norm7,             x, wf->v[v1].x, wf->v[v4].x);
  get_normal( norm8, wf->v[v2].x,             x, wf->v[v4].x);

  /* Compute dot products of the normals to each adjoining pair of facets and
   * keep the minimum value (representing the sharpest angle)              */
  mindot = MIN( dot( norm1, norm5), dot(norm2, norm6));
  mindot = MIN( mindot, dot( norm3, norm7));
  mindot = MIN( mindot, dot( norm4, norm8));
  mindot = MIN( mindot, dot( norm5, norm6));
  mindot = MIN( mindot, dot( norm6, norm7));
  mindot = MIN( mindot, dot( norm7, norm8));
  mindot = MIN( mindot, dot( norm8, norm5));

  return mindot;
}


/* Given displacement vectors x1, x2, and x3 for a facet's 3 corners, ordered
 * in right-hand sense, compute facet's outward unit normal vector.
 * Return the facet area.                                             */

double get_normal( double normal[3], double x1[3], double x2[3], double x3[3])
{
  int i;
  double a[3], b[3], area;

  for (i=0; i<=2; i++) {
    a[i] = x2[i] - x1[i];
    b[i] = x3[i] - x2[i];
  }
  area = 0.5*cross( normal, a, b);
  normalize( normal);

  return area;
}


/* Get all sets of direction cosines that could be tried, plus an integer
 * telling how many of them to try          */

void get_directions( double a[3], double n1[3], double n2[3], double sidevec[3],
                     double u[3][3], int *ntry)
{
  int i;
  double testmag;
  
  /* The 1st set of direction cosines we'll try defines a ray that points
   * radially outward from the longest side's midpoint  */
  for (i=0; i<=2; i++)
    u[0][i] = a[i];
  normalize( u[0]);

  /* Since this produces poor results if the ray is tangential to some part of
   * input model's surface, we'll try 2 additional directions. One is the
   * normalized sum of the outward normals for the 2 facets attached to
   * the longest side.                                           */
  for (i=0; i<=2; i++)
    u[1][i] = n1[i] + n2[i];
  normalize( u[1]);

  /* The other direction points outward perpendicular to longest side in the
   * plane defined by the side and the base displacement. This approach will
   * fail if the side points nearly radially, so check for this and don't try
   * this direction if that's the case.                 */
  testmag = cross( u[2], u[0], sidevec);
  if (testmag > UTOL) {
      cross( u[2], sidevec, u[2]);
      normalize( u[2]);
      *ntry = 3;
  } else {
      *ntry = 2;
  }
}

/* Add a new vertex, 2 new facets, 3 new sides to sampled model, take care of
 * associated tasks (updating numbers of attached facets to each model side  */

void add_vertex( struct vertices_t *wf, int f1, int f2, int *v, int *nf, int *ns)
{
  int n, k, s, s1, s2;

  /* Initialize variables to avoid compilation warnings  */
   s = s1 = s2 = 0;

  /* Specify vertices for 2 new facets and compute outward normals        */
  wf->f[(*nf)  ].v[0] = (*v);
  wf->f[(*nf)  ].v[1] = wf->f[f1].v[1];
  wf->f[(*nf)  ].v[2] = wf->f[f1].v[2];
  wf->f[(*nf)+1].v[0] = (*v);
  wf->f[(*nf)+1].v[1] = wf->f[f2].v[1];
  wf->f[(*nf)+1].v[2] = wf->f[f2].v[2];
  facnrm( *wf, (*nf));
  facnrm( *wf, (*nf)+1);

  /* Specify vertices and attached facets for 3 new sides  */
  wf->s[(*ns)  ].v[0] = (*v);
  wf->s[(*ns)  ].v[1] = wf->f[f1].v[2];
  wf->s[(*ns)  ].f[0] = f1;
  wf->s[(*ns)  ].f[1] = (*nf);
  wf->s[(*ns)+1].v[0] = wf->f[f2].v[0];
  wf->s[(*ns)+1].v[1] = (*v);
  wf->s[(*ns)+1].f[0] = f2;
  wf->s[(*ns)+1].f[1] = (*nf);
  wf->s[(*ns)+2].v[0] = (*v);
  wf->s[(*ns)+2].v[1] = wf->f[f2].v[2];
  wf->s[(*ns)+2].f[0] = f2;
  wf->s[(*ns)+2].f[1] = (*nf) + 1;

  /* Find the side s that we're bisecting  */
  for (n=0; n<=2; n++) {
    k = wf->f[f1].s[n];
    if ((wf->s[k].v[0] == wf->f[f1].v[0] && wf->s[k].v[1] == wf->f[f1].v[1]) ||
        (wf->s[k].v[1] == wf->f[f1].v[0] && wf->s[k].v[0] == wf->f[f1].v[1])    )
      s = k;
  }

  /* Find side s1 that used to be attached to facet f1 (the side that was
   * opposite vertex 0, not the side we bisected) that is now attached to facet
   * nf instead; similarly side s2 that used to be opposite vertex 0 of facet
   * f2 is now attached to facet nf+1        */
  for (n=0; n<=2; n++) {
    k = wf->f[f1].s[n];
    if (wf->s[k].v[0] != wf->f[f1].v[0] && wf->s[k].v[1] != wf->f[f1].v[0])
      s1 = k;
    k = wf->f[f2].s[n];
    if (wf->s[k].v[0] != wf->f[f2].v[0] && wf->s[k].v[1] != wf->f[f2].v[0])
      s2 = k;
  }

  /* Assign bounding sides to the new facets  */
  wf->f[(*nf)  ].s[0] = (*ns) + 1;
  wf->f[(*nf)  ].s[1] = s1;
  wf->f[(*nf)  ].s[2] = (*ns);
  wf->f[(*nf)+1].s[0] = s;
  wf->f[(*nf)+1].s[1] = s2;
  wf->f[(*nf)+1].s[2] = (*ns) + 2;

  /* Change one vertex for the side we bisected, and add a new adjoining facet
   * to that side   */
  wf->s[s].v[0] = wf->f[f1].v[0];
  wf->s[s].v[1] = (*v);
  wf->s[s].f[0] = f1;
  wf->s[s].f[1] = (*nf) + 1;

  /* Change one vertex for each of the 2 facets that adjoined bisected side,
   * and update the outward normals to those facets  */
  wf->f[f1].v[1] = (*v);
  wf->f[f2].v[1] = (*v);
  facnrm( *wf, f1);
  facnrm( *wf, f2);

  /* Fix the attached facets for sides s1 and s2 (see comment above)  */
  if (wf->s[s1].f[0] == f1)
    wf->s[s1].f[0] = (*nf);
  else
    wf->s[s1].f[1] = (*nf);
  if (wf->s[s2].f[0] == f2)
    wf->s[s2].f[0] = (*nf) + 1;
  else
    wf->s[s2].f[1] = (*nf) + 1;

  /* Fix the bounding sides for facets f1 and f2  */
  for (n=0; n<=2; n++) {
    if (wf->f[f1].s[n] == s1)
      wf->f[f1].s[n] = (*ns);
    if (wf->f[f2].s[n] == s)
      wf->f[f2].s[n] = (*ns) + 1;
    else if (wf->f[f2].s[n] == s2)
      wf->f[f2].s[n] = (*ns) + 2;
  }

  /* Increment numbers of vertices, facets, and sides in the model  */
  (*v)++;
  (*nf) += 2;
  (*ns) += 3;

  if ((*v) % 100 == 0)
    printf("#     -- found sampled vertex %d\n", *v);
}

/* Check all pairs of adjoining facets to see if the connections between those
 * 4 vertices can be revised so the angle between facets is significantly
 * reduced.  The revision entails replacing the common side with a new side
 * that connects the 2 vertices that aren't part of the common side.  This
 * routine is a modified version of code found in the "optimization" block of
 * merge_comps.c.  The return value is 1 if any connections were revised,
 * 0 otherwise.              */

int reduce_angles( struct vertices_t *wf)
{
  int revised_connection, s, f1, f2, v1, v2, v3, v4, i, j, k, s1, s2, s3, s4,
      af1, af2, af3, af4;
  double norm1b[3], norm2b[3], meandot, meandotb;

  /* Initialize variables to avoid compilation warnings  */
   s1 = s2 = s3 = s4 = v4 = 0;

  /* Look at each side to see if we can redefine the 2 attached facets so as to
   * make them significantly more coplanar                       */
  revised_connection = 0;

  for (s=0; s<wf->ns; s++) {
    /* s is the side common to facets f1 and f2; it runs from vertex v1 to v2
     * for f1 and from v2 to v1 for f2.  The 3rd vertex in f1 is v3 and the 3rd
     * vertex in f2 is v4.                      */
    f1 = wf->s[s].f[0];
    if ((wf->f[f1].v[0] == wf->s[s].v[0] && wf->f[f1].v[1] == wf->s[s].v[1]) ||
        (wf->f[f1].v[0] == wf->s[s].v[1] && wf->f[f1].v[1] == wf->s[s].v[0])    ) {
        v1 = wf->f[f1].v[0];
        v2 = wf->f[f1].v[1];
        v3 = wf->f[f1].v[2];
    } else if ((wf->f[f1].v[1] == wf->s[s].v[0] && wf->f[f1].v[2] == wf->s[s].v[1]) ||
               (wf->f[f1].v[1] == wf->s[s].v[1] && wf->f[f1].v[2] == wf->s[s].v[0])    ) {
        v1 = wf->f[f1].v[1];
        v2 = wf->f[f1].v[2];
        v3 = wf->f[f1].v[0];
    } else {
        v1 = wf->f[f1].v[2];
        v2 = wf->f[f1].v[0];
        v3 = wf->f[f1].v[1];
    }
    f2 = wf->s[s].f[1];
    for (j=0; j<=2; j++)
      if (wf->f[f2].v[j] != v1 && wf->f[f2].v[j] != v2 && wf->f[f2].v[j] != v3)
        v4 = wf->f[f2].v[j];

    /* Find the other 4 sides of facets f1 and f2: f1 runs from side s to s1 to
     * s2 and f2 runs from side s to s3 to s4       */
    for (j=0; j<=2; j++) {
      k = wf->f[f1].s[j];
      if ((wf->s[k].v[0] == v2 && wf->s[k].v[1] == v3) ||
          (wf->s[k].v[1] == v2 && wf->s[k].v[0] == v3)    )
        s1 = k;
      else if ((wf->s[k].v[0] == v3 && wf->s[k].v[1] == v1) ||
               (wf->s[k].v[1] == v3 && wf->s[k].v[0] == v1)    )
        s2 = k;
    }
    for (j=0; j<=2; j++) {
      k = wf->f[f2].s[j];
      if ((wf->s[k].v[0] == v1 && wf->s[k].v[1] == v4) ||
          (wf->s[k].v[1] == v1 && wf->s[k].v[0] == v4)    )
        s3 = k;
      else if ((wf->s[k].v[0] == v4 && wf->s[k].v[1] == v2) ||
               (wf->s[k].v[1] == v4 && wf->s[k].v[0] == v2)    )
        s4 = k;
    }

    /* Find the 4 other facets attached to those 4 sides: facet af1 is attached
     * to side s1, etc.                    */
    af1 = (wf->s[s1].f[0] == f1) ? wf->s[s1].f[1] : wf->s[s1].f[0];
    af2 = (wf->s[s2].f[0] == f1) ? wf->s[s2].f[1] : wf->s[s2].f[0];
    af3 = (wf->s[s3].f[0] == f2) ? wf->s[s3].f[1] : wf->s[s3].f[0];
    af4 = (wf->s[s4].f[0] == f2) ? wf->s[s4].f[1] : wf->s[s4].f[0];

    /* If af2 and af3 are the same facet, v3 and v4 already have a direct
     * connection, so we can't even consider giving them a 2nd connection by
     * revising f1 and f2 - and similarly for af4 and af1                 */
    if (af2 != af3 && af4 != af1) {

      /* Compute mean dot product of the outward normals to the facets attached
       * to sides s, s1, s2, s3, and s4  */
      meandot = (dot( wf->f[f1].n, wf->f[f2].n)  +
                 dot( wf->f[f1].n, wf->f[af1].n) +
                 dot( wf->f[f1].n, wf->f[af2].n) +
                 dot( wf->f[f2].n, wf->f[af3].n) +
                 dot( wf->f[f2].n, wf->f[af4].n)   ) / 5;

      /* Compute outward unit normals to facets f1 and f2 as they would be if
       * we redefined the 2 facets such that their common side s ran from
       * vertex v3 to v4 rather than from v1 to v2                              */
      get_normal( norm1b, wf->v[v4].x, wf->v[v3].x, wf->v[v1].x);
      get_normal( norm2b, wf->v[v3].x, wf->v[v4].x, wf->v[v2].x);

      /* Compute the mean dot product for this revised situation  */
      meandotb = (dot( norm1b, norm2b)         +
                  dot( norm1b, wf->f[af1].n) +
                  dot( norm1b, wf->f[af2].n) +
                  dot( norm2b, wf->f[af3].n) +
                  dot( norm2b, wf->f[af4].n)   ) / 5;

      /* Revise facets if it would significantly increase the mean dot product
       * (i.e., reduce the mean angle between adjoining facets)  */
      if (meandotb > meandot + 0.01) {

        /* Redefine the facets: change common side s to run from v3 to v4, and
         * reconnect vertices v1, v2, v3, and v4 such that facet f1 runs from
         * v4 to v3 to v1 and facet f2 runs from v3 to v4 to v2.                */
        revised_connection = 1;

        /* Redefine side s to go from v3 to v4 rather than from v1 to v2  */
        wf->s[s].v[0] = v3;
        wf->s[s].v[1] = v4;

        /* Reassign vertices and sides for facets f1 and f2  */
        wf->f[f1].v[0] = v4;
        wf->f[f1].v[1] = v3;
        wf->f[f1].v[2] = v1;
        wf->f[f1].s[0] = s;
        wf->f[f1].s[1] = s2;
        wf->f[f1].s[2] = s3;
        wf->f[f2].v[0] = v3;
        wf->f[f2].v[1] = v4;
        wf->f[f2].v[2] = v2;
        wf->f[f2].s[0] = s;
        wf->f[f2].s[1] = s4;
        wf->f[f2].s[2] = s1;

        /* Revise outward unit normals to facets f1 and f2  */
        for (i=0; i<=2; i++) {
          wf->f[f1].n[i] = norm1b[i];
          wf->f[f2].n[i] = norm2b[i];
        }

        /* Reassign facets attached to sides s1 and s3 (nothing changes for s2 and s4)  */
        if (wf->s[s1].f[0] == f1)
          wf->s[s1].f[0] = f2;
        else
          wf->s[s1].f[1] = f2;

        if (wf->s[s3].f[0] == f2)
          wf->s[s3].f[0] = f1;
        else
          wf->s[s3].f[1] = f1;
      }
    }
  }
  return revised_connection;
}

#undef UTOL
#undef RTOL
#undef ONE_CENTIMETER
#undef MAXPASSES
