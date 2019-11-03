/*****************************************************************************************
                                                                            realize_mod.c

Takes a struct mod_t model and "realizes" its components as polyhedral solids made up of
triangular facets.

Modified 2016 July 9 by Matthias Engels:
	Adapted for use with shape-cuda.
------------------------------------------------------------------------------------------
Modified 2014 April 26 by CM:
    Increase the minimum permitted value of the highest-order coefficient in the cubic
        equation that locates an ovoid vertex: if the coefficient is smaller than this
        minimum, treat it as if it's zero and solve a quadratic equation instead

Modified 2014 March 22 by CM:
    Relax the tolerance for finding a valid ovoid vertex position

Modified 2014 March 10 by CM:
    Guard against roundoff problems when computing vertex positions for ovoid components
        with very small |k|

Modified 2014 February 10 by CM:
    Implement multiple radar and optical scattering laws

Modified 2013 August 28 by CM:
    Set the bad diameter flag for harmonic components with tiny or negative vertex
        displacements, and for harmonic and vertex components with tiny or negative
        "scale factor" values

Modified 2013 June 2 by CM:
    In the cubic_realroot routine, initialize nrealroots to avoid compilation warning
    Fix a comment

Modified 2013 May 20 by CM:
    Implement ovoid shape components

Modified 2012 July 4 by CM:
    Add test in "realize_coordinates" routine to avoid compilation warning

Modified 2011 September 2 by CM:
    Bug fix: the "check_surface" routine makes use of facet normals when identifying
        active vs. inactive vertices and facets, but facet normals weren't being computed
        until *after* check_surface was called
    Make the code more modular (and address the above bug) by introducing the
        "realize_coordinates" and "compute_moments" routines, as per the version of
        realize_mod in the SHERMAN package
    Store the area and the centroid coordinates of each facet
    Add "harmlambert" optical scattering law (compute facet angular coordinates)

Modified 2010 September 1 by CM:
    Add "facetnorm" argument to the rayfacint routine

Modified 2010 June 1 by CM:
    Change "scalefactor" parameter from a scalar to a 3-component vector

Modified 2010 March 19 by CM:
    Implement '=' state for vertex deviations

Modified 2009 November 15 by CM:
    In the "check_surface" routine, eliminate an unused variable and fix
        a couple of ambiguous nested if-then-else statements

Modified 2009 August 3 by CM:
    For the "harmlommel" "harmhapke" "harmkaas" and "harmcosine_diff"
        inhomogeneous scattering laws, compute the spherical coordinates
        (theta and phi) of each facet after each component's rotational and
        translational offsets have been applied rather than before, so that
        these laws can be used for multiple-component models
    For multiple-component models, use a more careful method (already used
        for facets) to determine which vertices are on the model's surface;
        also, for both vertices and facets, allow for a bit of roundoff
        error in this determination by adding a tolerance argument to the
        "rayfacint" routine
    For multiple-component models, determine the new "act" (active) flag
        for each model side
    For multiple-component models, fix a bug in computing the center of mass
        for individual components

Modified 2009 July 5 by CM:
    Turn each component's rotational offsets into a rotation matrix here
        rather than in the "read_mod" routine, in case the offsets are
        being allowed to float

Modified 2009 July 1 by CM:
    Add "check_surface" routine that determines which facets of a
        multiple-component model lie on the model's surface rather than
        interior to the model
    For multiple-component models, when computing the area and the moments
        of the overall model, ignore facets that lie interior to the model

Modified 2009 April 3 by CM:
    Fix slight bug in defining function a[i] = 1/radius^2 when a/b or b/c
        is tiny or negative for ellipsoid components
    Initialize the "baddiam_logfactor" parameter and set its value when
        2a, a/b, or b/c is tiny or negative for ellipsoid components

Modified 2007 August 10 by CM:
    Eliminate unused variable

Modified 2007 January 8 by CM:
    Define "scalefactor" state for vertex realizations of ellipsoid and
        harmonic components, not just its value

Modified 2006 October 1 by CM:
    Add "scalefactor" to harmonic and vertex shape structures
    Replace ellipsoid diameters D with two_a, a_over_b, b_over_c

Modified 2005 September 6 by CM:
    Add computation of facet angular coordinates for use with harmonic
        scattering laws

Modified 2005 August 17 by CM:
    Move computation of spherical harmonic functions afactor and bfactor
        from here to read_mod.c, so that it can be done just once per fit

Modified 2005 February 28 by CM:
    Initialize the "baddiam" parameter (flag indicating tiny or negative
        ellipsoid diameters) to 0 here rather than in bestfit.c so that
        it can be used for actions other than "fit"

Modified 2004 August 23 by CM:
    Eliminated newtheta and oldcostheta variables and THETATOL constant,
        since they weren't actually being used (i.e., the test in which
        they were included was always true)

Modified 2003 April 17 by CM:
    Added computation of component and model moments; this used to
        be done in function penalties (but wasn't always being done)
    Added code to cope with tiny or negative ellipsoid diameters;
        as a result, must now pass the model's parameter structure
        as an argument to realize_mod
    Added surface area computation for components and for the full model
 *****************************************************************************************/

#include "head.h"

#define HAIRWIDTH 1.0e-7
#define SMALLRATIO 0.01
#define SMALLOVOIDK1 0.01
#define SMALLOVOIDK2 1.0e-6
#define OVOIDTOL 1.0e-6
#define MAXEDGE 100
#define EDGETOL 1.0e-14
#define RTOL 1000*EDGETOL
#define SMALLCOEFF3 1.0e-5

void realize_coordinates( struct par_t *par, struct mod_t *mod);
void check_surface( struct mod_t *mod);
void compute_moments( struct mod_t *mod);
int cubic_realroots( double *coeff, double *realroot);


void realize_mod( struct par_t *par, struct mod_t *mod)
{

	/*  We need to realize each model component as a polyhedral solid with
      triangular facets.  The first step is to call realize_coordinates,
      which computes the displacement of each vertex in this realization,
      represented as a base displacement plus a vertex deviation (either
      positive or negative) along a specified set of direction cosines.
      Additionally, for each facet it computes the outward unit normal,
      the area, the mean coordinates of the corner vertices, and (for
      some scattering laws) the corresponding angular coordinates.        */

	realize_coordinates( par, mod);

	/*  For multiple-component models, figure out which facets lie on
      the model's surface and which fall within some other component;
      such facets will have their "act" (active) flag reset to zero.   */

	check_surface( mod);

	/*  Compute the area and moments (volume, center of mass, and
      inertia tensor) of each component and of the overall model  */

	compute_moments( mod);
}


/*  Compute the vertex coordinates and (if necessary) facet angular coordinates
    for each component of the model's vertex realization                         */

void realize_coordinates( struct par_t *par, struct mod_t *mod)
{
	int c, nv, nf, i, j, k, l, m, L, v_mirror, harmonic_scatlaw, ilaw, f, naf,
	nrealroots;
	double x[3], a[3], den, r, diam, diamratio, n[3], a_radius, a_over_b, b_over_c,
	k_asym, x0term, numer, denom, x0, a_over_c, h, alpha0, u_x, coeff[4],
	realroot[3], goodroot, x_over_a;
	struct vertices_t *real;

	/*  Loop over all model components, realizing each one as a polyhedral solid
      with triangular facets.  Compute the displacement of each vertex in this
      realization, represented as a base displacement plus a vertex deviation
      (either positive or negative) along a specified set of direction cosines.  */

	/*  Initialize the flag for tiny or negative ellipsoid diameters  */

	(*par).baddiam = 0;
	(*par).baddiam_logfactor = 0.0;

	/*  Loop over all model components, realizing each one as a polyhedral solid
      with triangular facets.  Compute the displacement of each vertex in this
      realization, represented as a base displacement plus a vertex deviation
      (either positive or negative) along a specified set of direction cosines.  */

	for (c=0; c<mod->shape.ncomp; c++) {

		real = &mod->shape.comp[c].real;
		nv = real->nv;
		nf = real->nf;

		/*  Check what type of component this is and
        create the corresponding vertex realization.  */

		switch (mod->shape.comp[c].type) {
		case ELLIPSE:

			/*
            In order to avoid negative diameters, or even incredibly
            small positive diameters, adjust the function
            a[i] = 1/radius[i]^2 so that it monotonically increases
            as diameter[i] decreases through zero and beyond, rather
            than being symmetric about zero diameter.

            Also set a "baddiam" flag when any diameter is very small
            or negative, so that extra penalties can later be applied
            to this model.
			 */

			diam = mod->shape.comp[c].desc.ell.two_a.val;
			if (diam > HAIRWIDTH) {
				a[0] = 2.0/diam; /* 1/radii */
			} else {
				a[0] = (2.0/HAIRWIDTH) * (1 + HAIRWIDTH - diam);
				(*par).baddiam = 1;
				(*par).baddiam_logfactor += log(1 + HAIRWIDTH - diam);
			}
			diam = (2.0/a[0]);
			diamratio = mod->shape.comp[c].desc.ell.a_over_b.val;
			if (diamratio > SMALLRATIO) {
				a[1] = 2.0/(diam/diamratio);
			} else {
				a[1] = (2.0/(diam/SMALLRATIO)) / (1 + SMALLRATIO - diamratio);
				(*par).baddiam = 1;
				(*par).baddiam_logfactor += log(1 + SMALLRATIO - diamratio);
			}
			diam = (2.0/a[1]);
			diamratio = mod->shape.comp[c].desc.ell.b_over_c.val;
			if (diamratio > SMALLRATIO) {
				a[2] = 2.0/(diam/diamratio);
			} else {
				a[2] = (2.0/(diam/SMALLRATIO)) / (1 + SMALLRATIO - diamratio);
				(*par).baddiam = 1;
				(*par).baddiam_logfactor += log(1 + SMALLRATIO - diamratio);
			}
			for (i=0; i<=2; i++)
				a[i] *= a[i];

			/*  Loop through all vertices to find each one's
            distance from the ellipsoid's center.         */

            for (i=0; i<nv; i++) {

            	/*
              Routine setuprealver (called by setupreal, which was called
              by read_mod) already created as many ellipsoid vertices as were
              needed for the specified value of theta_steps, and initialized
              the direction cosines u[j] for each vertex to be
              sin(theta)cos(phi), sin(theta)sin(phi), and cos(theta) for
              j=0, 1, and 2, respectively.  These values are x/r, y/r, and z/r,
              where r is the distance from the origin to the ellipsoid surface
              along direction (theta, phi) for a given vertex.

              Since an ellipsoid has (x/a)^2 + (y/b)^2 + (z/c)^2 = 1,
              the quantity "den" in the code below is equal to 1/(r^2)
              for vertex i.

              Note that setuprealver initialized all vertex "base points" a[j]
              to be zero for ellipsoid components; hence "deviation" r is in
              fact the entire thing.
            	 */

            	den = 0.0;
            	for (j=0; j<=2; j++)
            		den += a[j]*( real->v[i].u[j] * real->v[i].u[j] );
            	real->v[i].r.val = 1/sqrt(den);
            }
            real->scalefactor[0].state = mod->shape.comp[c].desc.ell.two_a.state;
            real->scalefactor[1].state = mod->shape.comp[c].desc.ell.a_over_b.state;
            real->scalefactor[2].state = mod->shape.comp[c].desc.ell.b_over_c.state;
            for (j=0; j<=2; j++)
            	real->scalefactor[j].val = 1.0;
            break;
		case OVOID:

			/*  Determine all shape parameters, making sure that none are out of bounds  */

			a_radius = mod->shape.comp[c].desc.ovoid.two_a.val / 2;
			if (a_radius <= HAIRWIDTH/2) {
				(*par).baddiam = 1;
				(*par).baddiam_logfactor += log(1 + HAIRWIDTH - 2*a_radius);
				a_radius = (HAIRWIDTH/2) / (1 + HAIRWIDTH - 2*a_radius);
			}
			a_over_b = mod->shape.comp[c].desc.ovoid.a_over_b.val;
			if (a_over_b <= SMALLRATIO) {
				(*par).baddiam = 1;
				(*par).baddiam_logfactor += log(1 + SMALLRATIO - a_over_b);
				a_over_b = SMALLRATIO / (1 + SMALLRATIO - a_over_b);
			}
			b_over_c = mod->shape.comp[c].desc.ovoid.b_over_c.val;
			if (b_over_c <= SMALLRATIO) {
				(*par).baddiam = 1;
				(*par).baddiam_logfactor += log(1 + SMALLRATIO - b_over_c);
				b_over_c = SMALLRATIO / (1 + SMALLRATIO - b_over_c);
			}
			k_asym = mod->shape.comp[c].desc.ovoid.k.val;
			if (fabs(k_asym) > 1 - SMALLVAL) {
				(*par).baddiam = 1;
				(*par).baddiam_logfactor += log(fabs(k_asym) + SMALLVAL);
				if (k_asym > 0.0)
					k_asym = 1 - SMALLVAL*(1 - SMALLVAL)/k_asym;
				else
					k_asym = -1 - SMALLVAL*(1 - SMALLVAL)/k_asym;
			}

			/*  Compute x0, the x-offset that places the ovoid's center of mass at the origin;
            for small |k|, use an analytical approximation to avoid roundoff problems       */

			if (fabs(k_asym) > SMALLOVOIDK1) {
				x0term = 3*(1 - k_asym*k_asym)*log((1 + k_asym)/(1 - k_asym));
				numer = 2.0*k_asym*(3 - 2*k_asym*k_asym) - x0term;
				denom = 2.0*k_asym*(3 -   k_asym*k_asym) - x0term;
				x0 = (a_radius/k_asym)*(numer/denom);
			} else {
				x0 = 0.4*k_asym*a_radius;
			}

			/*  Loop through all vertices to find each one's distance from the origin  */

			for (i=0; i<nv; i++) {

				/*
              Routine setuprealver (called by setupreal, which was called
              by read_mod) already created as many ovoid vertices as were
              needed for the specified value of theta_steps, and initialized
              the direction cosines u[j] for each vertex to be
              sin(theta)cos(phi), sin(theta)sin(phi), and cos(theta) for
              j=0, 1, and 2, respectively.  These values are x/r, y/r, and z/r,
              where r is the distance from the origin to the ovoid surface
              along direction (theta, phi) for a given vertex.

              Note that setuprealver initialized all vertex "base points" a[j]
              to be zero for ovoid components; hence "deviation" r is in
              fact the entire thing.

              We obtain r by solving a cubic equation -- actually a cubic equation
              in the quantity r/a so that the coefficients are of order unity.
              In the code below, coeff[3] is the coefficient of (r/a)^3 while
              coeff[0] is the constant term.
				 */

				a_over_c = a_over_b*b_over_c;
				h = a_over_b*a_over_b*real->v[i].u[1]*real->v[i].u[1]
				                                                       + a_over_c*a_over_c*real->v[i].u[2]*real->v[i].u[2];
				alpha0 = x0/a_radius;
				u_x = real->v[i].u[0];
				coeff[3] = (h - u_x*u_x)*k_asym*u_x;
				coeff[2] = (1 + 3*k_asym*alpha0)*u_x*u_x + h*(1 - k_asym*alpha0);
				coeff[1] = (k_asym - (2 + 3*k_asym*alpha0)*alpha0)*u_x;
				coeff[0] = -(1 - alpha0*alpha0)*(1 + k_asym*alpha0);
				if (fabs(k_asym) <= SMALLOVOIDK2) {

					/*  |k| is very small, so guard against roundoff error by
                  computing the vertex position for an ellipsoid (k = 0)
                  and then applying a first-order correction for nonzero k  */

					goodroot = 1/sqrt(u_x*u_x + h);
					goodroot -= (coeff[3]*goodroot*goodroot*goodroot + coeff[1]*goodroot)
                        		  / (3*coeff[3]*goodroot*goodroot + 2*coeff[2]*goodroot + coeff[1]);
				} else {

					/*  |k| isn't very small, so solve the cubic equation  */

					nrealroots = cubic_realroots( coeff, realroot);
					goodroot = -HUGENUMBER;
					for (k=0; k<nrealroots; k++)
						if (realroot[k] >= 0.0) {
							x_over_a = realroot[k]*u_x;
							if (fabs(x_over_a - alpha0) - 1 < OVOIDTOL)
								goodroot = MAX( goodroot, realroot[k]);
						}
				}
				if (goodroot < 0.0) {
					fprintf( stderr, "Can't compute vertex displacement for ovoid vertex %d\n", i);
					bailout("realize_mod.c\n");
				}
				real->v[i].r.val = goodroot*a_radius;
			}

			/*  Assign scalefactor values  */

			real->scalefactor[0].state = mod->shape.comp[c].desc.ovoid.two_a.state;
			real->scalefactor[1].state = mod->shape.comp[c].desc.ovoid.a_over_b.state;
			real->scalefactor[2].state = mod->shape.comp[c].desc.ovoid.b_over_c.state;
			for (j=0; j<=2; j++)
				real->scalefactor[j].val = 1.0;
			break;
		case HARMONIC:
			L = mod->shape.comp[c].desc.har.nhar;
			for (i=0; i<nv; i++) {
				r = 0.0;
				for (l=0; l<=L; l++) {
					r += mod->shape.comp[c].desc.har.a[l][0].val
							* real->v[i].afactor[l][0];
					for (m=1; m<=l; m++)
						r += mod->shape.comp[c].desc.har.a[l][m].val
						* real->v[i].afactor[l][m]
						                          + mod->shape.comp[c].desc.har.b[l][m].val
						                          * real->v[i].bfactor[l][m];
				}
				if (r > HAIRWIDTH/2) {
					real->v[i].r.val = r;
				} else {
					(*par).baddiam = 1;
					(*par).baddiam_logfactor += log(1 + HAIRWIDTH - 2*r) / ((L+1)*(L+1));
					real->v[i].r.val = (HAIRWIDTH/2) / (1 + HAIRWIDTH - 2*r);
				}
			}
			for (j=0; j<=2; j++) {
				if (j > 0 && mod->shape.comp[c].desc.har.scalefactor[j].state == '=')
					mod->shape.comp[c].desc.har.scalefactor[j].val
					= mod->shape.comp[c].desc.har.scalefactor[j-1].val;
				real->scalefactor[j].state = mod->shape.comp[c].desc.har.scalefactor[j].state;
				real->scalefactor[j].val = mod->shape.comp[c].desc.har.scalefactor[j].val;
				if (real->scalefactor[j].val <= SMALLRATIO) {
					(*par).baddiam = 1;
					(*par).baddiam_logfactor += log(1 + SMALLRATIO - real->scalefactor[j].val);
					real->scalefactor[j].val = SMALLRATIO
							/ (1 + SMALLRATIO - real->scalefactor[j].val);
				}
			}
			break;
		case VERTEX:

			/*  The vertex type is its own realization, but we still
            need to update the values of the "scale factor" parameters
            and update any vertex deviations that have the '=' state    */

			for (i=0; i<nv; i++)
				if (real->v[i].r.state == '=') {
					v_mirror = real->v[i].v_mirror;
					real->v[i].r.val = real->v[v_mirror].r.val;
				}
			for (j=0; j<=2; j++) {
				if (j > 0 && mod->shape.comp[c].desc.ver.scalefactor[j].state == '=')
					mod->shape.comp[c].desc.ver.scalefactor[j].val
					= mod->shape.comp[c].desc.ver.scalefactor[j-1].val;
				real->scalefactor[j].val = mod->shape.comp[c].desc.ver.scalefactor[j].val;
				if (real->scalefactor[j].val <= SMALLRATIO) {
					(*par).baddiam = 1;
					(*par).baddiam_logfactor += log(1 + SMALLRATIO - real->scalefactor[j].val);
					real->scalefactor[j].val = SMALLRATIO
							/ (1 + SMALLRATIO - real->scalefactor[j].val);
				}
			}
			break;
		default:
			bailout("realize_mod.c: don't know that component type\n");

		}      /* end of switch statement for component type */
		//-----------------------------------------------------------------------------------------
		/*  Calculate vertex coordinates for this component  */

		for (i=0; i<nv; i++) {
			for (j=0; j<=2; j++)
				real->v[i].x[j] = real->scalefactor[j].val
						* (real->v[i].u[j] * real->v[i].r.val + real->v[i].a[j]);
		}
		//-----------------------------------------------------------------------------------------
		/*  Use this component's rotational offset angles to create comp[c].m,
        the rotation matrix that will be applied to the vertex coordinates  */

		euler2mat( mod->shape.comp[c].m, mod->shape.comp[c].rot[0].val,
				mod->shape.comp[c].rot[1].val, mod->shape.comp[c].rot[2].val);

		//-----------------------------------------------------------------------------------------
		/*  If needed, perform rotation on this component  */

		if (!(mod->shape.comp[c].rot[0].val == 0 &&
				mod->shape.comp[c].rot[1].val == 0 &&
				mod->shape.comp[c].rot[2].val == 0    )) {
			for (i=0; i<nv; i++) {
				for (j=0; j<=2; j++) {
					x[j] = 0.0;
					for (k=0; k<=2; k++)
						x[j] += mod->shape.comp[c].m[j][k] * real->v[i].x[k];
				}
				for (j=0; j<=2; j++)
					real->v[i].x[j] = x[j];
			}
		}
		//-----------------------------------------------------------------------------------------
		/*  If needed, perform translation on this component  */

		if (!(mod->shape.comp[c].off[0].val == 0.0 &&
				mod->shape.comp[c].off[1].val == 0.0 &&
				mod->shape.comp[c].off[2].val == 0.0    )) {
			for (i=0; i<nv; i++)
				for (j=0; j<=2; j++)
					real->v[i].x[j] += mod->shape.comp[c].off[j].val;
		}
		//-----------------------------------------------------------------------------------------
		/* 	Figure out if optical/radar harmonic scattering laws are in use     *
		 *  and set the flag harmonic_scatlaw accordingly				        */

		harmonic_scatlaw = 0;
		for (ilaw=0; ilaw<mod->photo.noptlaws; ilaw++)
			if (mod->photo.opttype[ilaw] == HARMLAMBERT || mod->photo.opttype[ilaw] == HARMLOMMEL
					|| mod->photo.opttype[ilaw] == HARMHAPKE
					|| mod->photo.opttype[ilaw] == HARMKAAS)
				harmonic_scatlaw = 1;
		for (ilaw=0; ilaw<mod->photo.nradlaws; ilaw++)
			if (mod->photo.radtype[ilaw] == HARMCOSINE_DIFF)
				harmonic_scatlaw = 1;
		//-----------------------------------------------------------------------------------------
		/*  For each facet of this component, compute the outward unit normal,
        the area, the mean coordinates of the three corner vertices, and
        the corresponding angular coordinates (for some scattering laws)    */


		for (f=0; f<nf; f++) {
			real->f[f].area = facnrm( *real, f);

			for (j=0; j<=2; j++)
				real->f[f].x[j] = (real->v[ real->f[f].v[0] ].x[j] +
						real->v[ real->f[f].v[1] ].x[j] +
						real->v[ real->f[f].v[2] ].x[j]   )/3;

			if (harmonic_scatlaw) {
				real->f[f].theta = atan2( sqrt(real->f[f].x[0]*real->f[f].x[0] +
						real->f[f].x[1]*real->f[f].x[1]   ),
						real->f[f].x[2]);
				real->f[f].phi = atan2( real->f[f].x[1], real->f[f].x[0]);
			}
		}

		//-----------------------------------------------------------------------------------------
		/*  Calculate vertex normals for this component as normalized sums
        of the facet normals for all facets attached to each vertex     */

		for (i=0; i<nv; i++) {
			n[0] = n[1] = n[2] = 0.0;
			naf = real->v[i].naf;
			for (j=0; j<naf; j++) {
				f = real->v[i].af[j];
				for (k=0; k<=2; k++)
					n[k] += real->f[f].n[k];
			}
			normalize( n);
			for (k=0; k<=2; k++)
				real->v[i].n[k] = n[k];
		}
	}      /* end loop over all components */
}
/*.....................................................................................*/

/*  Determine which vertices, facets, and sides of a multiple-component
    model lie interior to the model rather than on the model's surface,
    and reset their "act" (active) flags to zero                         */

void check_surface( struct mod_t *mod)
{
	int c, c2, f, f2, i, n_intersections, nf, nv, ns, v, k, v1, v2, n,
	n_edge, new_edge;
	double a[3], u[3], dist, r, s, t;
	double *r_edge;
	struct vertices_t *real, *real2;

	/*  1-component model: flag all vertices and facets as active, then return  */

	if (mod->shape.ncomp == 1) {
		real = &mod->shape.comp[0].real;
		nv = real->nv;
		nf = real->nf;
		ns = real->ns;
		for (v=0; v<nv; v++)
			real->v[v].act = 1;
		for (f=0; f<nf; f++)
			real->f[f].act = 1;
		for (k=0; k<ns; k++)
			real->s[k].act = 1;
		return;
	}

	a[0] = a[1] = a[2] = 0.0;  /* vertex base displacement */

	r_edge = vector( 0, MAXEDGE-1);

	/*  Check each model component in turn  */

	for (c=0; c<mod->shape.ncomp; c++) {

		real = &mod->shape.comp[c].real;
		nv = real->nv;
		nf = real->nf;
		ns = real->ns;

		/*  Check this component's vertices  */

		for (v=0; v<nv; v++) {

			/*  Check whether vertex v of component c lies interior to
          any other component c2                                  */

			/*  Start by considering a ray that starts at the origin and passes through
          vertex v: the displacement vector for this vertex.  Vector u below
          holds the direction cosines of this ray, while dist is the magnitude
          of the displacement.  (The existing direction cosines real->v[v].u
          may not point directly away from the origin, so we compute from scratch.)  */

			for (i=0; i<=2; i++)
				u[i] = real->v[v].x[i];
			dist = normalize( u);

			/*  Now, for each other component c2, loop through all facets f2 to find
          the ones that are intersected by the ray defined above.  Count up all
          such facets of c2 for which the intersection point lies further from
          the origin than vertex v.  If this number is ODD, vertex v lies
          interior to component c2, so we mark it as inactive.                   */

			real->v[v].act = 1;

			c2 = (c == 0) ? 1 : 0;
			do {
				real2 = &mod->shape.comp[c2].real;
				n_intersections = 0;
				n_edge = 0;
				for (f2=0; f2<(*real2).nf; f2++) {
					if (rayfacint( &r, &s, &t, u, a,
							(*real2).v[ (*real2).f[f2].v[0] ].x,
							(*real2).v[ (*real2).f[f2].v[1] ].x,
							(*real2).v[ (*real2).f[f2].v[2] ].x,
							(*real2).f[f2].n, EDGETOL))
						if (r > dist + RTOL) {
							if (fabs(s) < EDGETOL || fabs(s - 1.0) < EDGETOL
									|| fabs(t) < EDGETOL || fabs(t - s) < EDGETOL) {

								/*  The ray intersects facet f2 at its edge or corner, give or take
                        a bit of roundoff error.  (Absent roundoff error, we would have
                        s = 0.0 or 1.0, or t = 0.0 or s.)  We need to make sure that we
                        count only one intersection for this edge, rather than counting
                        both facets that adjoin the edge.  Thus we check the distance r
                        from vertex v to the intersection point against the values of r
                        obtained for all previous edge intersections found for this
                        vertex.  If the current r value is the same (to within a small
                        tolerance) as a previous one, we've already counted this
                        intersection, so don't count it again.                           */

								new_edge = 1;
								if (n_edge > 0)
									for (n=0; n<n_edge; n++)
										if (fabs(r - r_edge[n]) < RTOL)
											new_edge = 0;
								if (new_edge) {
									if (n_edge == MAXEDGE)
										bailout("realize_mod.c: need to increase MAXEDGE\n");
									r_edge[n_edge] = r;
									n_edge++;
									n_intersections++;
								}

							} else {

								/*  The ray intersects the interior of facet f2, not the edge  */

								n_intersections++;
							}
						}
				}
				if (n_intersections % 2 == 1)
					real->v[v].act = 0;
				c2 = (c2 == c-1) ? c2 + 2 : c2 + 1;
			} while (real->v[v].act && c2 < mod->shape.ncomp);
		}

		/*  Check this component's facets, doing exactly what we just did for vertices
        but this time for the *mean displacement* of each facet's three vertices     */

		for (f=0; f<nf; f++) {

			for (i=0; i<=2; i++)
				u[i] = real->f[f].x[i];
			dist = normalize( u);

			real->f[f].act = 1;

			c2 = (c == 0) ? 1 : 0;
			do {
				real2 = &mod->shape.comp[c2].real;
				n_intersections = 0;
				n_edge = 0;
				for (f2=0; f2<(*real2).nf; f2++)
					if (rayfacint( &r, &s, &t, u, a,
							(*real2).v[ (*real2).f[f2].v[0] ].x,
							(*real2).v[ (*real2).f[f2].v[1] ].x,
							(*real2).v[ (*real2).f[f2].v[2] ].x,
							(*real2).f[f2].n, EDGETOL))
						if (r > dist + RTOL) {
							if (fabs(s) < EDGETOL || fabs(s - 1.0) < EDGETOL
									|| fabs(t) < EDGETOL || fabs(t - s) < EDGETOL) {
								new_edge = 1;
								if (n_edge > 0)
									for (n=0; n<n_edge; n++)
										if (fabs(r - r_edge[n]) < RTOL)
											new_edge = 0;
								if (new_edge) {
									if (n_edge == MAXEDGE)
										bailout("realize_mod.c: need to increase MAXEDGE\n");
									r_edge[n_edge] = r;
									n_edge++;
									n_intersections++;
								}
							} else {
								n_intersections++;
							}
						}
				if (n_intersections % 2 == 1)
					real->f[f].act = 0;
				c2 = (c2 == c-1) ? c2 + 2 : c2 + 1;
			} while (real->f[f].act && c2 < mod->shape.ncomp);
		}

		/*  Check this component's sides:
        a side is active IFF both of its end vertices are active  */

		for (k=0; k<ns; k++) {
			v1 = real->s[k].v[0];
			v2 = real->s[k].v[1];
			if (real->v[v1].act && real->v[v2].act)
				real->s[k].act = 1;
			else
				real->s[k].act = 0;
		}

	}      /* end loop over all components */

	free_vector( r_edge, 0, MAXEDGE-1);

}


/*  Compute the area and the 0,1,2-order moments (volume, center of mass, and
    inertia tensor) of each component and of the overall model, assuming uniform
    density and ignoring interior facets' contributions to the overall model      */

void compute_moments( struct mod_t *mod)
{
	int c, nv, nf, j, k, f;
	double dI[3][3], dcom[3], dv;
	struct vertices_t *real;

	/*  Initialize the model's surface area, volume,
      center-of-mass (COM) displacement, and inertia tensor  */

	mod->shape.area = 0.0;
	mod->shape.volume = 0.0;
	for (k=0; k<=2; k++) {
		mod->shape.com[k] = 0.0;
		for (j=0; j<=2; j++)
			mod->shape.inertia[k][j] = 0.0;
	}

	/*  Loop over all model components, computing areas and moments
      (volume, center of mass, and inertia tensor); the COM and
      inertia tensor are computed assuming uniform density.

      For multiple-component models, when computing the area and the
      moments for the overall model, ignore facets that are interior
      to the model (i.e., that are inside some other component).         */

	for (c=0; c<mod->shape.ncomp; c++) {

		real = &mod->shape.comp[c].real;
		nv = real->nv;
		nf = real->nf;

		/*  Calculate surface area for this component; for active facets,
        also add the contributions to the area of the overall model    */

		mod->shape.comp[c].area = 0.0;
		for (f=0; f<nf; f++) {
			mod->shape.comp[c].area += real->f[f].area;
			if (real->f[f].act)
				mod->shape.area += real->f[f].area;
		}

		/*  Calculate 0,1,2-order moments for this component; for active facets,
        also add the contributions to the moments of the overall model        */

		mod->shape.comp[c].volume = 0.0;
		for (k=0; k<=2; k++) {
			mod->shape.comp[c].com[k] = 0.0;
			for (j=0; j<=2; j++)
				mod->shape.comp[c].inertia[k][j] = 0.0;
		}
		for (f=0; f<nf; f++) {
			facmom( real->v[ real->f[f].v[0] ].x,
					real->v[ real->f[f].v[1] ].x,
					real->v[ real->f[f].v[2] ].x, real->f[f].n,
					&dv, dcom, dI);
			mod->shape.comp[c].volume += dv;
			for (j=0; j<=2; j++) {
				mod->shape.comp[c].com[j] += dcom[j];
				for (k=0; k<=2; k++)
					mod->shape.comp[c].inertia[j][k] += dI[j][k];
			}

			if (real->f[f].act) {
				mod->shape.volume += dv;
				for (j=0; j<=2; j++) {
					mod->shape.com[j] += dcom[j];
					for (k=0; k<=2; k++)
						mod->shape.inertia[j][k] += dI[j][k];
				}
			}
		}

		for (j=0; j<=2; j++)
			mod->shape.comp[c].com[j] /= mod->shape.comp[c].volume;

	}      /* end loop over all components */

	/*  Get COM displacement for the complete model  */

	for (j=0; j<=2; j++)
		mod->shape.com[j] /= mod->shape.volume;

	j=2;

}


/*  Find all real roots of a cubic equation, using methods given in section 5.6 of
    Numerical Recipes in C.  Element 3 of the input coeff vector is the cubic
    coefficient while element 0 is the constant term.  Up to three real roots are
    stored in the output realroot vector, with any unused elements set to a large
    negative dummy value.  The return value is the number of real roots found.
    The routine includes several tests for coefficients that are equal to zero;
    those tests assume that nonzero coefficients are of order unity.                */

int cubic_realroots( double *coeff, double *realroot)
{
	int nrealroots, bsign;
	double a, b, c, discriminant, q, qsqrt, r, r2minusq3, rsign, s, t, theta;

	nrealroots = 0;
	realroot[0] = realroot[1] = realroot[2] = -HUGENUMBER;

	if (fabs(coeff[3]) < SMALLCOEFF3) {

		/*  cubic term is zero  */

		a = coeff[2];
		b = coeff[1];
		c = coeff[0];

		if (fabs(a) < SMALLVAL) {

			if (fabs(b) < SMALLVAL) {

				/*  Error: the cubic, quadratic, and linear terms are zero  */

				if (fabs(c) < SMALLVAL)
					bailout("cubic_realroots in realize_mod.c: all four coefficients are zero\n");
				else
					bailout("cubic_realroots in realize_mod.c: only the constant term is nonzero\n");

			} else {

				/*  linear equation  */

				realroot[0] = -c/b;
				nrealroots = 1;
			}

		} else {

			/*  quadratic equation  */

			discriminant = b*b - 4*a*c;
			if (discriminant < 0.0)
				bailout("cubic_realroots in realize_mod.c: quadratic equation has no real roots\n");
			if (fabs(b) < SMALLVAL) {
				realroot[0] = sqrt(discriminant)/(2*a);
				realroot[1] = -realroot[0];
			} else {
				bsign = (b < 0.0) ? -1 : 1;
				q = -0.5*(b + bsign*sqrt(discriminant));
				realroot[0] = q/a;
				realroot[1] = c/q;
			}
			nrealroots = 2;
		}

	} else {

		/*  cubic term is nonzero: scale to standard form x^3 + ax^2 + b^x + c = 0  */

		a = coeff[2]/coeff[3];
		b = coeff[1]/coeff[3];
		c = coeff[0]/coeff[3];

		/*  Check if there is one real root or three

          Write out the test quantity r^2 - q^3 explicitly in terms of
          coefficients a, b, and c in order to cancel high-order terms
          and thus reduce the likelihood of roundoff problems           */

		q = (a*a - 3*b)/9;
		r = (2*a*a*a - 9*a*b + 27*c)/54;

		r2minusq3 = (4*a*a*a*c - a*a*b*b - 18*a*b*c + 27*c*c + 4*b*b*b)/108;
		if (r2minusq3 >= 0.0) {

			/*  one real root  */

			rsign = (r < 0.0) ? -1 : 1;
			s = -rsign*pow( fabs(r) + sqrt(r2minusq3), 1.0/3);
			t = (fabs(s) >= SMALLVAL) ? q/s : 0.0;
			realroot[0] = s + t - a/3;
			nrealroots = 1;

		} else {

			/*  three real roots  */

			qsqrt = sqrt(q);
			theta = acos(r/(q*qsqrt));
			realroot[0] = -2*qsqrt*cos(theta/3) - a/3;
			realroot[1] = -2*qsqrt*cos((theta + 2*PIE)/3) - a/3;
			realroot[2] = -2*qsqrt*cos((theta - 2*PIE)/3) - a/3;
			nrealroots = 3;
		}
	}
	return nrealroots;
}

#undef HAIRWIDTH
#undef SMALLRATIO
#undef SMALLOVOIDK1
#undef SMALLOVOIDK2
#undef OVOIDTOL
#undef MAXEDGE
#undef EDGETOL
#undef RTOL
#undef SMALLCOEFF3
