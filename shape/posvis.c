/*****************************************************************************************
                                                                                 posvis.c

Fill in the portion of a plane-of-sky image due to a particular model component: Assign
each relevant POS pixel a z-value in observer coordinates (distance from the origin
towards Earth) and a value of cos(scattering angle).

Return 1 if any portion of this component lies outside the specified POS window,
0 otherwise.

If the "src" argument is true, the "observer" is the Sun rather than Earth, and
"plane-of-sky" becomes "projection as viewed from the Sun."

Modified 2014 February 20 by CM:
    Allow facets that partly project outside the POS frame to contribute to the POS frame
        (thus avoiding see-through "holes" in the model at the edge of a POS image)

Modified 2010 May 18 by CM:
    Bug fix: When checking if a POS pixel hasn't already been assigned
        values during a previous call to posvis for a different component,
        check for fac[i][j] < 0 rather than cosa[i][j] == 0.0, since for
        bistatic situations the latter condition will also be true for
        pixels centered on Earth-facing facets that don't face the Sun

Modified 2009 July 2 by CM:
    Eliminate the check that facets are "active": this term is now being
        interpreted to mean "not lying interior to the model," so the
        check is unnecessary and the determination of active vs. inactive
        status is inaccurate for half-exposed facets at the intersections
        between model components

Modified 2009 April 3 by CM:
    Compute the "posbnd_logfactor" parameter: if the model extends beyond
        the POS frame, posbnd_logfactor is set to the logarithm of the
        ratio of the area that would have been required to "contain" the
        entire model divided by the area of the actual POS frame
    Work with floating-point pixel numbers (imin_dbl, etc.), at least
        initially, in case the sky rendering for a model with illegal
        parameters would involve huge pixel numbers that exceed the
        limits for valid integers

Modified 2007 August 4 by CM:
    Add "orbit_offset" and "body" parameters and remove "facet" parameter
    Add body, bodyill, comp, and compill matrices for POS frames

Modified 2006 June 21 by CM:
    For POS renderings, change res to km_per_pixel

Modified 2005 September 19 by CM:
    Allow for roundoff error when determining which POS pixels project
        onto each model facet

Modified 2005 June 27 by CM:
    Renamed "round" function to "iround" to avoid conflicts

Modified 2005 June 22 by CM:
    Slightly modified some comments

Modified 2005 January 25 by CM:
    Take care of unused and uninitialized variables

Modified 2004 December 19 by CM:
    Added more comments
    Put update of rectangular POS area into "POSrect" routine and applied it
        even to facets which lie outside the POS frame

Modified 2004 Feb 11 by CM:
    Added comments

Modified 2003 May 5 by CM:
    Removed redundant coordinate transformation of the unit normal n
    for the no-smoothing case
 *****************************************************************************************/

#include "head.h"
#include <limits.h>

void POSrect( struct pos_t *pos, int src,
		double imin_dbl, double imax_dbl, double jmin_dbl, double jmax_dbl,
		double *imin_overall, double *imax_overall,
		double *jmin_overall, double *jmax_overall);


int posvis( struct vertices_t *verts, double orbit_offset[3], struct pos_t *pos,
		int smooth, int src, int body, int comp)
{
	int f, i, j, imin, imax, jmin, jmax, k, outbnd, i1, i2, j1, j2;
	int **bod, **cmp, **fac;
	double v0[3], v1[3], v2[3], z, s, t, x[3], den, xfactor, yfactor,
	imin_dbl, imax_dbl, jmin_dbl, jmax_dbl, imin_overall, imax_overall,
	jmin_overall, jmax_overall, n[3], oa[3][3], usrc[3];
	double **cosa, **cosb, **zz;

	int dbg_cntr=0;

	/*  Initialize variables  */
	outbnd = 0;
	pos->posbnd_logfactor = 0.0;
	imin_overall = jmin_overall =  HUGENUMBER;
	imax_overall = jmax_overall = -HUGENUMBER;
	mtrnsps(oa, pos->ae);

	if (src) {

		/* We're viewing the model from the sun: at the center of each pixel
		 * in the projected view, we want cos(incidence angle), distance from
		 * the COM towards the sun, and the facet number.                */
		mmmul( oa, pos->se, oa);      /* oa takes ast into sun coords           */
		cosa = pos->cosill;           /* cos(incidence angle)                   */
		cosb = NULL;                  /* <avoid compilation warnings>           */
		zz = pos->zill;               /* distance towards sun                   */
		bod = pos->bodyill;           /* body number at projected pixel center  */
		cmp = pos->compill;           /* component number at projected pixel center */
		fac = pos->fill;              /* facet number at projected pixel center */

	} else {

		/* We're viewing the model from Earth: at the center of each POS pixel
		 * we want cos(scattering angle), distance from the COM towards Earth,
		 * and the facet number.  For bistatic situations (lightcurves) we also
          want cos(incidence angle) and the unit vector towards the source.     */
		mmmul( oa, pos->oe, oa);      /* oa takes ast into obs coords */
		cosa = pos->cose;             /* scattering angle */
		cosb = pos->cosi;             /* incident angle */
		zz = pos->z;                  /* observer z-coordinate (backwards along LOS) */
		bod = pos->body;              /* body number at projected pixel center  */
		cmp = pos->comp;              /* component number at POS pixel center */
		fac = pos->f;                 /* facet number at POS pixel center */
		if (pos->bistatic) {
			usrc[0] = usrc[1] = 0.0;      /* unit vector towards source */
			usrc[2] = 1.0;
			cotrans( usrc, pos->se, usrc, -1);
			cotrans( usrc, pos->oe, usrc,  1);   /* in observer coordinates */
		}
	}

	/* For all comments below this point: If the "src" argument is true,
	 * "observer" is actually "source" (sun), "Earth" is actually "source" and
	 * "POS" is actually "projection as viewed from the source"         */

	/*  Loop through all facets of this model component  */
	for (f=0; f<verts->nf; f++) {

		/* Get the normal to this facet in body-fixed (asteroid) coordinates
		 * and convert it to observer coordinates     */
		for (i=0; i<=2; i++)
			n[i] = verts->f[f].n[i];
		cotrans( n, oa, n, 1);

		/* Consider this facet further only if its normal points somewhat
		 * towards the observer rather than away         */
		if (n[2] > 0.0) {
			//dbg_cntr++;
			/* Convert the three sets of vertex coordinates from body to ob-
			 * server coordinates; orbit_offset is the center-of-mass offset
			 * (in observer coordinates) for this model at this frame's epoch
			 * due to orbital motion, in case the model is half of a binary
			 * system.  */
			cotrans(v0, oa, verts->v[verts->f[f].v[0]].x, 1);
			cotrans(v1, oa, verts->v[verts->f[f].v[1]].x, 1);
			cotrans(v2, oa, verts->v[verts->f[f].v[2]].x, 1);
			for (i=0; i<=2; i++) {
				v0[i] += orbit_offset[i];
				v1[i] += orbit_offset[i];
				v2[i] += orbit_offset[i];
			}

			/* Find rectangular region (in POS pixels) containing the projected
			 * facet - use floats in case model has illegal parameters and the
			 * pixel numbers exceed the limits for valid integers                         */
			imin_dbl = floor( MIN( v0[0], MIN( v1[0], v2[0]))/pos->km_per_pixel - SMALLVAL	+ 0.5);
			imax_dbl = floor( MAX( v0[0], MAX( v1[0], v2[0]))/pos->km_per_pixel + SMALLVAL	+ 0.5);
			jmin_dbl = floor( MIN( v0[1], MIN( v1[1], v2[1]))/pos->km_per_pixel - SMALLVAL	+ 0.5);
			jmax_dbl = floor( MAX( v0[1], MAX( v1[1], v2[1]))/pos->km_per_pixel + SMALLVAL	+ 0.5);

			imin = (imin_dbl < INT_MIN) ? INT_MIN : (int) imin_dbl;
			imax = (imax_dbl > INT_MAX) ? INT_MAX : (int) imax_dbl;
			jmin = (jmin_dbl < INT_MIN) ? INT_MIN : (int) jmin_dbl;
			jmax = (jmax_dbl > INT_MAX) ? INT_MAX : (int) jmax_dbl;

			/*  Set the outbnd flag if the facet extends beyond the POS window  */
			if ( (imin < (-pos->n)) || (imax > pos->n) ||
					(jmin < (-pos->n)) || (jmax > pos->n)    )
				outbnd = 1;

			/* Figure out if facet projects at least partly within POS window;
			 * if it does, look at each "contained" POS pixel and get the
			 * z-coordinate and cos(scattering angle)           */
			i1 = MAX( imin, -pos->n);
			i2 = MIN( imax,  pos->n);
			j1 = MAX( jmin, -pos->n);
			j2 = MIN( jmax,  pos->n);

			if (i1 > pos->n || i2 < -pos->n || j1 > pos->n || j2 < -pos->n) {

				/* Facet is entirely outside the POS frame: just keep track of
				 * changed POS region     */
				POSrect( pos, src, imin_dbl, imax_dbl, jmin_dbl, jmax_dbl,
						&imin_overall, &imax_overall, &jmin_overall, &jmax_overall);
			} else {


				/* Facet is at least partly within POS frame: find all POS
				 * pixels whose centers project onto this facet  */
				for (i=i1; i<=i2; i++) {
					x[0] = i*pos->km_per_pixel;
					for (j=j1; j<=j2; j++) {
						x[1] = j*pos->km_per_pixel;

						/* Compute parameters s(x,y) and t(x,y) which define a
						 * facet's surface as
						 *         z = z0 + s*(z1-z0) + t*(z2-z1)
						 * where z0, z1, and z2 are the z-coordinates at the
						 * vertices. The conditions 0 <= s <= 1 and
						 * 0 <= t <= s require the POS pixel center to be
						 * "within" the (projected) perimeter of facet f.    */
						s = ((x[0]-v0[0])*(v2[1]-v1[1]) - (v2[0]-v1[0])*(x[1]-v0[1]))
           					* (den = 1/( (v1[0]-v0[0])*(v2[1]-v1[1]) - (v2[0]-v1[0])*
           											(v1[1]-v0[1]) ));
						if ( (s >= -SMALLVAL) && (s <= 1.0+SMALLVAL) ) {
							t = ((v1[0]-v0[0])*(x[1]-v0[1]) -
									(x[0]-v0[0])*(v1[1]-v0[1])) * den;
							if ( (t >= -SMALLVAL) && (t <= s+SMALLVAL) ) {
								/* Compute z-coordinate of pixel center: its
								 * distance measured from the origin towards
								 * Earth.    */

								z = v0[2] + s*(v1[2] - v0[2]) + t*(v2[2] - v1[2]);

								/* If fac[i][j] is >= 0, pixel [i][j] was al-
								 * ready assigned values during a previous call
								 * to posvis for a different model component.
								 * If so, override only if the current component
								 * is blocking our view of (i.e., is closer to
								 * us than) the previous one.   */
								if ( (z > zz[i][j]) || (fac[i][j] < 0) ) {
									/* Next line assigns distance of POS pixel
									 * center from COM towards Earth; that is,
									 * by changing zz,it changes pos->z or
									 * pos->zill                */
									zz[i][j] = z;

									if (smooth) {
										/* Get smoothed version of facet unit
										 * normal: Take the linear combination
										 * of the three vertex normals; trans-
										 * form from body to observer coordina-
										 * tes; and make sure that it points
										 * somewhat in our direction.         */
										for (k=0; k<=2; k++)
											n[k] = verts->v[verts->f[f].v[0]].n[k]
										    + s*(  verts->v[verts->f[f].v[1]].n[k]
										    - verts->v[verts->f[f].v[0]].n[k])
										    + t*(  verts->v[verts->f[f].v[2]].n[k]
										    - verts->v[verts->f[f].v[1]].n[k]);
										cotrans( n, oa, n, 1);
										normalize( n);
									}

									/* Determine scattering and/or incidence
									 * angles. Next lines change pos->cose/
									 * cosill. If bistatic (lightcurves), where
									 * we are viewing from Earth (src = 0),
									 * pos->cosi is also changed.                 */
									if (n[2] > 0.0) {

										cosa[i][j] = n[2];
										if ((!src) && (pos->bistatic)) {

											cosb[i][j] = dot( n, usrc);
											if (cosb[i][j] <= 0.0) {
												cosa[i][j] = 0.0;
											}
										}
									}

									/*  Keep track of the changed POS region  */
									POSrect( pos, src, (double) i, (double) i,
											(double) j, (double) j, &imin_overall,
											&imax_overall, &jmin_overall, &jmax_overall);

									/* Next lines change pos->body/bodyill,
									 * pos->comp/compill, pos->f/fill          */
									bod[i][j] = body;
									cmp[i][j] = comp;
									fac[i][j] = f;

								}  /* end if (no other facet yet blocks this facet from view) */
							}  /* end if 0 <= t <= s (facet center is "in" this POS pixel) */
						}  /* end if 0 <= s <= 1 */
					}  /* end j-loop over POS rows */
				}  /* end i-loop over POS columns */
			}  /* end else (facet doesn't extend beyond POS window) */
		}  /* end if facet points towards observer */
	}  /* end f-loop over facets */

	/*  If the target extends beyond the POS frame, set pos->posbnd_logfactor equal
      to the logarithm of the ratio of the number of pixels in a frame extended to
      include the entire target, divided by the number of pixels in the actual frame  */

	if (outbnd) {
		xfactor = (MAX( imax_overall,  pos->n) - MIN( imin_overall, -pos->n) + 1)
            														  / (2*pos->n + 1);
		yfactor = (MAX( jmax_overall,  pos->n) - MIN( jmin_overall, -pos->n) + 1)
            														  / (2*pos->n + 1);
		pos->posbnd_logfactor = log(xfactor*yfactor);
	}
//	dbg_print_pos_arrays_full_host(pos);
//	dbg_print_facet_normals_dbl3(dbg_hn, verts->nf, "CPU_nrmls.csv");
//	dbg_print_facet_normals_dbl3(dbg_hn, nf, "FP64_nrmls.csv");
//	printf("dbg_cntr in posvis (CPU) = %i\n", dbg_cntr);
//	printf("xlim[0] = %i\n", pos->xlim[0]);
//	printf("xlim[0] = %i\n", pos->xlim[1]);
//	printf("ylim[1] = %i\n", pos->ylim[0]);
//	printf("ylim[1] = %i\n", pos->ylim[1]);
	return outbnd;
}


/* Update the rectangular plane-of-sky region that "contains" the target */

void POSrect( struct pos_t *pos, int src,
		double imin_dbl, double imax_dbl, double jmin_dbl, double jmax_dbl,
		double *imin_overall, double *imax_overall,
		double *jmin_overall, double *jmax_overall)
{
	int n, imin, imax, jmin, jmax;

	/*  Update the POS region that contains the target without
      regard to whether or not it extends beyond the POS frame  */

	*imin_overall = MIN( *imin_overall, imin_dbl);
	*imax_overall = MAX( *imax_overall, imax_dbl);
	*jmin_overall = MIN( *jmin_overall, jmin_dbl);
	*jmax_overall = MAX( *jmax_overall, jmax_dbl);

	/*  Update the subset of the POS frame that contains the target  */

	imin = (imin_dbl < INT_MIN) ? INT_MIN : (int) imin_dbl;
	imax = (imax_dbl > INT_MAX) ? INT_MAX : (int) imax_dbl;
	jmin = (jmin_dbl < INT_MIN) ? INT_MIN : (int) jmin_dbl;
	jmax = (jmax_dbl > INT_MAX) ? INT_MAX : (int) jmax_dbl;

	n = pos->n;
	if (src) {
		if (imin < pos->xlim2[0])
			pos->xlim2[0] = MAX( imin, -n);
		if (imax > pos->xlim2[1])
			pos->xlim2[1] = MIN( imax,  n);
		if (jmin < pos->ylim2[0])
			pos->ylim2[0] = MAX( jmin, -n);
		if (jmax > pos->ylim2[1])
			pos->ylim2[1] = MIN( jmax,  n);
	} else {
		if (imin < pos->xlim[0])
			pos->xlim[0] = MAX( imin, -n);
		if (imax > pos->xlim[1])
			pos->xlim[1] = MIN( imax,  n);
		if (jmin < pos->ylim[0])
			pos->ylim[0] = MAX( jmin, -n);
		if (jmax > pos->ylim[1])
			pos->ylim[1] = MIN( jmax,  n);
	}
}
