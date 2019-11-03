/*****************************************************************************************
                                                                             mirror_mod.c

Replaces a model with its mirror-ambiguity version.

Modified 2013 May 20 by CM:
    Implement ovoid shape components

Modified 2009 April 1 by CM:
    Permit ellipsoid and harmonic components
    Change the old mirroring scheme -- reflecting the shape about the yz
       plane, then negating "spin 2" and "angle 2" -- so that we don't have
       to deal with negative spin rates.  Instead, obtain a physically
       identical result by reflecting the shape about the xy plane, then
       rotating the spin vector by 180 degrees about the y-axis; this means
       negating all z-displacements and "spin 0" while also adjusting
       "angle 0" and "angle 1" to refer to a flipped pole direction.
    Adjust linear and rotational offsets for each component
*****************************************************************************************/

#include "head.h"

void mirror_mod( struct mod_t *mod)
{
  int c, l, m, L, i, f;

  /* Adjust shape specification  */
  for (c=0; c<mod->shape.ncomp; c++) {

    /* Negate z-components of linear offsets  */
    mod->shape.comp[c].off[2].val *= -1;

    /* Negate second Euler angle for rotational offsets (since pole direction
     * will be flipped later)      */
    mod->shape.comp[c].rot[1].val *= -1;

    /* Reflect this model component about the xy plane  */
    switch (mod->shape.comp[c].type) {
    case ELLIPSE:
    case OVOID:

      /* Nothing to do for ellipsoid or ovoid  */
      break;
    case HARMONIC:

      /* Negate a and b coefficients for which l+m (degree+order) is odd: these
       * spherical harmonic expansion terms are antisymmetric in z   */
      L = mod->shape.comp[c].desc.har.nhar;
      for (l=0; l<=L; l++) {
        if (l % 2 != 0)
          mod->shape.comp[c].desc.har.a[l][0].val *= -1;
        for (m=1; m<=l; m++) {
          if ((l + m) % 2 != 0) {
            mod->shape.comp[c].desc.har.a[l][m].val *= -1;
            mod->shape.comp[c].desc.har.b[l][m].val *= -1;
          }
        }
      }
      break;
    case VERTEX:

        /* Negate z components of vertex displacements (both direction cosines
         * u and base displacements a)  */
        for (i=0; i<mod->shape.comp[c].desc.ver.nv; i++) {
          mod->shape.comp[c].desc.ver.v[i].a[2] *= -1.0;
          mod->shape.comp[c].desc.ver.v[i].u[2] *= -1.0;
        }

        /* Reorder vertices of each facet so the facet normal, defined in right
         * hand sense, still points outward         */
        for (i=0; i<mod->shape.comp[c].desc.ver.nf; i++) {
          f = mod->shape.comp[c].desc.ver.f[i].v[1];
          mod->shape.comp[c].desc.ver.f[i].v[1] =
                                  mod->shape.comp[c].desc.ver.f[i].v[2];
          mod->shape.comp[c].desc.ver.f[i].v[2] = f;
        }
        break;
    default:
        bailout("mirror_mod.c: can't do that component type\n");
    }
  }

  /* Adjust spin specification: flip pole direction and negate x-comp of spin*/
  mod->spin.angle[0].val += PIE;
  mod->spin.angle[1].val = PIE - mod->spin.angle[1].val;
  mod->spin.omega[0].val *= -1;
}
