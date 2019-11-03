/*****************************************************************************************
                                                                                ref_mod.c

ref_mod.c implements the 'refshape' action by changing the explicit specification of
harmonic and vertex shape components without changing the shape; it has no effect on
ellipsoid or ovoid shape components.  For a harmonic component, it multiplies all "a" and
"b" coefficients by "scalefactor" (whose three components must be identical) and then
resets all "scalefactor" components to 1.0.  For a vertex component, it moves each base
vertex "a" through displacement "r" along the unit vector "u" and then multiples each
component of this sum by the appropriate component of "scalefactor"; it then resets each
vertex displacement "r" to 0.0 and all "scalefactor" components to 1.0.

The resulting model is identical to the input model; however, when the output model is
(later) written to a mod file, the finite precision of the "printf" statements may cause
small differences between input and output.

Since there is no simple way to scale harmonic "a" and "b" coefficients when the x, y,
and z dimensions are being scaled differently from each other, any attempt to call this
routine for a harmonic component with nonidentical "scalefactor" components causes the
program to quit with an error message telling the user to run the standalone "mkharmod"
program.

Modified 2013 May 20 by CM:
    Implement ovoid shape components

Modified 2010 June 1 by CM:
    Change scalefactor parameter from a scalar to a 3-component vector
        for harmonic and vertex shape structures

Modified by CM on 2006 October 1:
    Permit ellipsoid and harmonic components
    Add "scalefactor" to harmonic and vertex shape structures

Modified by CM on 2005 January 25:
    Changed routine from type "int" to "void"

Modified by CM on 2004 June 27:
    Added check that all model components are vertex components
*****************************************************************************************/

#include "head.h"

void ref_mod( struct mod_t *mod)
{
  int c, i, j, L, l, m;

  for (c=0; c<mod->shape.ncomp; c++) {
    switch (mod->shape.comp[c].type) {
    case ELLIPSE:
    case OVOID:
      break;
    case HARMONIC:
      if ((mod->shape.comp[c].desc.har.scalefactor[0].val
                       != mod->shape.comp[c].desc.har.scalefactor[1].val) ||
          (mod->shape.comp[c].desc.har.scalefactor[0].val
                       != mod->shape.comp[c].desc.har.scalefactor[2].val)    ) {
        printf("WARNING: harmonic component with unequal 'scale factor' values\n");
        printf("         -- can't use 'refshape' action, run 'mkharmod' instead\n");
        bailout("ref_mod.c\n");
      }
      L = mod->shape.comp[c].desc.har.nhar;
      for (l=0; l<=L; l++) {
        mod->shape.comp[c].desc.har.a[l][0].val *=
                     mod->shape.comp[c].desc.har.scalefactor[0].val;
        for (m=1; m<=l; m++) {
          mod->shape.comp[c].desc.har.a[l][m].val *=
                       mod->shape.comp[c].desc.har.scalefactor[0].val;
          mod->shape.comp[c].desc.har.b[l][m].val *=
                       mod->shape.comp[c].desc.har.scalefactor[0].val;
        }
      }
      for (j=0; j<=2; j++)
        mod->shape.comp[c].desc.har.scalefactor[j].val = 1.0;
      break;
    case VERTEX:
      for (i=0; i<mod->shape.comp[c].desc.ver.nv; i++) {
        for (j=0; j<=2; j++) {
          mod->shape.comp[c].desc.ver.v[i].a[j] +=
                        mod->shape.comp[c].desc.ver.v[i].u[j] *
                        mod->shape.comp[c].desc.ver.v[i].r.val;
          mod->shape.comp[c].desc.ver.v[i].a[j] *=
                               mod->shape.comp[c].desc.ver.scalefactor[j].val;
        }
        mod->shape.comp[c].desc.ver.v[i].r.val = 0.0;
      }
      for (j=0; j<=2; j++)
        mod->shape.comp[c].desc.ver.scalefactor[j].val = 1.0;
      break;
    default:
      bailout("ref_mod.c: can't do that component type\n");
    }
  }
}
