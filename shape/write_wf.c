/***************************************************************************

                                                                 write_wf.c

Outputs the model realization as a wavefront format file.  Can only be used
for a single component model.  File name is name of model file with .wf
appended.

Modified 2005 January 25 by CM:
    Removed unused variable
***************************************************************************/

#include "head.h"

void write_wf( struct mod_t *mod)
{
  char name[80];
  FILE *fp;
  int i;

  if (mod->shape.ncomp > 1)
    bailout("write_wf.c: doesn't work with more than one component.\n");
  sprintf( name, "%s.wf", mod->name);
  FOPEN( fp, name, "w");
  for (i=0; i<mod->shape.comp[0].real.nv; i++)
    fprintf( fp, "v %f %f %f\n", mod->shape.comp[0].real.v[i].x[0],
                                 mod->shape.comp[0].real.v[i].x[1],
                                 mod->shape.comp[0].real.v[i].x[2]);
  for (i=0; i<mod->shape.comp[0].real.nf; i++)
    fprintf( fp, "f %d %d %d\n", mod->shape.comp[0].real.f[i].v[0]+1,
                                 mod->shape.comp[0].real.f[i].v[1]+1,
                                 mod->shape.comp[0].real.f[i].v[2]+1);
  fclose( fp);
}
