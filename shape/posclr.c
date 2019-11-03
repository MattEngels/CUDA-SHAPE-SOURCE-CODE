/***************************************************************************
                                                                   posclr.c

Zeroes out a plane-of-sky (POS) image.

Modified 2007 August 4 by CM:
    Add body, bodyill, comp, and compill matrices to POS frames

Modified 2005 June 27 by CM:
    Rename INFINITY constant to HUGENUMBER to avoid conflicts

Modified 2005 June 22 by CM:
    Initialize facet numbers at POS pixel centers even if not bistatic

Modified 2005 January 20 by CM:
    Moved INFINITY constant from here to const.h

Modified 2004 February 10 by CM:
    Added comments
 ***************************************************************************/

#include "head.h"

void posclr( struct pos_t *pos)
{
	int i, j;

	/*  For each POS pixel, zero out the optical brightness (b) and
      cos(scattering angle), reset the z coordinate (distance from
      COM towards Earth) to a dummy value, and reset the body,
      component, and facet onto which the pixel center projects to
      dummy values                                                  */

	/* Debug notice:  pos->cosi[i][j] was inserted by Matt Engels on 4/28/17
	 * as this field kept getting overwritten by all frames from all sets.
	 */

	for (i=(-pos->n); i<=pos->n; i++)
		for (j=(-pos->n); j<=pos->n; j++) {
			pos->b[i][j] = pos->cose[i][j] = pos->cosi[i][j] =  0.0;
			pos->z[i][j] = -HUGENUMBER;
			pos->body[i][j] = pos->comp[i][j] = pos->f[i][j] = -1;
		}

	/*  In the x direction, reset the model's leftmost and rightmost
      pixel number to dummy values, and similarly for the y direction   */

	pos->xlim[0] = pos->ylim[0] =  pos->n;
	pos->xlim[1] = pos->ylim[1] = -pos->n;

	/*  For a bistatic situation (lightcurve or plane-of-sky dataset),
      zero out cos(incidence angle) and reset the distance towards the
      sun, the body, component, and facet numbers as viewed from the
      sun, and the model's maximum projected extent as viewed from the
      sun to dummy values                                               */

	if (pos->bistatic) {
		for (i=(-pos->n); i<=pos->n; i++)
			for (j=(-pos->n); j<=pos->n; j++) {
				pos->cosill[i][j] = 0.0;
				pos->zill[i][j] = -HUGENUMBER;
				pos->bodyill[i][j] = pos->compill[i][j] = pos->fill[i][j] = -1;
			}
		pos->xlim2[0] = pos->ylim2[0] =  pos->n;
		pos->xlim2[1] = pos->ylim2[1] = -pos->n;
	}
}
