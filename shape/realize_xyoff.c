/***************************************************************************
                                                            realize_xyoff.c
  
Implements the '=' state for horizontal and vertical offsets for
plane-of-sky frames (optical images):

For each plane-of-sky frame whose horizontal offset has the '=' state, go
backwards within that dataset until we find a frame whose horizontal
offset has state 'f' or 'c', and copy this value.  Do the same for
vertical offsets.  It's legal for a frame to use the '=' state for just
one of its two offsets.

There is no way to use the '=' state to copy offsets from a frame in one
plane-of-sky dataset to a frame in another plane-of-sky dataset.

Modified 2016 July 7 by Matt Engels:
	Adapted for use in shape-cuda.

Written 2005 February 24 by CM
***************************************************************************/

#include "head.h"

void realize_xyoff( struct dat_t *dat)
{
  int s, f, j;

  for (s=0; s<(*dat).nsets; s++) {

    if ((*dat).set[s].type == POS) {

      for (j=0; j<=1; j++)
        if ((*dat).set[s].desc.poset.frame[0].off[j].state == '=')
          bailout("can't use \"=\" state for the first frame in a plane-of-sky dataset\n");

      for (f=1; f<(*dat).set[s].desc.poset.nframes; f++)
        for (j=0; j<=1; j++)
          if ((*dat).set[s].desc.poset.frame[f].off[j].state == '=')
            (*dat).set[s].desc.poset.frame[f].off[j].val =
                          (*dat).set[s].desc.poset.frame[f-1].off[j].val;

    }
  }
}
