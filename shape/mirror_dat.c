/***************************************************************************
                                                               mirror_dat.c
  
Adjusts each dataset's angle and spin offsets for the "mirror" action

Modified 2009 November 15 by CM:
    Change from int to void

Written 2009 April 1 by CM
***************************************************************************/

#include "head.h"


void mirror_dat( struct dat_t *dat)
{
  int s;

  /*  For each dataset, negate the second Euler angle offset (since the
      mirror action flips the pole) and the x-component of the spin offset  */

  for (s=0; s<dat->nsets; s++) {
    dat->set[s].angleoff[1].val *= -1;
    dat->set[s].omegaoff[0].val *= -1;
  }
}
