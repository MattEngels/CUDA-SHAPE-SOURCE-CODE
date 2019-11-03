/*  Wait for a specified (floating-point) number of seconds  */
/*  Added 2005 January 14 by CM                              */

#include "basic.h"
#include <time.h>

void waitsecs( double seconds)
{
  clock_t endwait;

  endwait = clock() + (clock_t) (seconds*CLOCKS_PER_SEC);
  while (clock() < endwait) {}
}
