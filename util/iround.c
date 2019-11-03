/*  Modified 2005 June 27 by CM:
        Renamed function from "round" to "iround" (and filename
        from "round.c" to "iround.c") to avoid conflicts with
        built-in floating-point "round" function on some systems  */

#include "basic.h"

int iround(double x)
{
  if (x < 0.0)
    return ((int)(x - 0.5));
  else
    return ((int)(x + 0.5));
}
