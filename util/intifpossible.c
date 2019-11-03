/***************************************************************************
                                                            intifpossible.c

Convert a double-precision value to a string:
    integer format if the value is within the specified absolute tolerance
        of an integer;
    floating-point otherwise, using the specified floating-point format
        string (e.g., "%f")

valstring is the string which will contain the converted value; n is its
    maximum allowed length (including the null character at the end)

If the specified absolute tolerance is negative, a default value is used

If the conversion is successful, a pointer to the string is returned;
    if there is an error or if the converted value extends beyond the
    end of valstring, NULL is returned.

Added 2005 March 11 by CM

Modified 2005 June 27 by CM:
    Renamed "round" function to "iround" to avoid conflicts
***************************************************************************/

#include "basic.h"
#include "../util/util.h"

#define DEFAULTABSTOL 1.0e-10

char *intifpossible(char *valstring, int n, double val, double abstol,
                    const char *floatformat)
{
  double abstol_use;
  int nchars;

  abstol_use = (abstol >= 0.0) ? abstol : DEFAULTABSTOL;

  if (fabs(val - iround( val)) <= abstol_use)
    nchars = sprintf(valstring, "%d", iround( val));
  else
    nchars = sprintf(valstring, floatformat, val);

  return (nchars != EOF && nchars < n) ? valstring: NULL;
}

#undef DEFAULTABSTOL
