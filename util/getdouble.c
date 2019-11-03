/***************************************************************************
                                                                getdouble.c

Read a floating-point number from a file, skipping over any whitespace and
any comments enclosed between curly braces

Modified 2008 July 11 by CM:
     Split off the "string_to_double" routine so that it can be called by
         other routines

Modified 2006 April 10 by PT:
     Add if statement to catch end-of-file read by gettstr.c

Modified 2005 July 20 by CM:
    Use strtod rather than atof to get floating-point value from input
        string, so that we can test that *all* of the string was
        successfully converted
***************************************************************************/

#include "basic.h"
#include "../util/util.h"


double string_to_double( char *str)
{
  char *unconverted;
  double doubleval;

  /*  Test that a string represents a floating-point number;
      if it does, return it as a double-precision number      */

  doubleval = strtod( str, &unconverted);
  if (strlen(unconverted) > 0) {
    printf("ERROR: expected to read floating-point number, got %s instead\n", str);
    fprintf( stderr, "ERROR: getdouble.c\n");
    exit(2);
  }

  return doubleval;

}


double getdouble( FILE *fp)
{
  char tmp[256];

  /*  Read the next nonblank string (skipping any comments)  */

  gettstr( fp, tmp);

  if (!strcmp( tmp, "\0")) {
    fprintf( stderr, "ERROR: end of file (instead of double) read by gettstr.c\n");
    exit(2);
  }

  /*  Convert the string to a double-precision number  */

  return string_to_double( tmp);

}
