/***************************************************************************
                                                                   getint.c

Read a base-10 integer from a file, skipping over any whitespace and any
comments enclosed between curly braces

Modified 2008 July 11 by CM:
    Split off the "string_to_int" routine so that it can be called by
        other routines

Modified 2006 April 10 by PT:
    Add if statement to catch end-of-file read by gettstr.c

Modified 2005 July 20 by CM:
    Use strtol rather than atoi to get integer value from input string,
        so that we can test that *all* of the string was successfully
        converted

Modified 2005 July 1 by CM:
    exit rather than returning 0 if input is a nonnumeric string;
    exit rather than returning a truncated value if input is floating-point
***************************************************************************/

#include "basic.h"
#include <limits.h>
#include "../util/util.h"


int string_to_int( char *str)
{
  char *unconverted;
  long longval;
  int intval;

  /*  Test that a string represents a base-10 integer;
      if it does, return that integer                   */

  longval = strtol( str, &unconverted, 10);
  if (strlen(unconverted) > 0) {
      printf("ERROR: expected to read base-10 integer, got %s instead\n", str);
      fprintf( stderr, "ERROR: getint.c\n");
      exit(2);
  } else if (longval < INT_MIN || longval > INT_MAX) {
      printf("ERROR: input value %s is too large for the 'int' variable type\n", str);
      fprintf( stderr, "ERROR: getint.c\n");
      exit(2);
  }
  intval = (int) longval;

  return intval;

}


int getint( FILE *fp)
{
  char tmp[255];
  
  /*  Read the next nonblank string (skipping any comments)  */

  gettstr( fp, tmp);
  
  if (!strcmp( tmp, "\0")) {
    fprintf( stderr, "ERROR: end of file (instead of integer) read by gettstr.c\n");
    exit(2);
  }

  /*  Convert the string to a base-10 integer  */

  return string_to_int( tmp);

}
