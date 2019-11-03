/***************************************************************************
                                                                readparam.c

reads a param_t variable (i.e., one with a state and a value).

Modified 2006 April 10 by PT:
    Add if statement to catch end-of-file read by gettstr.c

Modified 2005 February 13 by CM:
    Minor revisions (mostly aesthetic, add \n to bailout error message)
***************************************************************************/

#include "head.h"

int readparam( FILE *fp, struct param_t *p)
{
  char str[80];
  int nfloatpar=0;

  /* If not at end-of-file, check for c, f, or = state */

  if (gettstr( fp, str)){
    if (!strcmp( str, "c")) {
      (*p).state = 'c';
    } else if (!strcmp( str, "=")) {
      (*p).state = '=';
    } else if (!strcmp( str, "f")) {
      (*p).state = 'f';
      nfloatpar = 1;
    } else {
      printf("str = %s\n", str);
      bailout("readparam.c expected \"f\" or \"c\" or \"=\"\n");
    }

  /* Get value that goes with state found above */

    (*p).val = getdouble( fp);
  }

  else {  /* EOF */
    printf("WARNING: end of file (instead of c, f, or =) read by gettstr.c\n");
    nfloatpar = -1;
  }
  
  /* Return 1 if parameter is floating, 0 if c/= state, or -1 if EOF */

  return nfloatpar;
}

