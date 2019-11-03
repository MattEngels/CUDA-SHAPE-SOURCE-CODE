/*  Check whether or not a string is all whitespace  */
/*  Added 2003 April 23 by CM                        */

#include "basic.h"
#define TRUE 1
#define FALSE 0

int allwhite(char *string)
{
  int n, len;
  char c;
    
  len = strlen(string);

  for (n=0; n<len; n++) {
    c = *(string + n);
    if (c != ' ' && c != '\t' && c != '\n')
        return FALSE;
  }

  return TRUE;
}

#undef TRUE
#undef FALSE
