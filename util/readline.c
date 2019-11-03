/*  Get string input from a file without overflowing the buffer  */
/*  Added 2003 April 18 by CM                                    */
/*  Modified 2012 Feb 1 by CM: changed name from "getline" to "readline"  */
/*      to avoid conflict with getline routine in stdio.h                 */


#include "basic.h"

int readline(FILE *fp, char line[], int maxSize)
{
  int n, limit;
  char c;
    
  n = 0;
  limit = maxSize;

  while (--limit > 0)
    {
    c = getc(fp);
    if (c == EOF || c == '\n')
        limit = 0;
    else
        line[n++] = c;
    }
  line[n] = '\0';

  fflush(fp);
        
  return n;
}
