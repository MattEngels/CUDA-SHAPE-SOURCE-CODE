/***************************************************************************
                                                              changepath.c

Change the path for a filename (instring).  If the output string would be
too long for the number of bytes provided, set it to the null string
instead.

Written 2004 November 30 by CM
***************************************************************************/

#include "basic.h"

void changepath(char *outstring, int maxSize, char *instring, char *newpath)
{
  char *lastslash_ptr, *filename_ptr;
  char separator[2];
  int newpathLen, outLen;

  /*  If the input string starts with a path, the filename starts
      one character past the last slash in the input string        */

  lastslash_ptr = strrchr(instring, '/');
  filename_ptr = (lastslash_ptr) ? lastslash_ptr+1 : instring;

  /*  Check whether or not we need to insert a slash after the new path  */

  newpathLen = strlen(newpath);
  if (newpathLen > 0 && newpath[newpathLen-1] != '/')
    strcpy(separator, "/");
  else
    strcpy(separator, "");

  /*  Create the output string  */

  outLen = strlen(filename_ptr) + strlen(separator) + newpathLen;
  if (outLen < maxSize) {
      strcpy(outstring, newpath);
      strcat(outstring, separator);
      strcat(outstring, filename_ptr);
  } else {
      strcpy(outstring, "");
  }

  return;
}
