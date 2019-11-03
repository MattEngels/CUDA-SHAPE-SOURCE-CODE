/***************************************************************************
                                                                 tempname.c

Use the current date/time along with a user-specified prefix (e.g., path)
and suffix to create a filename that's likely to be unique, thus avoiding
the obnoxious gcc compiler warnings about the ANSI C tmpnam function and
its (unimportant for our purposes) deficiencies.

A non-null prefix is separated from the date/time string by "." and
similarly for a non-null suffix.

The output name is set to the null string if it would otherwise be longer
than the specified maximum length

Written 2007 August 23 by CM
***************************************************************************/

#include "basic.h"
#include <time.h>

void tempname(const char *outname, int maxSize, char *prefix, char *suffix)
{
  struct tm *ptr;
  time_t lt;
  char timestring[255];
  size_t prefixLen, timestringLen, suffixLen, outLen;

  /*  Generate a numerical string from the current local date and time:
      yyyymmddhhmmss                                                     */

  lt = time(NULL);
  ptr = localtime(&lt);
  strftime(timestring, 255, "%Y%m%d%H%M%S", ptr);

  /*  Figure out how long the output name will be  */

  prefixLen = strlen(prefix);
  timestringLen = strlen(timestring);
  suffixLen = strlen(suffix);
  outLen = prefixLen + timestringLen + suffixLen;
  if (prefixLen > 0)
    outLen++;
  if (suffixLen > 0)
    outLen++;

  /*  Add the prefix and suffix if the result won't be too long,
      otherwise set the output name to the null string            */

  if (outLen < maxSize) {
      strcpy(outname, prefix);
      if (prefixLen > 0)
        strcat(outname, ".");
      strcat(outname, timestring);
      if (suffixLen > 0) {
        strcat(outname, ".");
        strcat(outname, suffix);
      }
  } else {
      strcpy(outname, "");
  }

  return;
}
