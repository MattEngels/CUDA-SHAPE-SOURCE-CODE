/***************************************************************************
                                                               timestring.c

Create a string from the current date/time.  The user can input a format
string to be used by the "strftime" function; if a null string is given,
a default format is used.

The output string is set to the null string if it would otherwise be longer
than the specified maximum string length

Written 2008 April 10 by CM
***************************************************************************/

#include "basic.h"
#include <time.h>

void timestring(char *outstring, int maxSize, char *formatstring)
{
  const char *defaultformat = "%Y %b %d %H:%M:%S %Z";
  char tempstring[255];
  time_t curtime;
  struct tm *loctime;

  /*  Generate a numerical string from the current local date and time  */

  curtime = time(NULL);
  loctime = localtime(&curtime);
  if (!strcmp(formatstring, ""))
    strftime(tempstring, 255, defaultformat, loctime);
  else
    strftime(tempstring, 255, formatstring, loctime);

  /*  Copy this string to the output string if it will fit  */

  if (strlen(tempstring) < maxSize)
    strcpy(outstring, tempstring);
  else
    strcpy(outstring, "");

  return;
}
