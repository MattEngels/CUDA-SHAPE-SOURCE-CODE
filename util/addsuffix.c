/***************************************************************************
                                                               addsuffix.c

Take an input string (filename) and append a new suffix.  If the input
string already ends in the new suffix, leave it unchanged; if instead it
ends in a specified old suffix, remove that old suffix before appending
the new suffix.  If the output string would be too long for the number of
bytes provided, set it to the null string instead.  Return 1 if the input
string already ends in the new suffix, 0 otherwise.

Modified 2009 August 9 by CM:
    Change routine from void to int so that we can return information on
        whether or not the old suffix was found

Written 2004 November 30 by CM
***************************************************************************/

#include "basic.h"

int addsuffix(char *outstring, int maxSize,
              char *instring, char *oldsuffix, char *newsuffix)
{
  char *in_ptr;
  size_t inLen, outLen, oldsufLen, newsufLen, n_copy;
  int found_oldsuffix;

  inLen = strlen(instring);
  oldsufLen = strlen(oldsuffix);
  newsufLen = strlen(newsuffix);
  outLen = inLen + newsufLen;  /* for starters */

  /*  Check whether or not the input string ends in the old suffix  */

  found_oldsuffix = 0;
  if (oldsufLen > 0 && oldsufLen <= inLen) {
    in_ptr = instring + (inLen - oldsufLen);
    if (!strcmp(in_ptr, oldsuffix)) {
      found_oldsuffix = 1;
      outLen = inLen - oldsufLen + newsufLen;
    }
  }

  /*  If the input string doesn't end in the old suffix,
      check whether or not it already ends in the new suffix  */

  if (!found_oldsuffix && newsufLen > 0 && newsufLen <= inLen) {
    in_ptr = instring + (inLen - newsufLen);
    if (!strcmp(in_ptr, newsuffix))
      outLen = inLen;
  }

  /*  Create the output string: copy n_copy characters from the
      beginning of instring to the beginning of outstring, then
      concatenate the new suffix to the end of outstring.        */

  if (outLen < maxSize) {
      n_copy = outLen - newsufLen;
      strncpy(outstring, instring, n_copy);
      outstring[n_copy] = '\0';
      strcat(outstring, newsuffix);
  } else {
      strcpy(outstring, "");
  }

  return found_oldsuffix;
}
