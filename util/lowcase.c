/***************************************************************************

								 lowcase.c

makes all characters in a string lower case.
***************************************************************************/

#include "basic.h"
#include <ctype.h>

void lowcase( char *str)
{
  int n, i;

  n = strlen( str);
  for (i=0;i<n;i++)
	str[i] = tolower( str[i]);
}

