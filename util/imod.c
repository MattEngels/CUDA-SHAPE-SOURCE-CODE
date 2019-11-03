/***************************************************************************

								  imod.c

returns the value of an integer i modulo the integer n.
***************************************************************************/

#include "basic.h"

int imod( int i, int n)
{
  int m;

  m = i;
  while (m>=n) {
	m -= n;
  }
  return m;
}
