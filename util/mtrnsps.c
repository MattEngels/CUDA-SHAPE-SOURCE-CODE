/***************************************************************************

								 mtrnsps.c

performs a matrix transpose operation.

***************************************************************************/

#include "basic.h"

void mtrnsps( double a[3][3], double b[3][3])
{
  double t[3][3];
  int i, j;

  for (i=0;i<=2;i++)
	for (j=0;j<=2;j++)
	  t[i][j] = b[j][i];
  for (i=0;i<=2;i++)
	for (j=0;j<=2;j++)
	  a[i][j] = t[i][j];
}
