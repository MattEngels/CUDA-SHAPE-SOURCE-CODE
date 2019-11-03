/***************************************************************************

								  mmmul.c

Performs a matrix-matrix multiplication: x = yz

***************************************************************************/

#include "basic.h"

void mmmul( double x[3][3], double y[3][3], double z[3][3])
{
  double t[3][3];
  int i, j, k;

  for (i=0;i<=2;i++)
	for (j=0;j<=2;j++) {
	  t[i][j] = 0.0;
	  for (k=0;k<=2;k++)
		t[i][j] += y[i][k]*z[k][j];
	}
  for (i=0;i<=2;i++)
	for (j=0;j<=2;j++)
	  x[i][j] = t[i][j];
}

