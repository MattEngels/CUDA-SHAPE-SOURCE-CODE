/***************************************************************************

							   cotrans.c

Performs a coordinate transformation y=ax id dir=1.  If dir=-1 then
y=(aT)x.

***************************************************************************/

#include "basic.h"

void cotrans( double y[3], double a[3][3], double x[3], int dir)
{
  double t[3];
  int i, j;

  if (dir==1)
	for (i=0;i<=2;i++) {
	  t[i] = 0.0;
	  for (j=0;j<=2;j++)
		t[i] += a[i][j]*x[j];
	}
  if (dir==(-1))
	for (i=0;i<=2;i++) {
	  t[i] = 0.0;
	  for (j=0;j<=2;j++)
		t[i] += a[j][i]*x[j];
	}
  for (i=0;i<=2;i++)
	y[i] = t[i];
}

