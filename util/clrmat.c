#include "basic.h"

void clrmat( double **mat, int i1, int i2, int j1, int j2)
{
  int i, j;

  for (i=i1;i<=i2;i++)
	for (j=j1;j<=j2;j++)
	  mat[i][j] = 0.0;
}
