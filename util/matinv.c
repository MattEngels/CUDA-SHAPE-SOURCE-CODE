/***************************************************************************

								 matinv.c

replaces the matrix m[1..n][1..n] (allocated by NR matrix routine) with its
inverse. Ultimately this should return the condition number of the matrix.

***************************************************************************/

#include "basic.h"
#include "../nr/nr.h"

double matinv( double **m, int n)
{
  double **inv, d, *col;
  int *indx, i, j;

  indx = ivector( 1, n);
  inv = matrix( 1, n, 1, n);
  col = vector( 1, n);
  ludcmp( m, n, indx, &d);
  for (j=1;j<=n;j++) {
	for (i=1;i<=n;i++)
	  col[i] = 0.0;
	col[j] = 1.0;
	lubksb( m, n, indx, col);
	for (i=1;i<=n;i++)
	  inv[i][j] = col[i];
  }
  for (i=1;i<=n;i++)
	for (j=1;j<=n;j++)
	  m[i][j] = inv[i][j];
  free_vector( col, 1, n);
  free_matrix( inv, 1, n, 1, n);
  free_ivector( indx, 1, n);
  return 0.0;
}
