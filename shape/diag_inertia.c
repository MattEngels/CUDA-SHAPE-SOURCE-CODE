/*****************************************************************
                                                   diag_inertia.c

Diagonalize a model's inertia tensor in order to get the
principal moments of inertia and the transformation matrix
between body-fixed and principal-axis coordinates.

Written 2004 March 27 by CM
******************************************************************/

#include "head.h"

void diag_inertia( double inertia[3][3], double pmoment[3], double ap[3][3])
{

  int i, j, nrot;
  double **jaca, **jacv, *jacd;

  /*
      Use Numerical Recipes "jacobi" routine to diagonalize the inertia tensor:

      jaca = inertia tensor in body coordinates
                 (jacobi destroys the part of jaca above the diagonal)
      jacd = eigenvalues of inertia tensor = principal moments of inertia
      jacv = matrix whose columns are the eigenvectors of the inertia tensor:

      Each column of jacv is the direction cosines, in body coordinates, of the
      corresponding principal axis; jacv as a whole is the transformation matrix
      that takes us from principal-axis coordinates to body coordinates.

      (Vector pmoment and matrix ap are the same as jacd and jacv, respectively,
       but with rows and columns numbered 0-2 instead of 1-3.)
  */

  jaca = matrix( 1, 3, 1, 3);
  jacv = matrix( 1, 3, 1, 3);
  jacd = vector( 1, 3);

  for (i=1; i<=3; i++)
    for (j=1; j<=3; j++)
      jaca[i][j] = inertia[i-1][j-1];
  jacobi( jaca, 3, jacd, jacv, &nrot);    /*  nrot = # of rotations required  */

  for (i=0; i<=2; i++) {
    pmoment[i] = jacd[i+1];
    for (j=0; j<=2; j++)
      ap[i][j] = jacv[i+1][j+1];
  }

  free_matrix( jaca, 1, 3, 1, 3);
  free_matrix( jacv, 1, 3, 1, 3);
  free_vector( jacd, 1, 3);
}
