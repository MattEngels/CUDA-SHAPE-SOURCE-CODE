/***************************************************************************

								fiteuler.c

***************************************************************************/


#include "basic.h"
#include "../nr/nr.h"
#include "util.h"

double eulererr( double *p);
static double meuler[3][3];

double fiteuler( double m[3][3], double *phi, double *theta, double *psi)
{
  double *p, **xi, err;
  int i, j;

  p = vector( 1, 3);
  xi = matrix( 1, 3, 1, 3);
  for (i=1;i<=3;i++) {
	for (j=1;j<=3;j++)
	  xi[i][j] = 0.0;
	xi[i][i] = 1.0;
  }
  for (i=0;i<=2;i++)
	for (j=0;j<=2;j++)
	  meuler[i][j] = m[i][j];
  mat2euler( m, &p[1], &p[2], &p[3]);
  powell( p, xi, 3, 1.0e-6, &i, &err, eulererr);
  (*phi) = p[1];
  (*theta) = p[2];
  (*psi) = p[3];
  free_matrix( xi, 1, 3, 1, 3);
  free_vector( p, 1, 3);
  return err;
}


double eulererr( double *p)
{
  double mt[3][3], err;
  int i, j;

  err = 0.0;
  euler2mat( mt, p[1], p[2], p[3]);
  for (i=0;i<=2;i++)
	for (j=0;j<=2;j++)
	  err += pow( mt[i][j]-meuler[i][j], 2.0);
  return err;
}
