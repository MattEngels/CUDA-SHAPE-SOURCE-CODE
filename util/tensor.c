#include "basic.h"
#include "../nr/nr.h"

double ***tensor(int a1,int a2,int b1,int b2,int c1,int c2)
{
  double ***m;
  int i;
  double **matrix(int,int,int,int);


  m=(double ***) malloc((unsigned) (a2-a1+1)*sizeof(double**));
  if (!m) nrerror("allocation failure in tensor()");
  m -= a1;
  for (i=a1;i<=a2;i++)
    m[i] = matrix(b1,b2,c1,c2);
  return m;
}
