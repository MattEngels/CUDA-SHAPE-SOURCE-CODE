/***************************************************************************

vav.c returns the average value of a double vector

 ***************************************************************************/

double vav( double *v, int n)
{
  double av=0;
  int i;

  for (i=0;i<n;i++)
    av += v[i];
  return av/n;
}
