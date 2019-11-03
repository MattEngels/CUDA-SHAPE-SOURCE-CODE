/***************************************************************************

vmin.c returns the minimum element value of a double vector

 ***************************************************************************/

double vmin( double *v, int n)
{
  double min;
  int i;

  min = v[0];
  for (i=1;i<n;i++)
    if (v[i]<min)
      min = v[i];
  return min;
}
