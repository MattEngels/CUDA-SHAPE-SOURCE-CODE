/***************************************************************************
                                                               swapendian.c

Test whether or not a program is running on a little-endian machine

EXAMPLE: if (is_little_endian())
           printf("Running Linux on a PC\n");
         else
           printf("Running on Sun or Mac\n");

Swap the byte order of a scalar

EXAMPLE: float diameter;
         fread( diameter, sizeof(float), 1, fp);
         float radius = 0.5*swap_float( diameter);

Swap the byte order of a structure

EXAMPLE: create and read a 10-element float buffer and swap bytes

         float *fbuf = (float *) calloc( 10, sizeof(float));
         fread( fbuf, sizeof(float), 10, fp);
         swapbuf_float( fbuf);

Written 2005 June 15 by CM, swap_float() by JLM

Modified 2005 June 21 by CM:
    generalized swap_float and created versions for other variable types
***************************************************************************/

#include "basic.h"

int is_little_endian()
{
  /*  Returns 1 on a little-endian machine, 0 on a big-endian machine  */

  int i = 1;
  char c = *(char *)&i;
  return (int) c;
}

short swap_short(short x)
{
  char in[16], out[16];
  int varsize, i;
  short y;

  varsize = sizeof(short);
  memcpy(in, &x, varsize);
  for (i=0; i<varsize; i++)
    out[i] = in[varsize - 1 - i];
  memcpy(&y, out, varsize);

  return y;
}

unsigned short swap_ushort(unsigned short x)
{
  char in[16], out[16];
  int varsize, i;
  unsigned short y;

  varsize = sizeof(unsigned short);
  memcpy(in, &x, varsize);
  for (i=0; i<varsize; i++)
    out[i] = in[varsize - 1 - i];
  memcpy(&y, out, varsize);

  return y;
}

int swap_int(int x)
{
  char in[16], out[16];
  int varsize, i;
  int y;

  varsize = sizeof(int);
  memcpy(in, &x, varsize);
  for (i=0; i<varsize; i++)
    out[i] = in[varsize - 1 - i];
  memcpy(&y, out, varsize);

  return y;
}

unsigned int swap_uint(unsigned int x)
{
  char in[16], out[16];
  int varsize, i;
  unsigned int y;

  varsize = sizeof(unsigned int);
  memcpy(in, &x, varsize);
  for (i=0; i<varsize; i++)
    out[i] = in[varsize - 1 - i];
  memcpy(&y, out, varsize);

  return y;
}

long swap_long(long x)
{
  char in[16], out[16];
  int varsize, i;
  long y;

  varsize = sizeof(long);
  memcpy(in, &x, varsize);
  for (i=0; i<varsize; i++)
    out[i] = in[varsize - 1 - i];
  memcpy(&y, out, varsize);

  return y;
}

unsigned long swap_ulong(unsigned long x)
{
  char in[16], out[16];
  int varsize, i;
  unsigned long y;

  varsize = sizeof(unsigned long);
  memcpy(in, &x, varsize);
  for (i=0; i<varsize; i++)
    out[i] = in[varsize - 1 - i];
  memcpy(&y, out, varsize);

  return y;
}

float swap_float(float x)
{
  char in[16], out[16];
  int varsize, i;
  float y;

  varsize = sizeof(float);
  memcpy(in, &x, varsize);
  for (i=0; i<varsize; i++)
    out[i] = in[varsize - 1 - i];
  memcpy(&y, out, varsize);

  return y;
}

double swap_double(double x)
{
  char in[16], out[16];
  int varsize, i;
  double y;

  varsize = sizeof(double);
  memcpy(in, &x, varsize);
  for (i=0; i<varsize; i++)
    out[i] = in[varsize - 1 - i];
  memcpy(&y, out, varsize);

  return y;
}

long double swap_ldouble(long double x)
{
  char in[16], out[16];
  int varsize, i;
  long double y;

  varsize = sizeof(long double);
  memcpy(in, &x, varsize);
  for (i=0; i<varsize; i++)
    out[i] = in[varsize - 1 - i];
  memcpy(&y, out, varsize);

  return y;
}

void swapbuf_short(short *buf, int nvals)
{
  static char swapped[16];
  int varsize, n, i;
  long bytesSoFar;

  varsize = sizeof(short);
  for (n=0, bytesSoFar=0; n<nvals; n++, bytesSoFar+=varsize) {
    for (i=0; i<varsize; i++)
      swapped[i] = *((char *)buf + bytesSoFar + varsize - 1 - i);
    buf[n] = *(short *) swapped;
  }
}

void swapbuf_ushort(unsigned short *buf, int nvals)
{
  static char swapped[16];
  int varsize, n, i;
  long bytesSoFar;

  varsize = sizeof(unsigned short);
  for (n=0, bytesSoFar=0; n<nvals; n++, bytesSoFar+=varsize) {
    for (i=0; i<varsize; i++)
      swapped[i] = *((char *)buf + bytesSoFar + varsize - 1 - i);
    buf[n] = *(unsigned short *) swapped;
  }
}

void swapbuf_int(int *buf, int nvals)
{
  static char swapped[16];
  int varsize, n, i;
  long bytesSoFar;

  varsize = sizeof(int);
  for (n=0, bytesSoFar=0; n<nvals; n++, bytesSoFar+=varsize) {
    for (i=0; i<varsize; i++)
      swapped[i] = *((char *)buf + bytesSoFar + varsize - 1 - i);
    buf[n] = *(int *) swapped;
  }
}

void swapbuf_uint(unsigned int *buf, int nvals)
{
  static char swapped[16];
  int varsize, n, i;
  long bytesSoFar;

  varsize = sizeof(unsigned int);
  for (n=0, bytesSoFar=0; n<nvals; n++, bytesSoFar+=varsize) {
    for (i=0; i<varsize; i++)
      swapped[i] = *((char *)buf + bytesSoFar + varsize - 1 - i);
    buf[n] = *(unsigned int *) swapped;
  }
}

void swapbuf_long(long *buf, int nvals)
{
  static char swapped[16];
  int varsize, n, i;
  long bytesSoFar;

  varsize = sizeof(long);
  for (n=0, bytesSoFar=0; n<nvals; n++, bytesSoFar+=varsize) {
    for (i=0; i<varsize; i++)
      swapped[i] = *((char *)buf + bytesSoFar + varsize - 1 - i);
    buf[n] = *(long *) swapped;
  }
}

void swapbuf_ulong(unsigned long *buf, int nvals)
{
  static char swapped[16];
  int varsize, n, i;
  long bytesSoFar;

  varsize = sizeof(unsigned long);
  for (n=0, bytesSoFar=0; n<nvals; n++, bytesSoFar+=varsize) {
    for (i=0; i<varsize; i++)
      swapped[i] = *((char *)buf + bytesSoFar + varsize - 1 - i);
    buf[n] = *(unsigned long *) swapped;
  }
}

void swapbuf_float(float *buf, int nvals)
{
  static char swapped[16];
  int varsize, n, i;
  long bytesSoFar;

  varsize = sizeof(float);
  for (n=0, bytesSoFar=0; n<nvals; n++, bytesSoFar+=varsize) {
    for (i=0; i<varsize; i++)
      swapped[i] = *((char *)buf + bytesSoFar + varsize - 1 - i);
    buf[n] = *(float *) swapped;
  }
}

void swapbuf_double(double *buf, int nvals)
{
  static char swapped[16];
  int varsize, n, i;
  long bytesSoFar;

  varsize = sizeof(double);
  for (n=0, bytesSoFar=0; n<nvals; n++, bytesSoFar+=varsize) {
    for (i=0; i<varsize; i++)
      swapped[i] = *((char *)buf + bytesSoFar + varsize - 1 - i);
    buf[n] = *(double *) swapped;
  }
}

void swapbuf_ldouble(long double *buf, int nvals)
{
  static char swapped[16];
  int varsize, n, i;
  long bytesSoFar;

  varsize = sizeof(long double);
  for (n=0, bytesSoFar=0; n<nvals; n++, bytesSoFar+=varsize) {
    for (i=0; i<varsize; i++)
      swapped[i] = *((char *)buf + bytesSoFar + varsize - 1 - i);
    buf[n] = *(long double *) swapped;
  }
}
