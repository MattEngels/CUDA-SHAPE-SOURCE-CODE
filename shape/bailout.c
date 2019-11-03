/***************************************************************************
                                                                  bailout.c

Print an error message and exit shape

Modified 2009 April 3 by CM:
    Add initial "fflush( stdout)" call so that the printed error message
        appears at the end of the screen display

Modified 2005 July 15 by CM and MCN:
    Add call to MPI_Finalize or (for parallel processing) MPI_Abort
    Move routine from ../util to shape directory
***************************************************************************/

#include "head.h"

void bailout(const char *message)
{
  fflush( stdout);
  fprintf( stderr, "ERROR: %s", message);
  // MPI_Finalize();  This should be a function that wraps everything up.
  exit(2);
}
