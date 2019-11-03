/*****************************************************************************************
                                                                             nomoredata.c

Check that there are no more valid data (ignoring comments enclosed between curly braces)
in a file.  The file position indicator is reset to its initial value after this check has
been performed.  Return 1 if there are indeed no more data, 0 if there are, and exit if
there's an unclosed comment.

The idea is that the user expects to be at the end of a file (other than perhaps
whitespace and comments) and will consider the existence of more data in the file to be an
error or warning condition.  This could happen, say, if a lightcurve datafile has more
points than the obs file claims.

Modified 2013 June 17 by CM:
    Reset file position indicator to its initial setting before returning

Modified 2009 November 15 by CM:
    Include ctype.h for "isspace" function

Written 2007 August 4 by CM, based on gettstr.c
*****************************************************************************************/

#include "basic.h"
#include <ctype.h>

int nomoredata( FILE *fp)
{
  int c; 
  fpos_t savefilepos;

  /*  Save the current file position  */

  fgetpos(fp, &savefilepos);

  /*  Check for a valid string (ignoring comments)  */

  do {
    c = fgetc(fp);
    
    /*  If end-of-file is found outside a comment, there were no more
        valid data in the file, so move the file position indicator
        back to its initial setting and return 1                       */

    if (c == EOF) {
      fsetpos(fp, &savefilepos);
      return 1;
    }

    /*  If a comment is starting, keep reading until the closing
        brace, then go back to the beginning of the loop.
        If EOF is found in comment, then terminate.               */

    if (c == '{') {
      while ((c = fgetc(fp)) != '}')
        if (c == EOF) {
	  printf("ERROR: no valid string available, unclosed comment\n");
	  fprintf( stderr, "ERROR: nomoredata.c\n");
	  exit(2);
        }
      c = fgetc(fp);
    }

  } while (isspace(c));

  /*  Read the word until a whitespace, EOF, or comment  */

  do {

    /*  Read the next character unless it is a opening brace in which
        case it has to be sent back to the buffer before finishing     */

    if ((c = fgetc(fp)) == '{')
      if (ungetc(c, fp) == EOF) {
	printf("ERROR: buffering must have been deactivated, put space or newline before comments\n");
        fprintf(stderr, "ERROR: nomoredata.c\n");
        exit(2);
      }
    
  } while (c != EOF && !isspace(c) && c != '{');

  /*  Got a valid string, so reset the file position indicator to its
      initial value and return 0: there were NOT no more data in the file  */

  fsetpos(fp, &savefilepos);
  return 0;
  
}
