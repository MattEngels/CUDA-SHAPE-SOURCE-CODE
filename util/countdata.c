/*****************************************************************************************
                                                                              countdata.c

Count the number of valid data entries (ignoring comments enclosed between curly braces)
in a file, from the current file position to the end of the file.  The file position
indicator is reset to its initial value after the counting is finished.  Return the
number of entries found, unless an unclosed comment is encountered, in which case exit
with an error message.

Written 2013 June 17 by CM, based on gettstr.c
*****************************************************************************************/

#include "basic.h"
#include <ctype.h>

int countdata( FILE *fp)
{
  int ndata, c;
  fpos_t savefilepos;

  /*  Save the current file position, initialize the number of data entries found
      (ignoring comments), and enter the main loop for finding new data entries    */

  fgetpos(fp, &savefilepos);
  ndata = 0;

  do {

      /*  Look for the next data entry (ignoring comments)  */

      do {
        c = fgetc(fp);
    
        /*  If end-of-file is found outside a comment, there were no more valid data
            in the file, so move the file position indicator back to its initial
            setting and return the number of data entries found up to this point      */

        if (c == EOF) {
          fsetpos(fp, &savefilepos);
          return ndata;
        }

        /*  If a comment is starting, keep reading until the closing
            brace, then go back to the beginning of the loop.
            If EOF is found in comment, then terminate.               */

        if (c == '{') {
          while ((c = fgetc(fp)) != '}')
            if (c == EOF) {
              printf("ERROR: no valid string available, unclosed comment\n");
              fprintf( stderr, "ERROR: countdata.c\n");
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
            fprintf(stderr, "ERROR: countdata.c\n");
            exit(2);
          }
    
      } while (c != EOF && !isspace(c) && c != '{');

      /*  Got a valid string, so add another entry to the count
          and loop back to look for the next data entry          */

      ndata++;

  } while (c != EOF);

  /*  Hit EOF: Reset the file position indicator to its
      initial value and return the number of data entries found  */

  fsetpos(fp, &savefilepos);
  return ndata;
  
}
