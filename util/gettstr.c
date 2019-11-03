/***************************************************************************
                                                                  gettstr.c

Read a character string from a file, skipping over any whitespace and any
comments enclosed between curly braces

Modified 2009 November 15 by CM:
    Include ctype.h for "isspace" function

Overhauled 2006 April 7 by PT (with the help of P. Rojo):
    Changed reading method from fscanf to fgetc to better handle
        new line characters and make more compatible with revised
        read_mod.c (reading optional lines for YORP)

Modified 2005 July 20 by CM:
    Added comments
***************************************************************************/

#include "basic.h"
#include "util.h"
#include <ctype.h>

int gettstr( FILE *fp, char *rstr)
{
  int c; 

  do{
    c = fgetc(fp);
    
    /* If end-of-file is found outside a comment, return 0 & set string to NULL*/

    if (c == EOF){
      printf("WARNING: no valid string available\n");
      *rstr = '\0';
      return 0;
    }

    /* If a comment is starting, keep reading until the closing brace, then go
     * back to beginning of loop. If EOF is found in comment, then terminate */

    if (c == '{'){
      while((c=fgetc(fp)) != '}')
        if (c == EOF){
	  printf("ERROR: no valid string available, unclosed comment\n");
	  fprintf( stderr, "ERROR: gettstr.c\n");
	  exit(2);
        }
      c = fgetc(fp);
    }

  }while(isspace(c));

  /* Read the word until a whitespace, EOF, or comment */

  do{

    /* Build the return string. */

    *rstr++ = c;

    /* Read next character unless it is a opening brace in which case it has to
     * be sent back to the buffer before finishing      */

    if((c=fgetc(fp)) == '{')
    	if(ungetc(c, fp) == EOF){
    		printf("ERROR: buffering must have been deactivated, put space or newline before comments\n");
    		fprintf(stderr, "ERROR: gettstr.c\n");
    		exit(2);
    	}
    
  }while (c != EOF && !isspace(c) && c != '{');

  *rstr = '\0';
  
  /* Got a valid string */

  return 1;
  
}
