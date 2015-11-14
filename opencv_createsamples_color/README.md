Fork of OpenCV 2.4.12.2 createsamples utility, modified to handle color images using -pngfnformat flag.

Changes 

Modified meaning of -bgvalue and -bgthresh. These can be 24-bit hex values (e.g. 0xaabbcc) for color operations, formatted as 0xrrggbb for both value and threshold.
Added -pngfnformat to take an [sf]printf format string. This is used to generate mulitple output png files from a single input. The output files will be shifted, modified in intensity and masked based various command line options
Passing in no background image will make the background randomized for color and grayscale inputs. This is only implemented for cases where the output is a .vec file (for grayscale) or for multi-PNG file output (for color ops)

