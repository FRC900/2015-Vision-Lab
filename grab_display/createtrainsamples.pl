#!/usr/bin/perl
use File::Basename;
use strict;
##########################################################################
# Create samples from an image applying distortions repeatedly 
# (create many many samples from many images applying distortions)
#
#  perl createtrainsamples.pl <positives.dat> <negatives.dat> <vec_output_dir>
#      [<totalnum = 7000>] [<createsample_command_options = ./createsamples -w 20 -h 20...>]
#  ex) perl createtrainsamples.pl positives.dat negatives.dat samples
#
# Author: Naotoshi Seo
# Date  : 09/12/2008 Add <totalnum> and <createsample_command_options> options
# Date  : 06/02/2007
# Date  : 03/12/2006
#########################################################################
my $cmd = '../opencv_createsamples_color/opencv_createsamples_color -bgcolor 0x96c997 -bgthresh 0x143667 -maxxangle 1.1 -maxyangle 1.1 -maxzangle 0.5 -maxidev 40 -w 128 -h 128 -hsv';
my $totalnum = 7000;

if ($#ARGV < 1) {
    print "Usage: perl createtrainsamples.pl\n";
    print "  <positives_files_dir>\n";
    print "  <output_dirname>\n";
    print "  [<totalnum = " . $totalnum . ">]\n";
    print "  [<createsample_command_options = '" . $cmd . "'>]\n";
    exit;
}
my $positive  = $ARGV[0];
my $outputdir = $ARGV[1];
$totalnum     = $ARGV[2] if ($#ARGV > 1);
$cmd          = $ARGV[3] if ($#ARGV > 2);

opendir(POSITIVE, "$positive");
my @positives = ();
while (my $file = readdir(POSITIVE)) {
    push @positives, $file;
}
print @positives;
closedir(POSITIVE);

# number of generated images from one image so that total will be $totalnum
my $numfloor  = int($totalnum / ($#positives+1));
my $numremain = $totalnum - $numfloor * ($#positives+1);

for (my $k = 0; $k <= $#positives; $k++ ) {
    my $img = $positives[$k];
    my $num = ($k < $numremain) ? $numfloor + 1 : $numfloor;

    #system("cat $tmpfile");

    !chomp($img);
    my $imgdirlen = length(dirname($img));
    print "$cmd -img \"$positive/$img\" -pngfnformat $outputdir/images/$img%1.1d.png -num $num" . "\n";
    system("$cmd -img \"$positive/$img\" -pngfnformat $outputdir/images/$img%1.1d.png -num $num");
}
