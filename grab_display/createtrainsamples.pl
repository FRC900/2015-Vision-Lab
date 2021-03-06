#!/usr/bin/perl
use File::Basename;
use strict;
##########################################################################
# Detects object from background and resizes it randomly
# Creates distorted samples from images
# Shifts images dx dy ds for calibration
#
#
# Original Author: Naotoshi Seo
# Modified By: Benjamin Decker
# Date  : 12/13/2015 Almost total overhaul, revised to be specific to FRC Team 900's needs for neural networks.
# Date  : 09/12/2008 Add <totalnum> and <createsample_command_options> options
# Date  : 06/02/2007
# Date  : 03/12/2006
#########################################################################
my $cmd = '../opencv_createsamples_color/opencv_createsamples_color -bgcolor 0x96c997 -bgthresh 0x153768 -maxxangle 1.1 -maxyangle 1.1 -maxzangle 0.5 -maxidev 40 -w 128 -h 128 -hsv';
my $totalnum = 7000;
my $stage1 = "false";
my $stage2 = "true";
my $stage3 = "true";

if ($#ARGV < 1) {
    print "Usage: perl createtrainsamples.pl\n";
    print "  <input_dirname>\n";
    print "  <output_dirname>\n";
    print "  [<totalnum = " . $totalnum . ">]\n";
    print "  [<imagecreate = " . $stage1 . ">]\n";
    print "  [<distortion = " . $stage2 . ">]\n";
    print "  [<calibration = " . $stage3 . ">]\n";
    exit;
}
my $inputdir  = $ARGV[0];
my $outputdir = $ARGV[1];
$totalnum     = $ARGV[2] if ($#ARGV > 1);
$stage1       = $ARGV[3] if ($#ARGV > 2);
$stage2       = $ARGV[4] if ($#ARGV > 3);
$stage3       = $ARGV[5] if ($#ARGV > 4);

if($stage1 eq "true")
{
    opendir(VIDEO, "$inputdir");
    my @videos = ();
    while (my $file = readdir(VIDEO))
    {
        next unless $file =~ m/.+\.avi/;
       	print "$file\n";
        push @videos, $file;
    }
    closedir(VIDEO);
    for( my $k = 0; $k <= $#videos; $k++)
    {
        my $video = @videos[$k];
        print "./display -o $outputdir $inputdir/$video" . "\n";
        system("./display -o $outputdir $inputdir/$video");
    }
    $inputdir = $outputdir;
}
if($stage2 eq "true")
{
    opendir(POSITIVE, "$inputdir/images");
    my @positives = ();
    while (my $file = readdir(POSITIVE))
    {
        next unless $file =~ m/.+\.png/;
        push @positives, $file;
    }
    closedir(POSITIVE);

    # number of generated images from one image so that total will be $totalnum
    my $numfloor  = int($totalnum / ($#positives+1));
    my $numremain = $totalnum - $numfloor * ($#positives+1);

    for (my $k = 0; $k <= $#positives; $k++ ) {
        my $img = $positives[$k];
        my $num = ($k < $numremain) ? $numfloor + 1 : $numfloor;

        !chomp($img);
        my $imgdirlen = length(dirname($img));
        print "$cmd -img \"$inputdir/images/$img\" -pngfnformat $outputdir/images/distorted/$img%1.1d.png -num $num" . "\n";
        system("$cmd -img \"$inputdir/images/$img\" -pngfnformat $outputdir/images/distorted/$img%1.1d.png -num $num");
    }
    $inputdir = "$outputdir/images/distorted";
}
if($stage3 eq "true")
{
    for(my $k = 0; $k < 45; $k++)
    {
      mkdir "$outputdir/images/$k";
    }
    opendir(LINE, "$inputdir");
    my @calibration = ();
    while (my $file = readdir(LINE)) {
        next unless $file =~ m/.+\.png/;
        push @calibration, $file;
    }
    closedir(LINE);
    for( my $k = 0; $k <= $#calibration; $k++)
    {
        my $img = $calibration[$k];
        print "./imageShift $inputdir/$img $outputdir/images" . "\n";
        system("./imageShift $inputdir/$img $outputdir/images");
    }
}
