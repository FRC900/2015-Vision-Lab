
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/opencv.hpp>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include <sstream>

using namespace std;
using namespace cv;

void scalefactor(Mat inputimage, Size objectsize, Size minsize, Size maxsize, float scaleFactor, vector<Mat> &images, vector<float> &scales) 
{
	images.clear(); 
	scales.clear();
	/*
	Loop multiplying the image size by the scalefactor upto the maxsize	
	Store each image in the images vector
	Store the scale factor in the scales vector 
	*/

	//for(Size i = objectsize; i < maxsize;)
	
	//only works for square image
	float scale = objectsize.width / minsize.width;

	while(scale > objectsize.width / maxsize.width)
	{	
		//set objectsize.width to scalefactor * objectsize.width
		//set objectsize.height to scalefactor * objectsize.height
		Mat outputimage;
		resize(inputimage, outputimage, Size(), scale, scale);
		
		images.push_back(outputimage);
		scales.push_back(scale);

		scale /= scaleFactor;		
	
	}	

}

int main(int argc, char** argv)
{
	Mat input = imread("/home/ubuntu/2015-Vision-Lab/chroma_key_bins/image03.png", CV_LOAD_IMAGE_COLOR);
	Size minsize(50,50);
	Size maxsize(400,400);
	vector<Mat> images;
	vector<float> scales;

	scalefactor(input, input.size(), minsize, maxsize, 1.5f, images, scales);
	while(1)
	{
		imshow("image", input); 
		for(int i = 0; i < images.size(); i++)
		{
		stringstream label; 
		label << i;
		imshow(label.str(), images[i]);
		}

		if (waitKey(5) >= 0) break;
	}

	
		

	

	
	

	return 0;
}
