#include <iostream>
#include <opencv2/highgui/highgui.hpp>

#include "scalefactor.hpp"

using namespace std;
using namespace cv;

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
