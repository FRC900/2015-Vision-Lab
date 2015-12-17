#include <opencv2/imgproc/imgproc.hpp>

template<class MatT>
void scalefactor(MatT inputimage, cv::Size objectsize, cv::Size minsize, cv::Size maxsize, float scaleFactor, std::vector<MatT> &images, std::vector<float> &scales) 
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
	float scale = (float)objectsize.width / minsize.width;

	while(scale > (float)objectsize.width / maxsize.width)
	{	
		//set objectsize.width to scalefactor * objectsize.width
		//set objectsize.height to scalefactor * objectsize.height
		MatT outputimage;
		resize(inputimage, outputimage, cv::Size(), scale, scale);
		
		images.push_back(outputimage);
		scales.push_back(scale);

		scale /= scaleFactor;		
	
	}	
}

// Explicitly generate code for Mat and GpuMat options
template void scalefactor(cv::Mat inputimage, cv::Size objectsize, cv::Size minsize, cv::Size maxsize, float scaleFactor, std::vector<cv::Mat> &images, std::vector<float> &scales);
template void scalefactor(cv::gpu::GpuMat inputimage, cv::Size objectsize, cv::Size minsize, cv::Size maxsize, float scaleFactor, std::vector<cv::gpu::GpuMat> &images, std::vector<float> &scales);
