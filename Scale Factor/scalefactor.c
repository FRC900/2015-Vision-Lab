void scalefactor(Mat inputimage, Size objectsize, Size minsize, Size maxsize, float scalefactor, vector<Mat> &images, vector<float> &scales) 
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

		scale /= scalefactor;		
	
	}	

}
