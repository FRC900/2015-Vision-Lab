#include <iostream>
#include "caffeclassifier.h"
#include "scalefactor.hpp"
#include "fast_nms.hpp"

double gtod_wrapper(void)
{
   struct timeval tv;
   gettimeofday(&tv, NULL);
   return (double)tv.tv_sec + (double)tv.tv_usec/1000000.0;
}

int main(int argc, char *argv[])
{
   if (argc != 6) 
   {
      std::cerr << "Usage: " << argv[0]
	 << " deploy.prototxt network.caffemodel"
	 << " mean.binaryproto labels.txt img.jpg" << std::endl;
      return 1;
   }
   ::google::InitGoogleLogging(argv[0]);
   std::string model_file   = argv[1];
   std::string trained_file = argv[2];
   std::string mean_file    = argv[3];
   std::string label_file   = argv[4];
   CaffeClassifier classifier(model_file, trained_file, mean_file, label_file, true, 3 );

   std::string file = argv[5];
   cv::Mat img = cv::imread(file, -1);
   CHECK(!img.empty()) << "Unable to decode image " << file;

   std::vector<cv::Mat> imgs;
   std::vector<cv::Rect> rects;

   std::vector<cv::Mat> scaledImages;
   std::vector<float> scales;

   std::vector<Detected> detected;

   cv::Size minSize(100,100);
   cv::Size maxSize(700,700);
   cv::Size classifierSize = classifier.getInputGeometry();

   int step = 6;
   //int step = std::min(img.cols, img.rows) *0.05;
   double start = gtod_wrapper();

   cv::Mat f32Img;
   img.convertTo(f32Img, CV_32FC3);
   scalefactor(f32Img, classifierSize, minSize, maxSize, 1.35, scaledImages, scales);
   int posCount = 0;
   for (size_t scale = 0; scale < scaledImages.size(); ++scale)
   {
      for (int r = 0; (r + classifierSize.height) < scaledImages[scale].rows; r +=step)
      {
	 for (int c = 0; (c + classifierSize.width) < scaledImages[scale].cols; c += step)
	 {
	    rects.push_back(cv::Rect(c, r, classifierSize.width, classifierSize.height));
	    imgs.push_back(scaledImages[scale](rects[rects.size() - 1]));
	    if (imgs.size() == classifier.BatchSize())
	    {
	       std::vector <std::vector<Prediction> >predictions = classifier.ClassifyBatch(imgs, 1);
	       for (size_t i = 0; i < predictions.size(); ++i)
	       {
		  for (std::vector <Prediction>::const_iterator it = predictions[i].begin(); it != predictions[i].end(); ++it)
		  {
		     if (it->first == "bin") 
		     {
			if (it->second > 0.9)
			{
#if 0
			   stringstream s;
			   std::string fn(argv[5]);

			   s << fn.substr(0, fn.find_last_of(".")) << "_";
			   s << std::setfill('0') << std::setw(4) << rects[i].x << "_";
			   s << std::setfill('0') << std::setw(4) << rects[i].y << "_";
			   s << std::setfill('0') << std::setw(4) << rects[i].width << "_";
			   s << std::setfill('0') << std::setw(4) << rects[i].height << ".png";
			   imwrite(s.str().c_str(), imgCopy(rects[i]));
#endif
			   cv::Rect dr(rects[i].x / scales[scale], rects[i].y / scales[scale],
			               rects[i].width / scales[scale], rects[i].height / scales[scale]);
			   detected.push_back(Detected(dr, it->second));
			}
			break;
		     }
		  }
	       }
	       if (rects.size())
	       {
		  while (rects.size() < classifier.BatchSize())
		  {
		     rects.push_back(cv::Rect(0, 0, classifierSize.width, classifierSize.height));
		     imgs.push_back(scaledImages[scale](rects[rects.size() - 1]));
		     std::vector <std::vector<Prediction> >predictions = classifier.ClassifyBatch(imgs, 1);
		     for (size_t i = 0; i < predictions.size(); ++i)
		     {
			for (std::vector <Prediction>::const_iterator it = predictions[i].begin(); it != predictions[i].end(); ++it)
			{
			   if (it->first == "bin") 
			   {
			      if (it->second > 0.9)
			      {
				 cv::Rect dr(rects[i].x / scales[scale], rects[i].y / scales[scale],
				             rects[i].width / scales[scale], rects[i].height / scales[scale]);
				 detected.push_back(Detected(dr, it->second));
			      }
			      break;
			   }
			}
		     }
		  }
	       }
	       imgs.clear();
	       rects.clear();
	    }
	 }
      }
   } 
   double end = gtod_wrapper();
   std::cout << "Elapsed time = " << (end - start) << std::endl;

#if 1
   namedWindow("Image", cv::WINDOW_AUTOSIZE);
   for (std::vector<Detected>::const_iterator it = detected.begin(); it != detected.end(); ++it)
   {
      std::cout << it->first << " " << it->second << std::endl;
      rectangle(img, it->first, cv::Scalar(0,0,255));
   }
   std::vector<cv::Rect> filteredRects;
   fastNMS(detected, 0.4f, filteredRects);
   for (std::vector<cv::Rect>::const_iterator it = filteredRects.begin(); it != filteredRects.end(); ++it)
   {
      std::cout << *it << std::endl;
      rectangle(img, *it, cv::Scalar(0,255,255));
   }
   imshow("Image", img);
   imwrite("detect.png", img);
   cv::waitKey(0);
#endif
   return 0;
}
