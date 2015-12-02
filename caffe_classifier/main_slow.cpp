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
   cv::Mat imgCopy(img.clone());
   typedef std::pair<cv::Rect, float> Detected;
   std::vector<Detected> detected;

   int size = 700;
   int step = 6;
   //int step = std::min(img.cols, img.rows) *0.05;
   double start = gtod_wrapper();

#if 0
   size = 150;
   rects.push_back(cv::Rect(72, 12, size, size));
   imgs.push_back(img(rects[rects.size()-1]).clone());
   rects.push_back(cv::Rect(72, 18, size,size));
   imgs.push_back(img(rects[rects.size()-1]).clone());
   size = 130;
   rects.push_back(cv::Rect(78, 18, size,size));
   imgs.push_back(img(rects[rects.size()-1]).clone());
   std::vector <std::vector<Prediction> >predictions = classifier.ClassifyBatch(imgs, 1);
   for (size_t i = 0; i < predictions.size(); ++i) 
   {
      for (size_t j = 0; j < predictions[i].size(); ++j) 
      {
	 Prediction p = predictions[i][j];
	 std::cout << "i = " << i << " j = " << j << " " << rects[i] << " first = " << p.first << " second = " << p.second << std::endl;
      }
   }
#else
   int posCount = 0;
   do
   {
      for (int c = 0; (c + size) < img.cols; c += step)
      {
	 for (int r = 0; (r + size) < img.rows; r +=step)
	 {
	    rects.push_back(cv::Rect(c, r, size, size));
	    imgs.push_back(imgCopy(rects[rects.size() - 1]));
	    if (imgs.size() == classifier.BatchSize())
	    {
	       std::vector <std::vector<Prediction> >predictions = classifier.ClassifyBatch(imgs, 6);
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
			   detected.push_back(Detected(rects[i], it->second));
			}
			break;
		     }
		  }
	       }
	       imgs.clear();
	       rects.clear();
	    }
	 }
      }
      size /= 1.25;
   } while (size > 48);
   double end = gtod_wrapper();
   std::cout << "Elapsed time = " << (end - start) << std::endl;
#endif

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
#if 0
   std::cout << "---------- Prediction for " << file << " ----------" << std::endl;

   std::vector<Prediction> predictions = classifier.Classify(img, 2);

   std::cout <<  predictions.size() <<  std::endl;

   /* Print the top N predictions. */
   for (size_t i = 0; i < predictions.size(); ++i) {
      Prediction p = predictions[i];
      std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
	 << p.first << "\"" << std::endl;
   }
#endif
}
