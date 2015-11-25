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

void doBatchPrediction(const std::vector<cv::Mat> &imgs, const std::vector<cv::Rect> &rects, std::vector<Detected> &detected);

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

   // Simple multi-scale detect.  Take a single image, scale it into a number
   // of diffent sized images. Run a fixed-size detection window across each
   // of them.  Keep track of the scale of each scaled image to map the
   // detected rectangles back to the correct location and size on the
   // original input images

   // List of scaled images and the corresponding resize scale
   // used to create it. TODO : redo as a std::pair?
   std::vector<cv::Mat> scaledImages;
   std::vector<float> scales;

   // The detector can take a batch of input images
   // at a time. These arrays hold a set of sub-images
   // and the location on the full image they came from.
   // When enough are collected, run the whole batch through
   // the detectora in one batch
   std::vector<cv::Mat> imgs;    // input sub-images for this batch
   std::vector<cv::Rect> rects;  // where those sub-images came from
                                 // in the full input image

   std::vector<Detected> detected; // list of detected rects & confidence values

   // min and max size of object we're looking for.  Also, the 
   // fixed input size of the classifier itself.  The input
   // image will be scaled so that these min and max sizes
   // line up with the classifier input size.  Other scales will
   // fill in the range between those two end points.
   cv::Size minSize(100,100);
   cv::Size maxSize(700,700);
   const cv::Size classifierSize = classifier.getInputGeometry();

   // How many pixels to move the window for each step
   // TODO : figure out if it makes sense to change this depending on
   // the size of the scaled input image - i.e. it is possible that
   // a small step size on an image scaled way larger than the input
   // will end up detecting too much stuff ... each step on the larger
   // image might not correspond to a step of 1 pixel on the
   // input image?
   const int step = 6;
   //int step = std::min(img.cols, img.rows) *0.05;

   double start = gtod_wrapper(); // grab start time

   // The net expects each pixel to be 3x 32-bit floating point
   // values. Convert it once here rather than later for every
   // individual input image.
   cv::Mat f32Img;
   img.convertTo(f32Img, CV_32FC3);
   
   // Create array of scaled images
   scalefactor(f32Img, classifierSize, minSize, maxSize, 1.35, scaledImages, scales);

   // Main loop.  Look at each scaled image in turn
   for (size_t scale = 0; scale < scaledImages.size(); ++scale)
   {
      // Start at the upper left corner.  Loop through the rows and cols until
      // the detection window falls off the edges of the scaled image
      for (int r = 0; (r + classifierSize.height) < scaledImages[scale].rows; r +=step)
      {
	 for (int c = 0; (c + classifierSize.width) < scaledImages[scale].cols; c += step)
	 {
	    // Save location and image data for each sub-image
	    rects.push_back(cv::Rect(c, r, classifierSize.width, classifierSize.height));
	    imgs.push_back(scaledImages[scale](rects[rects.size() - 1]));
	    // If batch_size images are collected, run the detection code
	    if (imgs.size() == classifier.BatchSize())
	    {
	       doBatchPrediction(imgs, rects, detected);
	       // Reset image list
	       imgs.clear();
	       rects.clear();
	    }
	 }
      }
      // Finish up any left-over detection windows for this scaled image
      if (rects.size())
      {
	 doBatchPrediction(imgs, rects, detected);
	 imgs.clear();
	 rects.clear();
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

// do 1 run of the classifier. This takes up batch_size predictions and adds anything found
// to the detected list
void doBatchPrediction(const std::vector<cv::Mat> &imgs, const std::vector<cv::Rect> &rects, std::vector<Detected> &detected)
{
   std::vector <std::vector<Prediction> >predictions = classifier.ClassifyBatch(imgs, 1);
   // Each outer loop is the predictions for one input image
   for (size_t i = 0; i < rects.size(); ++i)
   {
      // Look for bins, > 90% confidence
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
	       // If found, create a detect rect scaled back to fit correctly
	       // on the original image, add it to the list of detections
	       cv::Rect dr(rects[i].x / scales[scale], rects[i].y / scales[scale],
		     rects[i].width / scales[scale], rects[i].height / scales[scale]);
	       detected.push_back(Detected(dr, it->second));
	    }
	    break;
	 }
      }
   }
}

