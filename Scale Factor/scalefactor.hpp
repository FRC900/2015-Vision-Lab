#ifndef INC_SCALEFACTOR_HPP__
#define INC_SCALEFACTOR_HPP__

#include <opencv2/core/core.hpp>
#include <vector>

void scalefactor(cv::Mat inputimage, cv::Size objectsize, cv::Size minsize, cv::Size maxsize, float scaleFactor, std::vector<cv::Mat> &images, std::vector<float> &scales);

#endif

