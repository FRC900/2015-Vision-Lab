#ifndef INC__CAFFEBATCHPREDICTION_HPP_
#define INC__CAFFEBATCHPREDICTION_HPP_

#include <memory>
#include <string>
#include <vector>
#include <utility>

#include <caffe/caffe.hpp>
#include <opencv2/highgui/highgui.hpp>

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<std::string, float> Prediction;

class CaffeClassifier {
 public:
  CaffeClassifier(const std::string& model_file,
             const std::string& trained_file,
             const std::string& mean_file,
             const std::string& label_file,
             const bool use_GPU,
             const int batch_size);

  // Given an input image, return the N predictions with the highest
  // confidences
  std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

  // Given X input images, return X vectors of predictions.
  // Each of the X vectors are themselves a vector which will have the 
  // N predictions with the highest confidences for the corresponding
  // input image
  std::vector< std::vector<Prediction> > ClassifyBatch(const std::vector< cv::Mat > &imgs, size_t num_classes);

  // Get the width and height of an input image to the net
  cv::Size getInputGeometry(void) const;

  // Change the batch size of the model
  size_t BatchSize(void) const;
  void setBatchSize(size_t batch_size);

  // Get an image with the mean value of all of the training images
  const cv::Mat getMean(void) const;
 private:
  void SetMean(const std::string& mean_file);

  // Get the output values for a given image
  // These values will be in the same order as the labels
  // That is, [0] = value for label 0 and so on
  std::vector<float> Predict(const cv::Mat& img);

  // Wrap input layer of the net into separate Mat objects
  // This sets them up to be written with actual data
  // in Preprocess()
  void WrapInputLayer(std::vector<cv::Mat>& input_channels);

  // Take image in Mat, convert it to the correct image type,
  // color depth, size to match the net input. Convert to 
  // F32 type, since that's what the net inputs are. 
  // Subtract out the mean before passing to the net input
  // Then actually write the image to the net input memory buffers
  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>& input_channels);

  void reshapeNet(void);

  // Get the output values for a set of images
  // These values will be in the same order as the labels for each
  // image, and each set of labels for an image next adjacent to the
  // one for the next image.
  // That is, [0] = value for label 0 for the first image up to 
  // [n] = value for label n for the first image. It then starts again
  // for the next image - [n+1] = label 0 for image #2.
  std::vector<float> PredictBatch(const std::vector< cv::Mat > &imgs);

  // Wrap input layer of the net into separate Mat objects
  // This sets them up to be written with actual data
  // in PreprocessBatch()
  void WrapBatchInputLayer(std::vector<std::vector<cv::Mat> > &input_batch);

  // Take each image in Mat, convert it to the correct image type,
  // color depth, size to match the net input. Convert to 
  // F32 type, since that's what the net inputs are. 
  // Subtract out the mean before passing to the net input
  // Then actually write the images to the net input memory buffers
  void PreprocessBatch(const std::vector<cv::Mat> &imgs,
                             std::vector< std::vector<cv::Mat> > &input_batch);

 private:
  std::shared_ptr<caffe::Net<float> > net_; // the net itself
  cv::Size input_geometry_;         // size of one input image
  int num_channels_;                // num color channels per input image
  size_t batch_size_;               // number of images to process in one go
  cv::Mat mean_;                    // mean value of input images
  std::vector<std::string> labels_; // labels for each output value
};
#endif
