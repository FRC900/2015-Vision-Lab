
#ifndef INC__CAFFECLASSIFIER_H__
#define INC__CAFFECLASSIFIER_H__

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

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

  void setBatchSize(size_t batch_size);
  std::vector< std::vector<Prediction> > ClassifyBatch(const std::vector< cv::Mat > &imgs, int num_classes);
  size_t BatchSize(void) const;
 private:
  void SetMean(const std::string& mean_file);

  std::vector<float> Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

  void reshapeNet(void);
  std::vector<float> PredictBatch(const std::vector< cv::Mat > &imgs);
  void WrapBatchInputLayer(std::vector<std::vector<cv::Mat> > *input_batch);
  void PreprocessBatch(const std::vector<cv::Mat> &imgs,
                             std::vector< std::vector<cv::Mat> >* input_batch);
 private:
  std::shared_ptr<caffe::Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  size_t batch_size_;
  cv::Mat mean_;
  std::vector<std::string> labels_;
};
#endif
