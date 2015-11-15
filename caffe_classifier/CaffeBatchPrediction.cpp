#include <iostream>
#include <opencv2/gpu/gpu.hpp>
#include "caffeclassifier.h"
#include "fast_nms.hpp"

using namespace caffe;

CaffeClassifier::CaffeClassifier(const std::string& model_file,
      const std::string& trained_file,
      const std::string& mean_file,
      const std::string& label_file,
      const bool use_GPU,
      const int batch_size) {

   if (use_GPU)
      Caffe::set_mode(Caffe::GPU);
   else
      Caffe::set_mode(Caffe::CPU);

   /* Set batchsize */
   batch_size_ = batch_size;

   /* Load the network. */
   net_.reset(new Net<float>(model_file, TEST));
   net_->CopyTrainedLayersFrom(trained_file);

   CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
   CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

   Blob<float>* input_layer = net_->input_blobs()[0];
   num_channels_ = input_layer->channels();
   CHECK(num_channels_ == 3 || num_channels_ == 1)
      << "Input layer should have 1 or 3 channels.";
   input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

   /* Load the binaryproto mean file. */
   SetMean(mean_file);

   /* Load labels. */
   std::ifstream labels(label_file.c_str());
   CHECK(labels) << "Unable to open labels file " << label_file;
   std::string line;
   while (std::getline(labels, line))
      labels_.push_back(string(line));

   Blob<float>* output_layer = net_->output_blobs()[0];
   CHECK_EQ(labels_.size(), output_layer->channels())
      << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
      const std::pair<float, int>& rhs) {
   return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
   std::vector<std::pair<float, int> > pairs;
   for (size_t i = 0; i < v.size(); ++i)
      pairs.push_back(std::make_pair(v[i], i));
   std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

   std::vector<int> result;
   for (int i = 0; i < N; ++i)
      result.push_back(pairs[i].second);
   return result;
}

/* Return the top N predictions. */
std::vector<Prediction> CaffeClassifier::Classify(const cv::Mat& img, int N) {
   std::vector<float> output = Predict(img);

   N = std::min<int>(labels_.size(), N);
   std::vector<int> maxN = Argmax(output, N);
   std::vector<Prediction> predictions;
   for (int i = 0; i < N; ++i) {
      int idx = maxN[i];
      predictions.push_back(std::make_pair(labels_[idx], output[idx]));
   }

   return predictions;
}

std::vector<float> CaffeClassifier::Predict(const cv::Mat& img) {
   Blob<float>* input_layer = net_->input_blobs()[0];
   input_layer->Reshape(1, num_channels_,
	 input_geometry_.height, input_geometry_.width);
   /* Forward dimension change to all layers. */
   net_->Reshape();

   std::vector<cv::Mat> input_channels;
   WrapInputLayer(&input_channels);

   Preprocess(img, &input_channels);

   net_->ForwardPrefilled();

   /* Copy the output layer to a std::vector */
   Blob<float>* output_layer = net_->output_blobs()[0];
   const float* begin = output_layer->cpu_data();
   const float* end = begin + output_layer->channels();
   return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void CaffeClassifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
   Blob<float>* input_layer = net_->input_blobs()[0];

   int width = input_layer->width();
   int height = input_layer->height();
   float* input_data = input_layer->mutable_cpu_data();
   for (int i = 0; i < input_layer->channels(); ++i) {
      cv::Mat channel(height, width, CV_32FC1, input_data);
      input_channels->push_back(channel);
      input_data += width * height;
   }
}

void CaffeClassifier::Preprocess(const cv::Mat& img,
      std::vector<cv::Mat>* input_channels) {
   /* Convert the input image to the input image format of the network. */
   cv::Mat sample;
   if (img.channels() == 3 && num_channels_ == 1)
      cv::cvtColor(img, sample, CV_BGR2GRAY);
   else if (img.channels() == 4 && num_channels_ == 1)
      cv::cvtColor(img, sample, CV_BGRA2GRAY);
   else if (img.channels() == 4 && num_channels_ == 3)
      cv::cvtColor(img, sample, CV_BGRA2BGR);
   else if (img.channels() == 1 && num_channels_ == 3)
      cv::cvtColor(img, sample, CV_GRAY2BGR);
   else
      sample = img;

   cv::Mat sample_resized;
   if (sample.size() != input_geometry_)
      cv::resize(sample, sample_resized, input_geometry_);
   else
      sample_resized = sample;

   cv::Mat sample_float;
   if (num_channels_ == 3)
      sample_resized.convertTo(sample_float, CV_32FC3);
   else
      sample_resized.convertTo(sample_float, CV_32FC1);

   cv::Mat sample_normalized;
   cv::subtract(sample_float, mean_, sample_normalized);

   /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the cv::Mat
    * objects in input_channels. */
   cv::split(sample_normalized, *input_channels);

   CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
	 == net_->input_blobs()[0]->cpu_data())
      << "Input channels are not wrapping the input layer of the network.";
}

std::vector< std::vector<Prediction> > CaffeClassifier::ClassifyBatch(const std::vector< cv::Mat > &imgs, int num_classes){
   std::vector<float> output_batch = PredictBatch(imgs);
   std::vector< std::vector<Prediction> > predictions;
   for(size_t j = 0; j < imgs.size(); j++){
      std::vector<float> output(output_batch.begin() + j*num_classes, output_batch.begin() + (j+1)*num_classes);
      std::vector<int> maxN = Argmax(output, num_classes);
      std::vector<Prediction> prediction_single;
      for (int i = 0; i < num_classes; ++i) {
	 int idx = maxN[i];
	 prediction_single.push_back(std::make_pair(labels_[idx], output[idx]));
      }
      predictions.push_back(std::vector<Prediction>(prediction_single));
   }
   return predictions;
}

/* Load the mean file in binaryproto format. */
void CaffeClassifier::SetMean(const std::string& mean_file) {
   BlobProto blob_proto;
   ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

   /* Convert from BlobProto to Blob<float> */
   Blob<float> mean_blob;
   mean_blob.FromProto(blob_proto);
   CHECK_EQ(mean_blob.channels(), num_channels_)
      << "Number of channels of mean file doesn't match input layer.";

   /* The format of the mean file is planar 32-bit float BGR or grayscale. */
   std::vector<cv::Mat> channels;
   float* data = mean_blob.mutable_cpu_data();
   for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
      channels.push_back(channel);
      data += mean_blob.height() * mean_blob.width();
   }

   /* Merge the separate channels into a single image. */
   cv::Mat mean;
   cv::merge(channels, mean);

   /* Compute the global mean pixel value and create a mean image
    * filled with this value. */
   cv::Scalar channel_mean = cv::mean(mean);
   mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

void CaffeClassifier::setBatchSize(size_t batch_size) {
   CHECK(batch_size != 0);
   if (batch_size == batch_size_) return;
   batch_size_ = batch_size;
   reshapeNet();
}

void CaffeClassifier::reshapeNet() {
   CHECK(net_->input_blobs().size() == 1);
   caffe::Blob<float>* input_layer = net_->input_blobs()[0];
   input_layer->Reshape(batch_size_, num_channels_,
	 input_geometry_.height,
	 input_geometry_.width);
   net_->Reshape();
}

std::vector< float >  CaffeClassifier::PredictBatch(const std::vector< cv::Mat > &imgs) {
   Blob<float>* input_layer = net_->input_blobs()[0];

   input_layer->Reshape(batch_size_, num_channels_,
	 input_geometry_.height,
	 input_geometry_.width);

   /* Forward dimension change to all layers. */
   net_->Reshape();

   std::vector< std::vector<cv::Mat> > input_batch;
   WrapBatchInputLayer(&input_batch);

   PreprocessBatch(imgs, &input_batch);

   net_->ForwardPrefilled();

   /* Copy the output layer to a std::vector */
   Blob<float>* output_layer = net_->output_blobs()[0];
   const float* begin = output_layer->cpu_data();
   const float* end = begin + output_layer->channels()*imgs.size();
   return std::vector<float>(begin, end);
}


void CaffeClassifier::WrapBatchInputLayer(std::vector<std::vector<cv::Mat> > *input_batch){
   Blob<float>* input_layer = net_->input_blobs()[0];

   int width = input_layer->width();
   int height = input_layer->height();
   int num = input_layer->num();
   float* input_data = input_layer->mutable_cpu_data();
   for ( int j = 0; j < num; j++){
      std::vector<cv::Mat> input_channels;
      for (int i = 0; i < input_layer->channels(); ++i){
	 cv::Mat channel(height, width, CV_32FC1, input_data);
	 input_channels.push_back(channel);
	 input_data += width * height;
      }
      input_batch -> push_back(std::vector<cv::Mat>(input_channels));
   }
   //cv::imshow("bla", input_batch->at(1).at(0));
   //cv::waitKey(1);
}


void CaffeClassifier::PreprocessBatch(const std::vector<cv::Mat> &imgs,
      std::vector< std::vector<cv::Mat> >* input_batch){
   for (int i = 0 ; i < imgs.size(); i++){
      cv::Mat img = imgs[i];
      std::vector<cv::Mat> *input_channels = &(input_batch->at(i));

      /* Convert the input image to the input image format of the network. */
      cv::Mat sample;
      if (img.channels() == 3 && num_channels_ == 1)
	 cv::cvtColor(img, sample, CV_BGR2GRAY);
      else if (img.channels() == 4 && num_channels_ == 1)
	 cv::cvtColor(img, sample, CV_BGRA2GRAY);
      else if (img.channels() == 4 && num_channels_ == 3)
	 cv::cvtColor(img, sample, CV_BGRA2BGR);
      else if (img.channels() == 1 && num_channels_ == 3)
	 cv::cvtColor(img, sample, CV_GRAY2BGR);
      else
	 sample = img;

      cv::Mat sample_resized;
      if (sample.size() != input_geometry_)
	 cv::resize(sample, sample_resized, input_geometry_);
      else
	 sample_resized = sample;

      cv::Mat sample_float;
      if (num_channels_ == 3)
	 sample_resized.convertTo(sample_float, CV_32FC3);
      else
	 sample_resized.convertTo(sample_float, CV_32FC1);

      cv::Mat sample_normalized;
      cv::subtract(sample_float, mean_, sample_normalized);

      /* This operation will write the separate BGR planes directly to the
       * input layer of the network because it is wrapped by the cv::Mat
       * objects in input_channels. */
      cv::split(sample_normalized, *input_channels);

      //        CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
      //              == net_->input_blobs()[0]->cpu_data())
      //          << "Input channels are not wrapping the input layer of the network.";
   }
}

size_t CaffeClassifier::BatchSize(void) const
{
   return batch_size_;
}

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
   CaffeClassifier classifier(model_file, trained_file, mean_file, label_file, true, 20 );

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
   std::vector <std::vector<Prediction> >predictions = classifier.ClassifyBatch(imgs, 6);
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
	       std::cerr << ".";
	       for (size_t i = 0; i < predictions.size(); ++i)
	       {
		  for (std::vector <Prediction>::const_iterator it = predictions[i].begin(); it != predictions[i].end(); ++it)
		  {
		     if (it->first == "bin") 
		     {
			if (it->second > 0.9)
			{
			   //stringstream s;
			   //s << "falsepositive" << posCount++ << ".png";
			   //imwrite(s.str().c_str(), imgCopy(rects[i]));
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
      size /= 1.15;
   } while (size > 48);
   double end = gtod_wrapper();
   std::cout << "Elapsed time = " << (end - start) << std::endl;
#endif

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
   cv::waitKey(0);
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

