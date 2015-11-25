#include <iostream>
#include <opencv2/gpu/gpu.hpp>
#include "CaffeBatchPrediction.hpp"

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

   /* Load the network - this includes model geometry and trained weights */
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
   // This will be used to give each index of the output
   // a human-readable name
   std::ifstream labels(label_file.c_str());
   CHECK(labels) << "Unable to open labels file " << label_file;
   std::string line;
   while (std::getline(labels, line))
      labels_.push_back(string(line));

   Blob<float>* output_layer = net_->output_blobs()[0];
   CHECK_EQ(labels_.size(), output_layer->channels())
      << "Number of labels is different from the output layer dimension.";
}

// Helper function for compare - used to sort values by pair.first keys
static bool PairCompare(const std::pair<float, int>& lhs, 
			const std::pair<float, int>& rhs) 
{
   return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) 
{
   std::vector<std::pair<float, int> > pairs;
   for (size_t i = 0; i < v.size(); ++i)
      pairs.push_back(std::make_pair(v[i], i));
   std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

   std::vector<int> result;
   for (size_t i = 0; i < N; ++i)
      result.push_back(pairs[i].second);
   return result;
}

/* Return the top N predictions. */
std::vector<Prediction> CaffeClassifier::Classify(const cv::Mat& img, int N) 
{
   // Grab array of outputs ordered by label #
   std::vector<float> output = Predict(img);

   // This will return a vector of label #s, sorted by the 
   // prediction value of each label. That is, maxN[0] will
   // be the index of the label/output for the top ranked
   // precition.   
   N = std::min<int>(labels_.size(), N);
   std::vector<int> maxN = Argmax(output, N);
   std::vector<Prediction> predictions;

   // Create a predictions array combining the label and value
   // for the top N predictions for this image
   for (int i = 0; i < N; ++i) {
      int idx = maxN[i];
      predictions.push_back(std::make_pair(labels_[idx], output[idx]));
   }

   return predictions;
}

// Get the output values for a given image
// These values will be in the same order as the labels
// That is, [0] = value for label 0 and so on
std::vector<float> CaffeClassifier::Predict(const cv::Mat& img) 
{
   // Set input size, forward that through all the layers
   Blob<float>* input_layer = net_->input_blobs()[0];
   input_layer->Reshape(1, num_channels_,
	 input_geometry_.height, input_geometry_.width);
   net_->Reshape();

   // input_channels will be an array of Mat objects
   // Each Mat in the list is initialized to one of the 
   // input channels in the net.
   std::vector<cv::Mat> input_channels;
   WrapInputLayer(input_channels);

   // Preprocess img so it matches the expected
   // type, format and size of the net inputs
   // Copy that processed image into the 
   // input_channel Mats to actually load
   // the data into the net input buffers
   Preprocess(img, input_channels);

   // Run the forward pass with the
   // data just loaded above
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
void CaffeClassifier::WrapInputLayer(std::vector<cv::Mat>& input_channels) 
{
   Blob<float>* input_layer = net_->input_blobs()[0];

   int width = input_layer->width();
   int height = input_layer->height();
   float* input_data = input_layer->mutable_cpu_data();
   for (int i = 0; i < input_layer->channels(); ++i) 
   {
      // Creates a Mat with the correct header.  Sets the 
      // data to input_data instead of allocating it from the heap
      cv::Mat channel(height, width, CV_32FC1, input_data);
      input_channels->push_back(channel);
      input_data += width * height;
   }
}

// Take image in Mat, convert it to the correct image type,
// color depth, size to match the net input. Convert to 
// F32 type, since that's what the net inputs are. 
// Subtract out the mean before passing to the net input
// Then actually write the image to the net input memory buffers
void CaffeClassifier::Preprocess(const cv::Mat& img,
				 std::vector<cv::Mat>& input_channels) 
{
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

   // If needed, resize the input image to match
   // the net height and width
   cv::Mat sample_resized;
   if (sample.size() != input_geometry_)
      cv::resize(sample, sample_resized, input_geometry_);
   else
      sample_resized = sample;

   // Convert to 32-bit floats to match the net
   // input format
   cv::Mat sample_float;
   if (num_channels_ == 3)
      sample_resized.convertTo(sample_float, CV_32FC3);
   else
      sample_resized.convertTo(sample_float, CV_32FC1);

   // Subtract out the mean image from the input data
   cv::Mat sample_normalized;
   cv::subtract(sample_float, mean_, sample_normalized);

   /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the cv::Mat
    * objects in input_channels. */
   cv::split(sample_normalized, input_channels);

   CHECK(reinterpret_cast<float*>(input_channels.at(0).data)
	 == net_->input_blobs()[0]->cpu_data())
      << "Input channels are not wrapping the input layer of the network.";
}

// Given X input images, return X vectors of predictions.
// Each of the X vectors are themselves a vector which will have the 
// N predictions with the highest confidences for the corresponding
// input image
std::vector< std::vector<Prediction> > CaffeClassifier::ClassifyBatch(const std::vector< cv::Mat > &imgs, 
                                                                      size_t num_classes)
{
   // output_batch will be a flat vector of N floating point values 
   // per image (1 per N output labels), repeated
   // times the number of input images batched per run
   // Convert that into the output vector of vectors
   std::vector<float> output_batch = PredictBatch(imgs);
   std::vector< std::vector<Prediction> > predictions;
   size_t labels_size = labels_.size();
   num_classes = std::min(num_classes, labels_size);

   // For each image, find the top num_classes values
   for(size_t j = 0; j < imgs.size(); j++)
   {
      // Create an output vector just for values for this image. Since
      // each image has labels_size values, that's output_batch[j*labels_size]
      // through output_batch[(j+1) * labels_size]
      std::vector<float> output(output_batch.begin() + j*labels_size, output_batch.begin() + (j+1)*labels_size);
      // For the output specific to the jth image, grab the
      // indexes of the top num_classes predictions
      std::vector<int> maxN = Argmax(output, num_classes);
      // Using those top N indexes, create a set of labels/value predictions
      // specific to this jth image
      std::vector<Prediction> prediction_single;
      for (size_t i = 0; i < num_classes; ++i) 
      {
	 int idx = maxN[i];
	 prediction_single.push_back(std::make_pair(labels_[idx], output[idx]));
      }
      // Add the predictions for this image to the list of
      // predictions for all images
      predictions.push_back(std::vector<Prediction>(prediction_single));
   }
   return predictions;
}

/* Load the mean file in binaryproto format. */
void CaffeClassifier::SetMean(const std::string& mean_file) 
{
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
   for (int i = 0; i < num_channels_; ++i) 
   {
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

// TODO : see if we can do this once at startup or if
// it has to be done each pass.  If it can be done once,
// we can wrap the nets in Mat arrays in the constructor 
// and re-use them multiple times?
void CaffeClassifier::setBatchSize(size_t batch_size) 
{
   CHECK(batch_size >= 0);
   if (batch_size == batch_size_) return;
   batch_size_ = batch_size;
   reshapeNet();
}

// TODO : see if we can do this once at startup or if
// it has to be done each pass.  If it can be done once,
// we can wrap the nets in Mat arrays in the constructor 
// and re-use them multiple times?
void CaffeClassifier::reshapeNet() 
{
   CHECK(net_->input_blobs().size() == 1);
   caffe::Blob<float>* input_layer = net_->input_blobs()[0];
   input_layer->Reshape(batch_size_, num_channels_,
	 input_geometry_.height,
	 input_geometry_.width);
   net_->Reshape();
}

// Get the output values for a set of images in one flat vector
// These values will be in the same order as the labels for each
// image, and each set of labels for an image next adjacent to the
// one for the next image.
// That is, [0] = value for label 0 for the first image up to 
// [n] = value for label n for the first image. It then starts again
// for the next image - [n+1] = label 0 for image #2.
std::vector< float >  CaffeClassifier::PredictBatch(const std::vector< cv::Mat > &imgs) 
{
   Blob<float>* input_layer = net_->input_blobs()[0];

   input_layer->Reshape(batch_size_, num_channels_,
	 input_geometry_.height,
	 input_geometry_.width);

   /* Forward dimension change to all layers. */
   net_->Reshape();

   // The wrap code puts the buffer for one individual channel
   // input to the net (one color channel of one image) into 
   // a separate Mat 
   // The inner vector here will be one Mat per channel of the 
   // input to the net. The outer vector is a vector of those
   // one for each of the batched inputs.
   // This allows an easy copy from the input images
   // into the input buffers for the net
   std::vector< std::vector<cv::Mat> > input_batch;
   WrapBatchInputLayer(input_batch);

   // Process each image so they match the format
   // expected by the net, then copy the images
   // into the net's input buffers
   PreprocessBatch(imgs, input_batch);

   // Run a forward pass with the data filled in from above
   net_->ForwardPrefilled();

   /* Copy the output layer to a flat std::vector */
   Blob<float>* output_layer = net_->output_blobs()[0];
   const float* begin = output_layer->cpu_data();
   const float* end = begin + output_layer->channels()*imgs.size();
   return std::vector<float>(begin, end);
}


// Wrap input layer of the net into separate Mat objects
// This sets them up to be written with actual data
// in PreprocessBatch()
// TODO : see if this can be done once, or at the worst
// once every time the batch size changes instead of
// for every single batch we need to process
void CaffeClassifier::WrapBatchInputLayer(std::vector<std::vector<cv::Mat> > &input_batch)
{
   Blob<float>* input_layer = net_->input_blobs()[0];

   int width = input_layer->width();
   int height = input_layer->height();
   int num = input_layer->num();
   float* input_data = input_layer->mutable_cpu_data();
   for ( int j = 0; j < num; j++)
   {
      std::vector<cv::Mat> input_channels;
      for (int i = 0; i < input_layer->channels(); ++i)
      {
	 cv::Mat channel(height, width, CV_32FC1, input_data);
	 input_channels.push_back(channel);
	 input_data += width * height;
      }
      input_batch->push_back(std::vector<cv::Mat>(input_channels));
   }
}

// Take each image in Mat, convert it to the correct image type,
// color depth, size to match the net input. Convert to 
// F32 type, since that's what the net inputs are. 
// Subtract out the mean before passing to the net input
// Then actually write the images to the net input memory buffers
void CaffeClassifier::PreprocessBatch(const std::vector<cv::Mat> &imgs,
      std::vector< std::vector<cv::Mat> > &input_batch)
{
   for (int i = 0 ; i < imgs.size(); i++)
   {
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

#if 0
      // KCJ - add assert to make sure sample size and 
      // format are correctly set in calling function
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
#endif

      cv::Mat sample_normalized;
      cv::subtract(sample, mean_, sample_normalized);

      /* This operation will write the separate BGR planes directly to the
       * input layer of the network because it is wrapped by the cv::Mat
       * objects in input_channels. */
      cv::split(sample_normalized, *input_channels);

      //        CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
      //              == net_->input_blobs()[0]->cpu_data())
      //          << "Input channels are not wrapping the input layer of the network.";
   }
}

// Assorted helper functions
size_t CaffeClassifier::BatchSize(void) const
{
   return batch_size_;
}

cv::Size CaffeClassifier::getInputGeometry(void) const
{
   return input_geometry_;
}

const cv::Mat CaffeClassifier::getMean(void) const
{
   return mean_;
}
