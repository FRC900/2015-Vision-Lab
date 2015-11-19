#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/opencv.hpp>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>

using namespace std;
using namespace cv;
int frequency = 2;
int myblur= 2;
int lowedge= 100;
int highedge= 300;
int HueMax= 170;
int HueMin= 130;
int SatMax= 255;
int SatMin= 147;
int ValMax= 255;
int ValMin= 48;
int dilation_size = 2;
RNG rng(12345);
bool findRect(Mat &input, Rect &output, int HMin, int HMax, int SMin, int SMax, int VMin, int VMax)
{
        /* prec: input image &input, integers 0-255 for each min and max HSV value
        *  postc: a rectangle bounding the image we want
        *  takes the input image, filters to only find values in the range we want, finds
        *  the counters of the object and bounds it with a rectangle
        */
        Mat btrack;
        inRange(input, Scalar(HMin, SMin, VMin), Scalar(HMax, SMax, VMax), btrack);
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        vector<Point> points;
        findContours( btrack, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
        int screenContour = -1;
        for( size_t i = 0; i < hierarchy.size(); i++ )
        {
            if(hierarchy[i][3] >= 0 && boundingRect(contours[i]).area() > 1000)
            {
                points = contours[i];
                break;
            }
        }
        if(points.empty())
        {
            return false;
        }
        output = boundingRect(points);
        return true;
}
Rect adjustRect(Rect &input, float ratio)
{
    // adjusts the size of the rectangle to a fixed aspect ratio 
    Size rectsize = input.size();
    int width = rectsize.width;
    int height = rectsize.height;
    if (width / ratio > height)
    {
        height = width / ratio;
    }
    else if (width / ratio < height)
    {
        width = height * ratio;
    }
    Point tl = input.tl();
    tl.x = tl.x - (width - rectsize.width)/2;
    tl.y = tl.y - (height - rectsize.height)/2;
    Point br = input.br();
    br.x = br.x + (width - rectsize.width)/2;
    br.y = br.y + (height - rectsize.height)/2;
    return Rect(tl, br);
}
bool resizeRect(Rect &theRect, float pMax, const Mat imageCool)
{
    //takes the rect &theRect and randomly resizes it larger within range pMax, outputs the rect
    //to image imageCool
    Point tl = theRect.tl();
    Point br = theRect.br();
    if(tl.x < 0 || tl.y < 0 || br.x > imageCool.cols || br.y > imageCool.rows)
    {
        cout << "Rectangle out of bounds!";
        return false;
    }
    tl = Point(-1,-1);
    while (tl.x < 0 || tl.y < 0 || br.x > imageCool.cols || br.y > imageCool.rows)
    {
        // Sanity check to make sure we don't get stuck in an infinite loop
        // in case we can't adjust the rectangle within the bounds of the Mat.
        int intMax = int(pMax * 100 + .5);
        float adjust = rng.uniform(0,intMax);
        Size rectsize = theRect.size();
        float width = rectsize.width * (1 + adjust/100);
        int intwidth = int(width+.5);
        float height = rectsize.height * (1 + adjust/100);
        int intheight = int(height+.5);
        tl = theRect.tl();
        tl.x = tl.x - (width - rectsize.width)/2;
        tl.y = tl.y - (height - rectsize.height)/2;
        br = theRect.br();
        br.x = br.x + (width - rectsize.width)/2;
        br.y = br.y + (height - rectsize.height)/2;
    }
    theRect = Rect(tl, br);
    return true;
}
Mat rgbValThresh(int HMin,int HMax,int SMin,int SMax,int VMin, int VMax)
{
    //takes the HSV min/max values and returns as RGB middle value +- threshold
    Mat color = (Mat_<cv::Vec3b>(1,2) << Vec3b(HMin,SMin,VMin), Vec3b(HMax,SMax,VMax));
    cvtColor(color, color, COLOR_HSV2BGR);
    int valB = (color.at<cv::Vec3b>(0,0)[0]/2) + (color.at<cv::Vec3b>(0,1)[0])/2;
    int valG = (color.at<cv::Vec3b>(0,0)[1]/2) + (color.at<cv::Vec3b>(0,1)[1])/2;
    int valR = (color.at<cv::Vec3b>(0,0)[2]/2) + (color.at<cv::Vec3b>(0,1)[2])/2;
    int threshB = valB - color.at<cv::Vec3b>(0,0)[0];
    int threshG = valG - color.at<cv::Vec3b>(0,0)[1];
    int threshR = valR - color.at<cv::Vec3b>(0,0)[2];
    return (Mat_<cv::Vec3b>(1,2) << Vec3b(valB,valG,valR), Vec3b(threshB, threshG, threshR));
}
int main(int argc, char *argv[]) {
    if(argc < 2)
    {
        cout << "usage: " << argv[0] << " <filename1 filename2 ...>" << endl;
        return 0;
    }
    namedWindow("Original", WINDOW_AUTOSIZE);
    namedWindow("RangeControl", WINDOW_AUTOSIZE);
    namedWindow("Tracking", WINDOW_AUTOSIZE);

    createTrackbar( "Kernel size:\n 2n + 1", "RangeControl", &dilation_size, 21);

    createTrackbar("HueMax","RangeControl", &HueMax,255);
    createTrackbar("HueMin","RangeControl", &HueMin,255);

    createTrackbar("SatMax","RangeControl", &SatMax,255);
    createTrackbar("SatMin","RangeControl", &SatMin,255);

    createTrackbar("ValMax","RangeControl", &ValMax,255);
    createTrackbar("ValMin","RangeControl", &ValMin,255);


    createTrackbar("Blur","Parameters", &myblur,10);
    createTrackbar("LowEdge","Parameters", &lowedge,1000);
    createTrackbar("HighEdge","Parameters", &highedge,2000);

    createTrackbar("Frequency","Parameters", &frequency,10);
    String vidName = "";
    for(int i = 1; i < argc; i++)
    {
        vidName = argv[i];
        cout << vidName;
        VideoCapture inputVideo(vidName);

        if(!inputVideo.isOpened())
        {
            cout << "Capture not open; invalid video" << endl;
            continue;
        }

        Mat input;
        Mat hsvinput;

        vector<Mat> channels;
        vector<Mat> temp2(3);

        int count = 0;
        bool isColor = true;

        Mat temp;
        Mat hue;
        Mat sat;
        Mat val;
        Mat rgbVal;
        Mat ret;
        Mat frame;
        Mat gframe;
        Mat variancem;
        Mat tempm;
        float variance;
    
        rgbVal = rgbValThresh(HueMin, HueMax, SatMin, SatMax, ValMin, ValMax);
        while(1) {
            inputVideo >> frame;
            if(frame.empty())
            {
                break;
            }
            cvtColor( frame, gframe, CV_BGR2GRAY);
            Laplacian(gframe, temp, CV_8UC1);
            meanStdDev(temp, tempm, variancem);
            variance = pow(variancem.at<Scalar>(0,0)[0], 2);
            cout << variance;
            cvtColor( frame, hsvinput, CV_BGR2HSV);
            input = frame;
            Mat zero = Mat::zeros(input.rows, input.cols, CV_8UC1);
            imshow("HSV", hsvinput);
            Mat btrack;
            inRange(hsvinput, Scalar(HueMin, SatMin, ValMin), Scalar(HueMax, SatMax, ValMax), btrack);
            int dilation_type = MORPH_RECT;
            Mat element = getStructuringElement( dilation_type,
                                             Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                             Point( dilation_size, dilation_size ) );

            dilate(btrack, btrack, element);
            imshow("Tracking", btrack);

            vector<vector<Point> > contours;
            vector<Vec4i> hierarchy;

            /// Detect edges using canny
            Canny(btrack, temp, lowedge, highedge, 3);
            /// Find contours
            findContours( temp, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
            vector<Moments> mu(contours.size() );
            for( int i = 0; i < contours.size(); i++ )
            {
                mu[i] = moments( contours[i], false );
            }
            Rect boundRect;
            bool exists;
            exists = findRect(hsvinput, boundRect, HueMin, HueMax, SatMin, SatMax, ValMin, ValMax);
            if(exists == false)
            {
                cout << "No rectangle found!";
                continue;
            }
            boundRect = adjustRect(boundRect, 1.0);
            exists = resizeRect(boundRect, .2, hsvinput);
            if(exists == false)
            {
                cout << "Could not resize rectangle!";
                continue;
            }

            ///  Get the mass centers:
            vector<Point2f> mc( contours.size() );
            for( int i = 0; i < contours.size(); i++ )
            {
                mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
            }

            /// Draw contours
            Mat drawing = Mat::zeros( temp.size(), CV_8UC3 ); 
            Scalar color = Scalar(0,0,0);
            rectangle( input, boundRect, color ,2 );
            imshow("Edges", input);
            if(waitKey(1) >= 0) break;
        }
    }
    return 0;
}
