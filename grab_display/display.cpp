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
int HueMax= 153;
int HueMin= 113;
int SatMax= 255;
int SatMin= 199;
int ValMax= 255;
int ValMin= 190;
int dilation_size = 0;
RNG rng(12345);


int main() {
    namedWindow("Original", WINDOW_AUTOSIZE);
    //namedWindow("Parameters",WINDOW_AUTOSIZE);
    //namedWindow("Red", WINDOW_AUTOSIZE);
    //namedWindow("Green", WINDOW_AUTOSIZE);
    //namedWindow("Blue", WINDOW_AUTOSIZE);
    namedWindow("RangeControl", WINDOW_AUTOSIZE);
    namedWindow("Tracking", WINDOW_AUTOSIZE);

    createTrackbar( "Kernel size:\n 2n + 1", "RangeControl", &dilation_size, 21);

    createTrackbar("HueMax","RangeControl", &HueMax,255);
    createTrackbar("HueMin","RangeControl", &HueMin,255);

    createTrackbar("SatMax","RangeControl", &SatMax,255);
    createTrackbar("SatMin","RangeControl", &SatMin,255);

    createTrackbar("ValMax","RangeControl", &ValMax,255);
    createTrackbar("Valmin","RangeControl", &ValMin,255);


    createTrackbar("Blur","Parameters", &myblur,10);
    createTrackbar("LowEdge","Parameters", &lowedge,1000);
    createTrackbar("HighEdge","Parameters", &highedge,2000);

    createTrackbar("Frequency","Parameters", &frequency,10);
    VideoCapture inputVideo(0);

    if(!inputVideo.isOpened())
        cout << "Capture not open" << endl;

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

    //inputVideo >> input;

    while(1) {
        input = imread("/home/eblau/code/2015VisionCode/camera_cal/Secret Test Images/15 ft 00 deg.jpg",CV_LOAD_IMAGE_COLOR);
        cvtColor( input, hsvinput, CV_BGR2HSV);
        Mat zero = Mat::zeros(input.rows, input.cols, CV_8UC1);

        //inputVideo >> input;
        if(frequency == 0)
            frequency = 1;
        if(count % frequency == 0)
            isColor = !isColor;
        split(hsvinput, channels);


        /* if(!isColor)
        {
        	temp = channels[2];
        	channels[2] = channels[0];
        	channels[0] = temp;
        	merge(channels, input);
        } */



        imshow("HSV", hsvinput);
        /*
        temp2[0] = channels[0];
        temp2[1] = zero;
        temp2[2] = zero;
        merge(temp2, blue);
        //imshow("Blue", blue);

        temp2[0] = zero;
        temp2[1] = channels[1];
        merge(temp2, green);
        //imshow("Green", green);

        temp2[1] = zero;
        temp2[2] = channels[2];
        merge(temp2, red);
        //imshow("Red", red);
        */
        vector<Mat> comp(3);


        inRange(channels[0], HueMin, HueMax, comp[0]);
        inRange(channels[1], SatMin, SatMax, comp[1]);
        inRange(channels[2], ValMin, ValMax, comp[2]);

        Mat btrack;

        bitwise_and(comp[0], comp[1], btrack);
        bitwise_and(btrack, comp[2], btrack);

        int dilation_type = MORPH_RECT;


        Mat element = getStructuringElement( dilation_type,
                                             Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                             Point( dilation_size, dilation_size ) );


        dilate(btrack, btrack, element);
        imshow("Tracking", btrack);

        //BlueMax 73, BlueMin 0, RedMax 174, RedMin 127, GreenMax 75, GreenMin 0


        /* GaussianBlur(input, input, Size(9,9), myblur);
        imshow("Blur", input); */
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

        ///  Get the mass centers:
        vector<Point2f> mc( contours.size() );
        for( int i = 0; i < contours.size(); i++ )
        {
            mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
        }

        /// Draw contours
        Mat drawing = Mat::zeros( temp.size(), CV_8UC3 );
        for( int i = 0; i< contours.size(); i++ )
        {
            std::cout<<contours.size()<<std::endl;
            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            drawContours( input, contours, i, color, 2, 8, hierarchy, 0, Point() );
            circle( input, mc[i], 4, color, -1, 8, 0 );
        }

        //imshow("temp", temp);
        imshow("Edges", input);
        count++;
        if(waitKey(5) >= 0) break;
    }
    return 0;
}
