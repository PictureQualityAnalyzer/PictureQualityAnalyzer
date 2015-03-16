#include <opencv2/opencv.hpp>
#include <opencv2/xphoto.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;
using namespace xphoto;

void showHistogram(Mat& img, string name) {

  unsigned short bins = 256;             // number of bins
  unsigned short nc = img.channels();    // number of channels

  vector<Mat> hist(nc);       // histogram arrays

  // Initalize histogram arrays
  for (int i = 0; i < hist.size(); i++) {
    hist[i] = Mat::zeros(1, bins, CV_32SC1);
  }

  // Calculate the histogram of the image
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      for (int k = 0; k < nc; k++) {
        uchar val = nc == 1 ? img.at<uchar>(i,j) : img.at<Vec3b>(i,j)[k];
        hist[k].at<int>(val) += 1;
      }
    }
  }

  // For each histogram arrays, obtain the maximum (peak) value
  // Needed to normalize the display later
  int hmax[3] = {0,0,0};
  for (int i = 0; i < nc; i++) {
    for (int j = 0; j < bins-1; j++) {
      hmax[i] = hist[i].at<int>(j) > hmax[i] ? hist[i].at<int>(j) : hmax[i];
    }
  }

  const string wname[3] = { name + " - BLUE", name + " - GREEN", name + " - RED" };
  Scalar colors[3] = { Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,255) };

  vector<Mat> canvas(nc);

  // Display each histogram in a canvas
  for (int i = 0; i < nc; i++) {

    canvas[i] = Mat::ones(125, bins, CV_8UC3);

    for (int j = 0, rows = canvas[i].rows; j < bins-1; j++) {

      line(
              canvas[i],
              Point(j, rows),
              Point(j, rows - (hist[i].at<int>(j) * rows/hmax[i])),
              nc == 1 ? Scalar(200,200,200) : colors[i],
              1, 8, 0
      );
    }

    imshow(nc == 1 ? "value" : wname[i], canvas[i]);
  }
}



void compare(Mat src_base, Mat src_test1) {

  Mat hsv_base;
  Mat hsv_test1;
  Mat hsv_half_down;

  /// Convert to HSV
  cvtColor( src_base, hsv_base, COLOR_BGR2HSV );
  cvtColor( src_test1, hsv_test1, COLOR_BGR2HSV );

  hsv_half_down = hsv_base( Range( hsv_base.rows/2, hsv_base.rows - 1 ), Range( 0, hsv_base.cols - 1 ) );

  /// Using 50 bins for hue and 60 for saturation
  int h_bins = 50;
  int s_bins = 60;
  int histSize[] = { h_bins, s_bins };

  // hue varies from 0 to 179, saturation from 0 to 255
  float h_ranges[] = { 0, 180 };
  float s_ranges[] = { 0, 256 };

  const float* ranges[] = { h_ranges, s_ranges };

  // Use the o-th and 1-st channels
  int channels[] = { 0, 1 };

  /// Histograms
  Mat hist_base;
  Mat hist_half_down;
  Mat hist_test1;

  /// Calculate the histograms for the HSV images
  calcHist( &hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false );
  normalize( hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat() );

  calcHist( &hsv_half_down, 1, channels, Mat(), hist_half_down, 2, histSize, ranges, true, false );
  normalize( hist_half_down, hist_half_down, 0, 1, NORM_MINMAX, -1, Mat() );

  calcHist( &hsv_test1, 1, channels, Mat(), hist_test1, 2, histSize, ranges, true, false );
  normalize( hist_test1, hist_test1, 0, 1, NORM_MINMAX, -1, Mat() );

  /// Apply the histogram comparison methods
  for( int i = 0; i < 4; i++ ) {

    int compare_method = i;
    double base_base = compareHist( hist_base, hist_base, compare_method );
    double base_half = compareHist( hist_base, hist_half_down, compare_method );
    double base_test1 = compareHist( hist_base, hist_test1, compare_method );

    printf( "Method [%d] Base, Base-Half, Base-Test : %f, %f, %f \n", i, base_base, base_half , base_test1 );
  }

}


void improveBalanceWhite(Mat img) {

  Mat dst;

  // This function is included in cv::xphoto namespace
  balanceWhite(img, dst, WHITE_BALANCE_SIMPLE);

  namedWindow("after white balance", 1);
  imshow("after white balance", dst);

  compare(img, dst);

  showHistogram(img, "original");
  showHistogram(dst, "improved balance");

  cout << "Press ESC to quit..." << endl;
  waitKey(0);

}

void dynamicRange(Mat img) {

  double min, max;
  minMaxLoc(img, &min, &max);
  int rango = max - min;

  cout << "Min: " << min << ", Max: " << max << ", Rango dinámico: " << rango << endl;

}

void detectFaces(Mat img) {

  // Load Face cascade (.xml file)
  CascadeClassifier face_cascade, smile_cascade;
  if(!face_cascade.load( "./haarcascade_frontalface_alt2.xml" )) {
    cout << "Error loading cascade for face" << endl;
    exit(0);
  }

  if(!smile_cascade.load( "./haarcascade_smile.xml" )) {
    cout << "Error loading cascade for smile" << endl;
    exit(0);
  }

  // Detect faces
  vector<Rect> faces;
  face_cascade.detectMultiScale( img, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

  cout << endl << "Faces detected: " << faces.size() << endl << endl;


  // Detect smiles
  // TODO Hay que detectar las sonrisas DENTRO de cada cara
  vector<Rect> smiles;
  smile_cascade.detectMultiScale( img, smiles, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

  cout << endl << "Smiles detected: " << smiles.size() << endl << endl;
}

int main(int argc, char** argv) {

  Mat img1;
  if(argc > 1) {
    img1 = imread(argv[1], 1);
  } else {
    cout << "What image do you want to process?" << endl;
    return 0;
  }

  namedWindow("original", 1);
  imshow("original", img1);

  dynamicRange(img1);

  detectFaces(img1);

  improveBalanceWhite(img1);
}
