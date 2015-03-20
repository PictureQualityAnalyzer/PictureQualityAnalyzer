#include <opencv2/opencv.hpp>
#include <opencv2/xphoto.hpp>
#include <iomanip>

using namespace cv;
using namespace std;
using namespace xphoto;

int const max_BINARY_value = 255;
int const CANNY_THRES = 50;

void countBlackPixels(Mat img) {

  double totalNumberOfPixels = img.rows * img.cols;
  double nonZeroPixels = countNonZero(img);
  double percentNonZeroPixels = (nonZeroPixels/totalNumberOfPixels) * 100;
  cout << "Total: " << nonZeroPixels << " de " << totalNumberOfPixels << endl;
  cout << setprecision (3) << "Porcentaje de pixeles no negros: " << percentNonZeroPixels << "%" << endl;

}

/**
 * <A short one line description>
 *
 * <Longer description>
 * <May span multiple lines or paragraphs as needed>
 *
 * @param  Description of method's or function's input parameter
 * @param  ...
 * @return Description of the return value
 */
void detectBorder(Mat img) {
  Mat img_filtered;
  cvtColor( img, img, COLOR_BGR2GRAY );

  /// Reduce noise with a kernel 3x3
  blur( img, img, Size(3,3) );
  /// Canny detector
  Canny( img, img, CANNY_THRES, 3 * CANNY_THRES, 3 );

  countBlackPixels(img);

  namedWindow("Filtered image");
  imshow("Filtered image", img);

}


void applyFilter(Mat img) {
  cvtColor( img, img, COLOR_BGR2GRAY );

  // Filter kernel for detecting vertical edges
  float vertical_fk[5][5] = {{0,0,0,0,0}, {0,0,0,0,0}, {-1,-2,6,-2,-1}, {0,0,0,0,0}, {0,0,0,0,0}}; // Filter kernel for detecting horizontal edges
  float horizontal_fk[5][5] = {{0,0,-1,0,0}, {0,0,-2,0,0}, {0,0,6,0,0}, {0,0,-2,0,0}, {0,0,-1,0,0}};
  Mat filter_kernel = Mat(5, 5, CV_32FC1, horizontal_fk);

  // Apply filter
  filter2D(img, img, -1, filter_kernel);
}

void showHistogram(Mat& img, string name) {

  // TODO Generalizar el histograma para diferente número de canales y diferente esquema de color.

  unsigned short bins = 256;             // number of bins
  unsigned short bins_hue = 180;         // number of bins for hue channel (if any)
  unsigned short nc = img.channels();    // number of channels

  vector<Mat> hist(nc);       // histogram arrays

  // Initalize histogram arrays
  for (int i = 0; i < hist.size(); i++) {
    hist[i] = Mat::zeros(1, bins, CV_8UC1);
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


void symmetry(Mat img) {

  // TODO quizá la simetría habiendo rotado antes la imagen? O daría igual?

  cvtColor( img, img, COLOR_BGR2HSV );

  Mat half_down, half_up, half_right, half_left;
  Mat hist_down, hist_up, hist_left, hist_right;

  half_up = img(Range(0, img.rows/2 - 1), Range(0, img.cols - 1));
  half_down = img(Range(img.rows/2, img.rows - 1), Range(0, img.cols - 1));

  half_left = img(Range(0, img.rows - 1), Range(0, img.cols/2 - 1));
  half_right = img(Range(0, img.rows - 1), Range(img.cols/2, img.cols - 1));

  // hue varies from 0 to 179, saturation from 0 to 255
  float h_ranges[] = { 0, 180 };
  float s_ranges[] = { 0, 256 };
  int h_bins = 50;
  int s_bins = 60;
  int histSize[] = { h_bins, s_bins };
  const float* ranges[] = { h_ranges, s_ranges };
  int channels[] = { 0, 1 };

  calcHist( &half_down, 1, channels, Mat(), hist_down, 2, histSize, ranges, true, false );
  calcHist( &half_up, 1, channels, Mat(), hist_up, 2, histSize, ranges, true, false );

  calcHist( &half_left, 1, channels, Mat(), hist_left, 2, histSize, ranges, true, false );
  calcHist( &half_right, 1, channels, Mat(), hist_right, 2, histSize, ranges, true, false );

  double compUpDown = compareHist( hist_up, hist_down, 0 );
  cout << "Mitad superior contra inferior: " << compUpDown << endl;

  double compLeftRight = compareHist( hist_left, hist_right, 0 );
  cout << "Mitad derecha contra izquierda: " << compLeftRight << endl;

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

  //showHistogram(img, "original");
  //showHistogram(dst, "improved balance");

}

void dynamicRange(Mat img) {

  vector<Mat> rgbChannels(img.channels());

  split(img, rgbChannels);

  for(int i=0; i<3; i++) {
    double min, max;
    minMaxLoc(rgbChannels[i], &min, &max);
    int rango = max - min;

    cout << "Min: " << min << ", Max: " << max << ", Rango dinámico: " << rango << endl;
  }

  cvtColor(img, img, COLOR_BGR2HSV);

  split(img, rgbChannels);

  for(int i=0; i<3; i++) {
    double min, max;
    minMaxLoc(rgbChannels[i], &min, &max);
    int rango = max - min;

    cout << "Min: " << min << ", Max: " << max << ", Rango dinámico: " << rango << endl;
  }

}

void detectFaces(Mat img) {

  Mat cara;

  // Load Face cascade (.xml file)
  CascadeClassifier face_cascade, smile_cascade;
  if(!face_cascade.load( "./haarcascade_frontalface_alt2.xml" )) {
    cout << "Error loading cascade for face" << endl;
    exit(0);
  }

  // Detect faces
  vector<Rect> faces;
  face_cascade.detectMultiScale( img, faces, 1.05, 3, 0|CASCADE_SCALE_IMAGE, Size(20, 20) );

  cout << endl << "Faces detected: " << faces.size() << endl << endl;

  if(faces.size() > 0) {

    if(!smile_cascade.load( "./haarcascade_smile.xml" )) {
      cout << "Error loading cascade for smile" << endl;
      exit(0);
    }

    vector<Rect> smiles;
    int sonrisas = 0;

    for(int j=0; j<faces.size(); j++) {
      cara = img(faces[j]);
      smile_cascade.detectMultiScale( cara, smiles, 1.05, 2, 0|CASCADE_SCALE_IMAGE, Size(5, 5) );

      if(smiles.size() > 0) {

        int x = smiles[0].x;
        int y = smiles[0].y;
        int h = y+smiles[0].height;
        int w = x+smiles[0].width;
        rectangle(cara,
                Point (x,y),
                Point (w,h),
                Scalar(255,0,255));

        imshow("cara"+to_string(j), cara);
        sonrisas++;
      }
    }
    cout << endl << "Smiles detected: " << sonrisas << endl << endl;

  } else {
    cout << "Al no haber caras, no hay sonrisas" << endl;
  }

}

int main(int argc, char** argv) {

  Mat img_color, img_grayscale;
  if(argc > 1) {
    img_color = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  } else {
    cout << "Please enter the image you want to process." << endl;
    return 0;
  }

  cvtColor(img_color, img_grayscale, COLOR_BGR2GRAY);


  //detectBorder(img_color);

  dynamicRange(img_color);

  detectFaces(img_color);

  improveBalanceWhite(img_color);

  //symmetry(img_color);

  namedWindow("original", 1);
  imshow("original", img_color);

  waitKey(0);
  return 0;

}
