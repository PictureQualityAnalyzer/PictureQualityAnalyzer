#include <opencv2/opencv.hpp>
#include <opencv2/xphoto.hpp>
#include <iomanip>
#include <iostream>

using namespace cv;
using namespace std;
using namespace xphoto;

int const CANNY_THRES = 50;
int const RGB_CHANNEL_SIZE = 256;
int const HUE_CHANNEL_SIZE = 180;

void countBlackPixels(Mat);
void detectBorder(Mat);
void detectFaces(Mat);
void createHistogram(Mat);
void fourierTransform(Mat);
void improveBalanceWhite(Mat);
void showHistogram(Mat&);
void dynamicRange(Mat);
void compare(Mat, Mat);

int main(int argc, char** argv) {

    cout << "Using OpenCV version " << CV_VERSION << endl << endl;

    Mat img_color, img_grayscale;

    if(argc > 1) {
        img_color = imread(argv[1], CV_LOAD_IMAGE_COLOR);
        if(!img_color.data) {
            cout << "Could not open or find the image." << endl ;
            return -1;
        }
    } else {
        cout << "Please enter the image you want to process." << endl;
        cout << "Usage: " << argv[0] << " [<image>]" << endl;
        return 0;
    }

    cvtColor(img_color, img_grayscale, COLOR_BGR2GRAY);

    namedWindow("original", 1);
    imshow("original", img_color);

    dynamicRange(img_color);

    detectFaces(img_color);

    detectBorder(img_color);

    createHistogram(img_color);

    showHistogram(img_color);

    fourierTransform(img_grayscale);

    improveBalanceWhite(img_color);

    cout << "Press any key to finish..." << endl << endl;
    waitKey(0);
    return 0;

}


void dynamicRange(Mat img_rgb) {

    double min, max;
    int rango;

    Mat img_hsv, img_grayscale;

    vector<Mat> rgbChannels(img_rgb.channels());

    //-- The image is supposed tu come in RGB format
    split(img_rgb, rgbChannels);

    cout << "* RGB" << endl;

    for(int i=0; i<3; i++) {
        minMaxLoc(rgbChannels[i], &min, &max);
        rango = max - min;

        cout << "  Min: " << min << ", Max: " << max << ", Dynamic range: " << rango << endl;
    }

    cvtColor(img_rgb, img_hsv, COLOR_BGR2HSV);

    split(img_hsv, rgbChannels);

    cout << "* HSV" << endl;

    for(int i=0; i<3; i++) {
        minMaxLoc(rgbChannels[i], &min, &max);
        rango = max - min;

        cout << "  Min: " << min << ", Max: " << max << ", Dynamic range: " << rango << endl;
    }

    cvtColor(img_rgb, img_grayscale, COLOR_BGR2GRAY);

    minMaxLoc(img_grayscale, &min, &max);
    rango = max - min;

    cout << "* Grayscale" << endl;
    cout << "  Min: " << min << ", Max: " << max << ", Dynamic range: " << rango << endl;

}

/**
* Improve white balance
*/
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


/**
* Compare and show the correlation between two images
*/
void compare(Mat src_base, Mat src_test1) {

    Mat hsv_base;
    Mat hsv_test1;

    /// Convert to HSV
    cvtColor( src_base, hsv_base, COLOR_BGR2HSV );
    cvtColor( src_test1, hsv_test1, COLOR_BGR2HSV );

    /// Using 50 bins for hue and 60 for saturation
    int h_bins = HUE_CHANNEL_SIZE;
    int s_bins = RGB_CHANNEL_SIZE;
    int v_bins = RGB_CHANNEL_SIZE;
    int histSize[] = { h_bins, s_bins, v_bins };

    // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = { 0, HUE_CHANNEL_SIZE };
    float s_ranges[] = { 0, RGB_CHANNEL_SIZE };
    float v_ranges[] = { 0, RGB_CHANNEL_SIZE };


    const float* ranges[] = { h_ranges, s_ranges, v_ranges };

    // Use all channels
    int channels[] = { 0, 1, 2 };

    /// Histograms
    Mat hist_base;
    Mat hist_test1;

    /// Calculate the histograms for the HSV images
    calcHist( &hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false );
    normalize( hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat() );

    calcHist( &hsv_test1, 1, channels, Mat(), hist_test1, 2, histSize, ranges, true, false );
    normalize( hist_test1, hist_test1, 0, 1, NORM_MINMAX, -1, Mat() );

    cout << "* Correlation between the original image and the improved version" << endl;

    /// Apply the histogram comparison methods
    for( int i = 0; i < 4; i++ ) {

        int compare_method = i;
        double base_base = compareHist( hist_base, hist_base, compare_method );
        double base_test1 = compareHist( hist_base, hist_test1, compare_method );

        printf( "  - Method [%d] Self, Improved: %f, %f \n", i, base_base, base_test1 );
    }

}


/**
* Calculate and show the fourier transform
*/
void fourierTransform(Mat src) {

    //-- It tends to be the fastest for image sizes that are multiple of the numbers two, three and five.
    //-- Therefore, to achieve maximal performance it is generally a good idea to pad border values to the image
    //-- to get a size with such traits.
    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( src.rows );
    int n = getOptimalDFTSize( src.cols ); // on the border add zero values
    copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

    dft(complexI, complexI);            // this way the result may fit in the source matrix

    //-- compute the magnitude (module) and switch to logarithmic scale
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];

    //-- switch to logarithmic scale
    magI += Scalar::all(1);
    log(magI, magI);

    //-- crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

    //-- rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;

    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
    // viewable image form (float between values 0 and 1).

    imshow("Spectrum Magnitude", magI);
}

/**
* Create a single window histogram for three channels
*/
void createHistogram(Mat src) {

    vector<Mat> bgr_planes;
    split(src, bgr_planes);

    //-- Set the ranges for B,G,R
    float range[] = { 0, RGB_CHANNEL_SIZE };
    const float* histRange = { range };

    //-- We want our bins to have the same size (uniform) and to clear the histograms in the beginning
    bool uniform = true, accumulate = false;

    //-- Set some parameters for the histograms for R, G and B
    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/RGB_CHANNEL_SIZE );

    //-- create the Mat objects to save our histograms
    Mat hist[3];

    //-- Compute the histograms (we'll get vectors of 256 elements) and normalize them to [ 0, histImage.rows ]
    for(int i = 0; i < 3; i++) {
        calcHist(&bgr_planes[i], 1, 0, Mat(), hist[i], 1, &RGB_CHANNEL_SIZE, &histRange, uniform, accumulate);
        normalize(hist[i], hist[i], 0, hist_h, NORM_MINMAX, -1, Mat());
    }

    //-- Generate the empty matrix with the needed size for the histogram
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0,0,0));

    //-- Draw the three histograms using lines. Each line goes from the previous value to the current.
    for( int i = 1; i < RGB_CHANNEL_SIZE; i++ ) {
        line(histImage,
                Point(
                        bin_w * (i-1),
                        hist_h - cvRound( hist[0].at<float>(i-1) )
                ),
                Point(
                        bin_w * (i),
                        hist_h - cvRound( hist[0].at<float>(i) )
                ),
                Scalar(255, 0, 0),
                2, 8, 0
        );
        line(histImage,
                Point(
                        bin_w * (i-1),
                        hist_h - cvRound( hist[1].at<float>(i-1) )
                ),
                Point(
                        bin_w * (i),
                        hist_h - cvRound( hist[1].at<float>(i) )
                ),
                Scalar(0, 255, 0),
                2, 8, 0
        );
        line(histImage,
                Point(
                        bin_w * (i-1),
                        hist_h - cvRound( hist[2].at<float>(i-1) )
                ),
                Point(
                        bin_w * (i),
                        hist_h - cvRound(hist[2].at<float>(i))
                ),
                Scalar(0, 0, 255),
                2, 8, 0
        );
    }

    namedWindow("Combined Histogram", WINDOW_AUTOSIZE );
    imshow("Combined Histogram", histImage );
}


/**
* Count the black pixels in a image
*/
void countBlackPixels(Mat img) {

    double totalNumberOfPixels = img.rows * img.cols;
    double percentNonZeroPixels = (countNonZero(img)/totalNumberOfPixels) * 100;
    cout << setprecision (3) << "* Non-black pixels (border pixels): " << percentNonZeroPixels << "%" << endl;

}

/**
* Detect borders of an image
*/
void detectBorder(Mat img) {

    cvtColor(img, img, COLOR_BGR2GRAY);

    //-- Reduce noise with a kernel 3x3
    blur(img, img, Size(3,3));

    //-- Canny detector
    Canny(img, img, CANNY_THRES, 3 * CANNY_THRES, 3);

    countBlackPixels(img);

    namedWindow("Filtered image");
    imshow("Filtered image", img);

}

/**
* Detect faces and smiles
*/
void detectFaces(Mat frame) {

    string face_cascade_name = "./data/haarcascade_frontalface_alt.xml";
    string eyes_cascade_name = "./data/haarcascade_smile.xml";

    CascadeClassifier face_cascade, eyes_cascade;

    string window_name = "Capture - Face detection";

    vector<Rect> faces;
    Mat frame_gray;

    int smilesCount = 0;

    //-- Load the cascades
    if(!face_cascade.load(face_cascade_name)) {
        cout << "-- Error loading face cascade" << endl;
        //return -1;
    }
    if(!eyes_cascade.load(eyes_cascade_name)) {
        cout << "-- Error loading smile cascade" << endl;
        //return -1;
    }

    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    //-- Detect faces
    face_cascade.detectMultiScale(
            frame_gray,
            faces,
            1.1, 2, 0|CASCADE_SCALE_IMAGE,
            Size(30, 30)
    );

    // TODO Rotated faces are not detected, maybe try with several small rotations

    cout << "* Faces detected: " << faces.size() << endl;

    for( size_t i = 0; i < faces.size(); i++ ) {

        //-- Draw one ellipse for each face
        Point center(
                faces[i].x + faces[i].width / 2,
                faces[i].y + faces[i].height / 2
        );
        ellipse(
                frame,
                center,
                Size(faces[i].width / 2, faces[i].height / 2),
                0, 0, 360,
                Scalar( 255, 0, 255 ),
                4, 8, 0
        );

        //-- Get a ROI for each face
        // TODO Search only the bottom part of the face to avoid false positives
        Mat faceROI = frame_gray( faces[i] );
        vector<Rect> smiles;

        //-- In each face, detect if there is a smile
        // TODO Adjust Size parameter in order to be proportional to face size and avoid false positives
        eyes_cascade.detectMultiScale(
                faceROI,
                smiles,
                1.1, 2, 0|CASCADE_SCALE_IMAGE,
                Size(20, 20)
        );

        // TODO Only a smile per face is possible, adjust this
        for( size_t j = 0; j < smiles.size(); j++ ) {

            Point eye_center(
                    faces[i].x + smiles[j].x + smiles[j].width / 2,
                    faces[i].y + smiles[j].y + smiles[j].height / 2
            );
            int radius = cvRound((smiles[j].width + smiles[j].height) * 0.25);
            circle(
                    frame,
                    eye_center,
                    radius,
                    Scalar(255, 0, 0),
                    4, 8, 0
            );
            smilesCount++;
        }
    }

    cout << "* Smiles detected: " << smilesCount << endl;

    //-- Show what you got
    imshow(window_name, frame);
}

/**
* Calculate the histogram manually and using three diferent windows
*/
void showHistogram(Mat& img)
{
    int bins = 256;             // number of bins
    int nc = img.channels();    // number of channels
    vector<Mat> hist(nc);       // array for storing the histograms
    vector<Mat> canvas(nc);     // images for displaying the histogram
    int hmax[3] = {0,0,0};      // peak value for each histogram

    //-- Initialize the hist arrays
    for (int i = 0; i < hist.size(); i++)
        hist[i] = Mat::zeros(1, bins, CV_32SC1);

    // Calculate the histogram of the image
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            for (int k = 0; k < nc; k++)
            {
                uchar val = nc == 1 ? img.at<uchar>(i,j) : img.at<Vec3b>(i,j)[k];
                hist[k].at<int>(val) += 1;
            }
        }
    }

    // For each histogram arrays, obtain the maximum (peak) value
    // Needed to normalize the display later
    for (int i = 0; i < nc; i++)
    {
        for (int j = 0; j < bins-1; j++)
            hmax[i] = hist[i].at<int>(j) > hmax[i] ? hist[i].at<int>(j) : hmax[i];
    }

    const char* wname[3] = { "blue", "green", "red" };
    Scalar colors[3] = { Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,255) };

    // Display each histogram in a canvas
    for (int i = 0; i < nc; i++)
    {
        canvas[i] = Mat::ones(125, bins, CV_8UC3);

        for (int j = 0, rows = canvas[i].rows; j < bins-1; j++)
        {
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