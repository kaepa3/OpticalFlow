#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/superres/optical_flow.hpp>
#include <string>

using namespace cv;
using namespace cv::superres;
using namespace std;

Mat CreateFeaturePointImage(Mat img, vector<Point2f> points) {
  for (Point2f &p : points) {
    circle(img, p, 10, Scalar_(0, 0, 255), 5);
  }
  return img;
}

void DisplayMat(Mat src) {
  Size size = src.size();
  for (int r = 0; r < size.height; r++) {
    for (int c = 0; c < size.width; c++) {
      cout << r << ":" << c << "->" << +src.at<uchar>(c, r) << endl;
    }
  }
}

int main(int argc, char *argv[]) {

  Mat beforeImg, beforeGray;
  Mat afterImg, afterGray;
  beforeImg = imread("before.png");
  afterImg = imread("after.png");

  beforeImg = imread("sample.jpeg");
  Size imgSize = beforeImg.size();
	printf("%d", imgSize.width);
  Mat matrix =
      getRotationMatrix2D(Point2f(imgSize.width / 2, imgSize.height / 2), 5.0, 1.0);
  warpAffine(beforeImg, afterImg, matrix, Size(imgSize.width, imgSize.height));

  cvtColor(beforeImg, beforeGray, COLOR_RGB2GRAY);
  cvtColor(afterImg, afterGray, COLOR_RGB2GRAY);

  std::vector<cv::Point2f> p0, p1;
  goodFeaturesToTrack(beforeGray, p0, 5, 0.01, 10, Mat(), 3, 3, 0, 0.04);

  //  imshow("feature", CreateFeaturePointImage(beforeImg, p0));
  //  waitKey(0);

  std::vector<uchar> status;
  std::vector<float> err;

  calcOpticalFlowPyrLK(beforeGray, afterGray, p0, p1, status, err);
  //  imshow("feature", CreateFeaturePointImage(afterImg, p1));
  //  waitKey(0);
  Mat flow;
  calcOpticalFlowFarneback(beforeGray, afterGray, flow, 0.8, 10, 15, 3, 5, 1.1,
                           0);
  Mat map(flow.size(), CV_32FC2);
  for (int y = 0; y < map.rows; ++y) {
    for (int x = 0; x < map.cols; ++x) {
      Point2f f = flow.at<Point2f>(y, x);
      map.at<Point2f>(y, x) = Point2f(x + f.x, y + f.y);
    }
  }

  Mat newFrame;
  remap(afterGray, newFrame, map, Mat(), INTER_LINEAR);
  imshow("newFrame", newFrame);
  imshow("after", afterGray);
  imshow("before", beforeGray);
  waitKey(0);

  Ptr<DenseOpticalFlowExt> opticalFlow = superres::createOptFlow_DualTVL1();
  Mat flowX, flowY;
  opticalFlow->calc(beforeGray, afterGray, flowX );


  for (int y = 0; y < map.rows; ++y) {
    for (int x = 0; x < map.cols; ++x) {
      Point2f f = flowX.at<Point2f>(y, x);
      map.at<Point2f>(y, x) = Point2f(x + f.x, y + f.y);
    }
  }

  remap(afterGray, newFrame, map, Mat(), INTER_LINEAR);
  imshow("newFrame2", newFrame);
  imshow("after2", afterGray);
  imshow("before2", beforeGray);
  waitKey(0);

  cout << "flowX" << flowX.size << endl;
//  DisplayMat(flowX);
  cout << "flowY:" << flowY.size << endl;
//  DisplayMat(flowY);
  imshow("flowX", flowX);
  imshow("flowY", flowY);
  imwrite("flowX.bmp", flowX);
  imwrite("flowY.bmp", flowY);

  waitKey(0);
}
