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

int main(int argc, char *argv[]) {

  string path = "hoge.png";
  Mat beforeImg, beforeGray;
  Mat afterImg, afterGray;
  beforeImg = imread("before.png");
  afterImg = imread("after.png");
  cvtColor(beforeImg, beforeGray, COLOR_RGB2GRAY);
  cvtColor(afterImg, afterGray, COLOR_RGB2GRAY);

  std::vector<cv::Point2f> p0, p1;
  goodFeaturesToTrack(beforeGray, p0, 5, 0.01, 10, Mat(), 3, 3, 0, 0.04);

  imshow("feature", CreateFeaturePointImage(beforeImg, p0));
  waitKey(0);

  std::vector<uchar> status;
  std::vector<float> err;

  calcOpticalFlowPyrLK(beforeGray, afterGray, p0, p1, status, err);
  imshow("feature", CreateFeaturePointImage(afterImg, p1));

  waitKey(0);

  Ptr<DenseOpticalFlowExt> opticalFlow = superres::createOptFlow_DualTVL1();
  Mat flowX, flowY;
  opticalFlow->calc(beforeGray, afterGray, flowX, flowY);
	imshow("flowX", flowX);
	imshow("flowY", flowY);
	waitKey(0);
}
