#ifndef matching2D_hpp
#define matching2D_hpp

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, const cv::Mat &img, bool createKeyPointsimgFlag=false);
void detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, const cv::Mat &img, bool createKeyPointsimgFlag=false);
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool createKeyPointsimgFlag=false);
void descKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, std::string descriptorType);
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorCategory, std::string matcherType, std::string selectorType);
// Store detection durations in an inline variable, det... functions utilize this namespace 
namespace PerformanceMetric{
	inline double detection_duration_{};
    inline double desriptor_extraction_duration_{};
    inline double match_keypoints_duration_{};

}
#endif /* matching2D_hpp */