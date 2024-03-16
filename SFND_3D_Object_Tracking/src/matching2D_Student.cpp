#include <numeric>
#include "matching2D.hpp"
#include <map>
#include <functional>
#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <fstream>
using namespace std;

void detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, const cv::Mat &img, bool createKeyPointsimgFlag)
{
    // Parameters for Shi-Tomasi corner detection
    int blockSize = 4;       
    double maxOverlap = 0.0; 
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = static_cast<int>(img.rows * img.cols / std::max(1.0, minDistance));

    double qualityLevel = 0.01; 
    double k = 0.04;

    // Apply corner detection
    double t = static_cast<double>(cv::getTickCount());
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // Add corners to result vector
    for (const auto &corner : corners)
    {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = corner;
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
    double t_ms = 1000 * t / 1.0;
    PerformanceMetric::detection_duration_ = t_ms;
    std::cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << t_ms << " ms" << std::endl;

    // Visualize results
    if (createKeyPointsimgFlag)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        std::string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint>& keypoints, const cv::Mat& img, bool createKeyPointsimgFlag)
{
    // Detector parameters
    int blockSize = 2;
    int apertureSize = 3;
    double maxOverlap = 0.0; // Max. permissible overlap between two features in %
    int minResponse = 100;
    double k = 0.04;

    // Apply corner detection
    double t = static_cast<double>(cv::getTickCount());

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // Locate local maxima in the Harris response matrix
    for (int j = 0; j < dst_norm.rows; j++)
    {
        for (int i = 0; i < dst_norm.cols; i++)
        {
            int response = static_cast<int>(dst_norm.at<float>(j, i));
            if (response > minResponse)
            {
                cv::KeyPoint newKeypoint;
                newKeypoint.pt = cv::Point2f(i, j);
                newKeypoint.size = 2 * apertureSize;
                newKeypoint.response = response;

                if (true)
                {
                    // Perform non-maximum suppression (NMS) in a local neighborhood around each maximum
                    bool isOverlapped = false;
                    for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                    {
                        double overlap = cv::KeyPoint::overlap(newKeypoint, *it);
                        if (overlap > maxOverlap)
                        {
                            isOverlapped = true;
                            if (newKeypoint.response > (*it).response)
                            {
                                *it = newKeypoint; // Replace the keypoint with a higher response one
                                break;
                            }
                        }
                    }

                    // Add the new keypoint which isn't considered to have overlap with the keypoints already stored in the list
                    if (!isOverlapped)
                    {
                        keypoints.push_back(newKeypoint);
                    }
                }
                else
                {
                    keypoints.push_back(newKeypoint);
                }
            }
        }
    }
    t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
    double t_ms = 1000 * t / 1.0;
    PerformanceMetric::detection_duration_ = t_ms;
    std::cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << t_ms << " ms" << std::endl;

    // Visualize results
    if (createKeyPointsimgFlag)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        std::string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in image using the modern detectors
void detKeypointsModern(vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool createKeyPointsimgFlag)
{
    cv::Ptr<cv::FeatureDetector> detector;

    if (detectorType.compare("FAST") == 0)
    {
        int threshold = 30; // difference between intensity of the central pixel and pixels of a circle around this pixel
        bool bNMS = true;
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;

        detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
    }
    else if (detectorType.compare("BRISK") == 0)
    {
        int threshold = 30;
        int octaves = 3;
        float patterScale = 1.0f;

        detector = cv::BRISK::create(threshold, octaves, patterScale);
    }
    else if (detectorType.compare("ORB") == 0)
    {
        int nfeatures = 500;
        float scaleFactor = 1.2f;
        int nlevels = 8;
        int edgeThreshold = 31;
        int firstLevel = 0;
        int WTA_K = 2;
        cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
        int patchSize = 31;
        int fastThreshold = 20;

        detector = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
    }
    else if (detectorType.compare("AKAZE") == 0)
    {
        cv::AKAZE::DescriptorType descriptorType = cv::AKAZE::DESCRIPTOR_MLDB;
        int descriptorSize = 0;
        int descriptorChannels = 3;
        float threshold = 0.001f;
        int nOctaves = 4;
        int nOctaveLayers = 4;
        cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2;

        detector = cv::AKAZE::create(descriptorType, descriptorSize, descriptorChannels, threshold, nOctaves, nOctaveLayers, diffusivity);
    }
    else if (detectorType.compare("SIFT") == 0)
    {
        int nfeatures = 0;
        int nOctaveLayers = 3;
        double contrastThreshold = 0.04;
        double edgeThreshold = 10.0;
        double sigma = 1.6;

        detector = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    }

    double t = static_cast<double>(cv::getTickCount());
    detector->detect(img, keypoints);
    t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
    double t_ms = 1000 * t / 1.0;
    PerformanceMetric::detection_duration_ = t_ms;
    cout << detectorType << " detector with n=" << keypoints.size() << " keypoints in " << t_ms << " ms" << endl;
    string windowName = detectorType + " Detector Results";
    // visualize results
    if (createKeyPointsimgFlag)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void descKeypoints(std::vector<cv::KeyPoint>& keypoints, cv::Mat& img, cv::Mat& descriptors, std::string descriptorType) {
    // Map to store descriptor creation functions
   
    std::map<std::string, std::function<cv::Ptr<cv::DescriptorExtractor>()>> descriptorCreators = {
        {"BRISK", []() {
            int threshold = 30;
            int octaves = 3;
            float patternScale = 1.0f;
            return cv::BRISK::create(threshold, octaves, patternScale);
        }},
        {"BRIEF", []() {
            int bytes = 32;
            bool bOrientation = false;
            return cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, bOrientation);
        }},
        {"ORB", []() {
            int nfeatures = 500;
            float scaleFactor = 1.2f;
            int nlevels = 8;
            int edgeThreshold = 31;
            int firstLevel = 0;
            int WTA_K = 2;
            cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
            int patchSize = 31;
            int fastThreshold = 20;
            return cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
        }},
        {"FREAK", []() {
            bool orientationNormalized = true;
            bool scaleNormalized = true;
            float patternScale = 22.0f;
            int nOctaves = 4;
            return cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, patternScale, nOctaves);
        }},
        {"AKAZE", []() {
            cv::AKAZE::DescriptorType descriptorType = cv::AKAZE::DESCRIPTOR_MLDB;
            int descriptorSize = 0;
            int descriptorChannels = 3;
            float threshold = 0.001f;
            int nOctaves = 4;
            int nOctaveLayers = 4;
            cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2;
            return cv::AKAZE::create(descriptorType, descriptorSize, descriptorChannels, threshold, nOctaves, nOctaveLayers, diffusivity);
        }},
        {"SIFT", []() {
          int nfeatures = 500; // Detect up to 500 keypoints
		  int nOctaveLayers = 4; // Number of layers in each octave
		  double contrastThreshold = 0.02; // Lower contrast threshold to detect more keypoints
		  double edgeThreshold = 20.0; // Increase edge threshold to filter out weaker edges
		  double sigma = 1.2; // Reduce the blurring effect by lowering the sigma value
		  return cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
		}}    
    };

    // Select appropriate descriptor
    auto it = descriptorCreators.find(descriptorType);
    if (it != descriptorCreators.end()) {
       // it retrieves the corresponding function from the map using it->second().
       // This function call returns a descriptor extractor object (e.g., BRISK, BRIEF, etc.).
        cv::Ptr<cv::DescriptorExtractor> extractor = it->second();

        // Perform feature description
        //  measures the time taken for descriptor computation using OpenCV's getTickCount() function.
        double t = static_cast<double>(cv::getTickCount());
        extractor->compute(img, keypoints, descriptors);
        t = (static_cast<double>(cv::getTickCount() - t)) / cv::getTickFrequency();
        std::cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << std::endl;
        PerformanceMetric::desriptor_extraction_duration_ = 1000 * t / 1.0;
    } 
   else {
        std::cerr << "#3 : EXTRACT DESCRIPTORS failed. Wrong descriptorType - " << descriptorType << ". Use one of the following descriptors: BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT" << std::endl;
        exit(-1);
    }
}

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorCategory, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    double t;
    if (matcherType == "MAT_BF")
    {
        int normType = descriptorCategory.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;;
        matcher = cv::BFMatcher::create(normType, crossCheck);
        cout << "MAT_BF matching (" << descriptorCategory << ") with cross-check=" << crossCheck;
    }
    else if (matcherType == "MAT_FLANN")
    {
        if (descSource.type() != CV_32F)
        {
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        std::cout << "MAT_FLANN matching";
    }
    else
    {
       std::cerr << "#4 : MATCH KEYPOINT DESCRIPTORS failed. Wrong matcherType - " << matcherType << ". Use one of the following matchers: MAT_BF, MAT_FLANN" << std::endl;
        exit(-1);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
        t = static_cast<double>(cv::getTickCount());
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
        cout << " (NN) with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else if (selectorType == "SEL_KNN")
    { // k nearest neighbors (k=2)
        vector<vector<cv::DMatch>> knnMatches;
        t = static_cast<double>(cv::getTickCount());
        matcher->knnMatch(descSource, descRef, knnMatches, 2); // Finds the best match for each descriptor
        t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
        cout << " (KNN) with n=" << knnMatches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;

        // filter matches using descriptor distance ratio test
        const double ratioThreshold = 0.8;
        for (const auto &knnMatch : knnMatches)
        {
            if (knnMatch[0].distance < ratioThreshold * knnMatch[1].distance)
            {
                matches.push_back(knnMatch[0]);
            }
        }
        cout << "Distance ratio test removed " << knnMatches.size() - matches.size() << " keypoints"<< endl;
    }
    else
    {
        std::cerr << "\n#4 : MATCH KEYPOINT DESCRIPTORS failed. Wrong selectorType - " << selectorType << ". Use one of the following selector: SEL_NN, SEL_KNN" << endl;
        exit(-1);
    }
    PerformanceMetric::match_keypoints_duration_ = 1000 * t / 1.0;
    
}
