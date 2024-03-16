#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // Loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // Assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // Project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // Pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // Pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // Shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // Check whether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // End of loop over all bounding boxes

        // Check whether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        {
            // Add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // End of loop over all Lidar points
}

/*
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size.
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // Create top view image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // Create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // Plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0;
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // World coordinates
            float xw = (*it2).x; // World position in m with x facing forward from sensor
            float yw = (*it2).y; // World position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // Top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // Find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // Draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // Draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // Augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);
    }

    // Plot distance markers
    float lineSpacing = 2.0; // Gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // Display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 4);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // Wait for key to be pressed
    }
}

/*
* Filters keypoints and matches within the given bounding box based on the Euclidean distance between matched keypoints.
* @param boundingBox: The bounding box within which keypoints should be clustered.
* @param previousKeypoints: Keypoints from the previous frame.
* @param currentKeypoints: Keypoints from the current frame.
* @param keypointMatches: Matches between keypoints in consecutive frames.
*/
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &previousKeypoints, std::vector<cv::KeyPoint> &currentKeypoints, std::vector<cv::DMatch> &keypointMatches) {
    const double maxDistanceThreshold = 140.0;

    // Precompute distances between each pair of matched keypoints
    std::vector<double> distances(keypointMatches.size());
    std::transform(keypointMatches.begin(), keypointMatches.end(), distances.begin(), [&](const cv::DMatch &match) {
        const cv::Point2f &currentPoint = currentKeypoints[match.trainIdx].pt;
        const cv::Point2f &previousPoint = previousKeypoints[match.queryIdx].pt;
        // Calculate and return the Euclidean distance between current and previous keypoints
        return cv::norm(currentPoint - previousPoint);
    });

    // Calculate the mean distance of all matches to identify outliers
    double meanDistance = std::accumulate(distances.begin(), distances.end(), 0.0) / distances.size();

    // Filter matches based on distance criteria and if the current keypoint is within the bounding box
    std::vector<cv::DMatch> filteredMatches;
    std::vector<cv::KeyPoint> filteredKeypoints;
    for (size_t i = 0; i < keypointMatches.size(); ++i) {
        const cv::DMatch &match = keypointMatches[i];
        const cv::KeyPoint &currentKeypoint = currentKeypoints[match.trainIdx];

        // Check if distance is within acceptable range and keypoint is within the bounding box
        if (distances[i] - meanDistance < maxDistanceThreshold && boundingBox.roi.contains(currentKeypoint.pt)) {
            filteredMatches.push_back(match);
            filteredKeypoints.push_back(currentKeypoint);
        }
    }

    // Update the bounding box with filtered matches and keypoints
    boundingBox.kptMatches = std::move(filteredMatches);
    boundingBox.keypoints = std::move(filteredKeypoints);
}

/*
* Computes the time-to-collision (TTC) for the camera based on keypoint matches between consecutive frames.
* @param previousKeypoints: Keypoints from the previous frame.
* @param currentKeypoints: Keypoints from the current frame.
* @param keypointMatches: KeyPoint matches between the two frames.
* @param frameRate: The frame rate of the sequence.
* @param timeToCollision: The computed time-to-collision (output parameter).
* @param visualizationImg: Optional parameter for visualizing the result.
*/
void computeTTCCamera(std::vector<cv::KeyPoint> &previousKeypoints, std::vector<cv::KeyPoint> &currentKeypoints, 
                      std::vector<cv::DMatch> keypointMatches, double frameRate, double &timeToCollision, cv::Mat *visualizationImg) {
    std::vector<double> distanceRatios; // Stores the ratio of distances between keypoints in consecutive frames

    // Iterate through all matches to compute distance ratios
    for (auto it1 = keypointMatches.begin(); it1 != keypointMatches.end() - 1; ++it1) {
        const cv::KeyPoint &outerCurrentKeypoint = currentKeypoints[it1->trainIdx];
        const cv::KeyPoint &outerPreviousKeypoint = previousKeypoints[it1->queryIdx];

        for (auto it2 = it1 + 1; it2 != keypointMatches.end(); ++it2) {
            const double minimumDistance = 100.0; // Minimum required distance to consider

            const cv::KeyPoint &innerCurrentKeypoint = currentKeypoints[it2->trainIdx];
            const cv::KeyPoint &innerPreviousKeypoint = previousKeypoints[it2->queryIdx];

            // Compute distances between keypoints in current and previous frames
            double currentDistance = cv::norm(outerCurrentKeypoint.pt - innerCurrentKeypoint.pt);
            double previousDistance = cv::norm(outerPreviousKeypoint.pt - innerPreviousKeypoint.pt);

            // Avoid division by zero and ensure current distance is above a threshold
            if (previousDistance > std::numeric_limits<double>::epsilon() && currentDistance >= minimumDistance) {
                double distanceRatio = currentDistance / previousDistance;
                distanceRatios.push_back(distanceRatio);
            }
        }
    }

    // Calculate TTC if there are valid distance ratios
    if (distanceRatios.empty()) {
        timeToCollision = NAN; 
        return;
    }

    // Sort distance ratios to find the median
    std::sort(distanceRatios.begin(), distanceRatios.end());
    size_t medianIndex = distanceRatios.size() / 2;
    double medianDistanceRatio = distanceRatios.size() % 2 == 0 ?
                                 (distanceRatios[medianIndex - 1] + distanceRatios[medianIndex]) / 2.0 :
                                 distanceRatios[medianIndex];

    double deltaTime = 1.0 / frameRate;
    timeToCollision = -deltaTime / (1 - medianDistanceRatio);
}

/*
* Computes the time-to-collision (TTC) for the Lidar based on consecutive Lidar measurements.
* @param previousLidarPoints: Lidar points from the previous frame.
* @param currentLidarPoints: Lidar points from the current frame.
* @param frameRate: The frame rate of the sequence.
* @param timeToCollision: The computed time-to-collision (output parameter).
*/
void computeTTCLidar(std::vector<LidarPoint> &previousLidarPoints,
                     std::vector<LidarPoint> &currentLidarPoints, double frameRate, double &timeToCollision)
{
    // Time between two measurements in seconds
    double timeDelta = 1 / frameRate;

    // Vectors to hold x-coordinates of Lidar points from previous and current frames
    std::vector<double> previousFrameXCoordinates, currentFrameXCoordinates;

    // Extract x-coordinates from Lidar points
    for (const auto& point : previousLidarPoints) {
        previousFrameXCoordinates.push_back(point.x);
    }
    for (const auto& point : currentLidarPoints) {
        currentFrameXCoordinates.push_back(point.x);
    }

    // Helper function to calculate median distance using nth_element for efficiency
    auto calculateMedianDistance = [](std::vector<double>& xCoordinates) -> double {
        size_t n = xCoordinates.size() / 2;
        std::nth_element(xCoordinates.begin(), xCoordinates.begin() + n, xCoordinates.end());
        double median = xCoordinates[n];
        if (xCoordinates.size() % 2 == 0) {
            std::nth_element(xCoordinates.begin(), xCoordinates.begin() + n - 1, xCoordinates.end());
            median = (median + xCoordinates[n - 1]) / 2.0;
        }
        return median;
    };

    const double medianDistancePrevious = calculateMedianDistance(previousFrameXCoordinates);
    const double medianDistanceCurrent = calculateMedianDistance(currentFrameXCoordinates);

    if (medianDistancePrevious - medianDistanceCurrent > 0) { 
        timeToCollision = medianDistanceCurrent * timeDelta / (medianDistancePrevious - medianDistanceCurrent);
    } else {
        timeToCollision = std::numeric_limits<double>::quiet_NaN();
    }
}

/*
* Matches bounding boxes between consecutive frames using the best matches of keypoints.
* @param matches: Matches between keypoints in consecutive frames.
* @param bbBestMatches: Map to store the best matching bounding box indices (output parameter).
* @param prevFrame: The previous frame data.
* @param currFrame: The current frame data.
*/
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // Initialize a 2D array to store counts of matches between bounding boxes
    std::vector<std::vector<int>> counts(prevFrame.boundingBoxes.size(), std::vector<int>(currFrame.boundingBoxes.size(), 0));

    // Iterate through all matches
    for (const auto &match : matches)
    {
        const cv::KeyPoint &prevKpt = prevFrame.keypoints[match.queryIdx];
        const cv::KeyPoint &currKpt = currFrame.keypoints[match.trainIdx];

        // Find bounding box IDs containing keypoints in the previous and current frames
        std::vector<int> prevBoundingBoxIds, currBoundingBoxIds;

        for (size_t i = 0; i < prevFrame.boundingBoxes.size(); ++i)
        {
            if (prevFrame.boundingBoxes[i].roi.contains(prevKpt.pt))
            {
                prevBoundingBoxIds.push_back(i);
            }
        }

        for (size_t i = 0; i < currFrame.boundingBoxes.size(); ++i)
        {
            if (currFrame.boundingBoxes[i].roi.contains(currKpt.pt))
            {
                currBoundingBoxIds.push_back(i);
            }
        }

        // Increment counts for matches between bounding boxes
        for (int prevId : prevBoundingBoxIds)
        {
            for (int currId : currBoundingBoxIds)
            {
                counts[prevId][currId]++;
            }
        }
    }

    // Find the best match for each bounding box in the previous frame
    for (size_t prevId = 0; prevId < counts.size(); ++prevId)
    {
        int maxCount = 0;
        int maxId = -1;
        for (size_t currId = 0; currId < counts[prevId].size(); ++currId)
        {
            if (counts[prevId][currId] > maxCount)
            {
                maxCount = counts[prevId][currId];
                maxId = currId;
            }
        }
        if (maxId != -1)
        {
            bbBestMatches[prevId] = maxId;
        }
    }
}

