#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

/**********************************************************BASELINE MODEL************************************************************************/
int mainFunction1()
{
    std::cout << "Executing Main Function 1" << std::endl;

    //Image and Patches Load

    cv::Mat imageTocomplate = cv::imread("pratodellavalle/image_to_complete.jpg");

    cv::imshow("Corrupted Image", imageTocomplate);
    cv::waitKey(0);

    std::string patchesPath = "pratodellavalle/patches_";
    cv::String pattern = patchesPath + "/*.jpg";
    std::vector<cv::String> patches;
    cv::glob(pattern, patches);


    // SIFT features and Finding keypoints
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints_image;
    cv::Mat descriptors_image;
    sift->detectAndCompute(imageTocomplate, cv::noArray(), keypoints_image, descriptors_image);

    cv::Mat imageWithKeypoints;
    cv::drawKeypoints(imageTocomplate, keypoints_image, imageWithKeypoints, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cv::imshow("Corrupted Image with Keypoints", imageWithKeypoints);
    cv::waitKey(0);

    // Construct matcher with NORM_L2
    cv::BFMatcher matcher(cv::NORM_L2);

    // Threshold ratio 
    float threshold_ratio = 0.7f;

    // SIFT features computation for the patches
    std::vector<std::vector<cv::KeyPoint>> keypoints_patch(patches.size());
    std::vector<cv::Mat> descriptors_patch(patches.size());
    for (size_t i = 0; i < patches.size(); ++i)
    {
        cv::Mat patch = cv::imread(patches[i]);


        std::vector<cv::KeyPoint> keypoints_patch;
        cv::Mat descriptors_patch;
        sift->detectAndCompute(patch, cv::noArray(), keypoints_patch, descriptors_patch);

        std::vector<std::vector<cv::DMatch>> matches;
        matcher.knnMatch(descriptors_patch, descriptors_image, matches, 2);

        // Matching considering the threshold
        std::vector<cv::DMatch> matchRefined;
        for (size_t j = 0; j < matches.size(); ++j)
        {
            if (matches[j][0].distance < threshold_ratio * matches[j][1].distance)
            {
                matchRefined.push_back(matches[j][0]);
            }
        }


        // Transformation considering RANSAC
        std::vector<cv::Point2f> patch_ransac;
        std::vector<cv::Point2f> image_ransac;
        for (size_t j = 0; j < matchRefined.size(); ++j)
        {
            patch_ransac.push_back(keypoints_patch[matchRefined[j].queryIdx].pt);
            image_ransac.push_back(keypoints_image[matchRefined[j].trainIdx].pt);
        }


        cv::Mat homography = cv::findHomography(patch_ransac, image_ransac, cv::RANSAC, 5.0);

        // Displaying the matches with inliers
        cv::Mat imageWithInliers;
        cv::drawMatches(patch, keypoints_patch, imageTocomplate, keypoints_image, matchRefined, imageWithInliers);

        cv::imshow("Image with Inliers", imageWithInliers);

        // Fixing the image by overlaying the patches
        cv::Mat image_fixed = imageTocomplate.clone();
        cv::warpPerspective(patch, image_fixed, homography, image_fixed.size());

        cv::Mat mask;
        cv::cvtColor(image_fixed, mask, cv::COLOR_BGR2GRAY);
        imageTocomplate.setTo(0, mask);

        cv::add(imageTocomplate, image_fixed, imageTocomplate);

        cv::waitKey(0);
    }
    cv::imshow("Image with Overlayed Patch", imageTocomplate);
    cv::waitKey(0);

    return 0;
}


/*******************************************AFFINE TRANSFORM ESTIMATION AND RANSAC MANUAL IMPLEMENTATION***************************************************/

int mainFunction2()
{
    std::cout << "Executing Main Function 2" << std::endl;

    //Loading image and patches
    cv::Mat imageTocomplate = cv::imread("pratodellavalle/image_to_complete.jpg");

    cv::imshow("Corrupted Image", imageTocomplate);
    cv::waitKey(0);


    std::string patchesPath = "pratodellavalle/patches_";
    cv::String pattern = patchesPath + "/*.jpg";  // Modify the pattern if your patches have a different file extension
    std::vector<cv::String> patches;
    cv::glob(pattern, patches);

    //SIFT features from the image
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints_image;
    cv::Mat descriptors_image;
    sift->detectAndCompute(imageTocomplate, cv::noArray(), keypoints_image, descriptors_image);

    cv::Mat imageWithKeypoints;
    cv::drawKeypoints(imageTocomplate, keypoints_image, imageWithKeypoints, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cv::imshow("Corrupted Image with Keypoints", imageWithKeypoints);
    cv::waitKey(0);

    // Matcher with NORM_L2
    cv::BFMatcher matcher(cv::NORM_L2);

    //threshold ratio for matching
    float threshold_ratio = 0.7f;

    // SIFT features extraction from the patches
    for (size_t i = 0; i < patches.size(); ++i)
    {
        cv::Mat patch = cv::imread(patches[i]);


        std::vector<cv::KeyPoint> keypoints_patch;
        cv::Mat descriptors_patch;
        sift->detectAndCompute(patch, cv::noArray(), keypoints_patch, descriptors_patch);

        // matching
        std::vector<std::vector<cv::DMatch>> matches;
        matcher.knnMatch(descriptors_patch, descriptors_image, matches, 2);

        std::vector<cv::DMatch> matchRefined;
        for (size_t j = 0; j < matches.size(); ++j)
        {
            if (matches[j][0].distance < threshold_ratio * matches[j][1].distance)
            {
                matchRefined.push_back(matches[j][0]);
            }
        }

        int numIterations = 200;  // Number of RANSAC iterations
        double threshold_manual = 3.0;  // Threshold distance to consider a match as an inlier

        // Variables to the best model and its inlier counts
        cv::Mat bestModel;
        int counts = 0;

        // RANSAC iteration
        for (int iter = 0; iter < numIterations; ++iter)
        {
            // selecting matches randomly 
            std::vector<cv::DMatch> randomMatches;
            std::set<int> indices;
            while (indices.size() < 3)
            {
                int randomIndex = cv::theRNG().uniform(0, matchRefined.size());
                if (indices.count(randomIndex) == 0)
                {
                    randomMatches.push_back(matchRefined[randomIndex]);
                    indices.insert(randomIndex);
                }
            }

            // Affine transform estimation
            std::vector<cv::Point2f> source_at;
            std::vector<cv::Point2f> dest_at;
            for (const auto& match : randomMatches)
            {
                source_at.push_back(keypoints_patch[match.queryIdx].pt);
                dest_at.push_back(keypoints_image[match.trainIdx].pt);
            }
            cv::Mat affineTransform = cv::estimateAffinePartial2D(source_at, dest_at);

            // Count inliers
            int inlierCount = 0;
            for (const auto& match : matchRefined)
            {
                cv::Point2f srcPoint = keypoints_patch[match.queryIdx].pt;
                cv::Point2f dstPoint = keypoints_image[match.trainIdx].pt;

                cv::Mat srcPointHomogeneous(3, 1, CV_64F);
                srcPointHomogeneous.at<double>(0) = srcPoint.x;
                srcPointHomogeneous.at<double>(1) = srcPoint.y;
                srcPointHomogeneous.at<double>(2) = 1.0;

                cv::Mat dstPointTransformed = affineTransform * srcPointHomogeneous;
                double dx = dstPointTransformed.at<double>(0) - dstPoint.x;
                double dy = dstPointTransformed.at<double>(1) - dstPoint.y;
                double distance = std::sqrt(dx * dx + dy * dy);

                if (distance < threshold_manual)
                {
                    inlierCount++;
                }
            }

            // Try to update and select best model
            if (inlierCount > counts)
            {
                bestModel = affineTransform;
                counts = inlierCount;
            }
        }

        // Overlaying the patches and displaying the corrupted image with manual implementation with best model

        cv::Mat image_fixed = imageTocomplate.clone();
        cv::warpAffine(patch, image_fixed, bestModel, image_fixed.size(), cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);

        cv::Mat mask;
        cv::cvtColor(image_fixed, mask, cv::COLOR_BGR2GRAY);
        imageTocomplate.setTo(0, mask);

        cv::add(imageTocomplate, image_fixed, imageTocomplate);

        cv::imshow("Image with Overlayed Patch", imageTocomplate);
        cv::waitKey(0);
    }

    cv::imshow("Final Result", imageTocomplate);
    cv::waitKey(0);

    return 0;
}

/******************************************************************TEMPLETE MATCHING***********************************************************************/

int mainFunction3()
{
    //Load image and patches
    cv::Mat imageTocomplate = cv::imread("pratodellavalle/image_to_complete.jpg");
    cv::imshow("Corrupted Image", imageTocomplate);
    cv::waitKey(0);

    std::string patchesPath = "pratodellavalle/patches_";
    cv::String pattern = patchesPath + "/*.jpg";
    std::vector<cv::String> patches;
    cv::glob(pattern, patches);

    //Templete matching
    cv::Mat resultImage = imageTocomplate.clone();

    for (const auto& patchPath : patches)
    {
        cv::Mat patch = cv::imread(patchPath);

        cv::Mat result;
        cv::matchTemplate(imageTocomplate, patch, result, cv::TM_CCOEFF_NORMED);

        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

        cv::Rect matchRect(maxLoc, cv::Size(patch.cols, patch.rows));
        cv::rectangle(resultImage, matchRect, cv::Scalar(0, 0, 255), 2);

        // putting matched patches to the corrupted region
        patch.copyTo(resultImage(matchRect));
    }

    cv::imshow("Image with Matches", resultImage);
    cv::waitKey(0);


    return 0;
}

// TAKING CHOICES FROM THE USER TO TRY DIFFERENT OPTIONAL APPROACHES
int main()
{
    int choice;
    std::cout << "Enter your choice: 1 or 2 or 3 \n 1:[(Baseline Assignment)] \n 2:[(Manual RANSAC & Affine Est.)]: \n 3:[(Templete Matching)]: \n Your choice:";
    std::cin >> choice;

    if (choice == 1)
    {
        mainFunction1();
    }
    else if (choice == 2)
    {
        mainFunction2();
    }
    else if (choice == 3)
    {
        mainFunction3();
    }
    else
    {
        std::cout << "Invalid choice!" << std::endl;
    }

    return 0;
}
