/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/


#include "test_precomp.hpp"

class MEstimationTest: public cvtest::BaseTest
{
public:

    MEstimationTest();
    ~MEstimationTest();

    virtual void run (int) {}
};

TEST(Dummy, test)
{
    EXPECT_TRUE(true);
}

TEST(SimilarityTransform, regression)
{
    cv::Ptr<cv::videostab::MotionEstimatorRansacL2> estimator = cv::makePtr<cv::videostab::MotionEstimatorRansacL2>(cv::videostab::MM_SIMILARITY);
    estimator->setRansacParams(cv::videostab::RansacParams(5, 1.5f, 0.3, 0.99));
    estimator->setMinInlierRatio(0.5f);

    const float minAngle = 0.f, maxAngle = CV_PI;
    const float minScale = 0.33f, maxScale = 3.f;
    const float maxTranslation = 100.f;

    const int pointsNumber = 10;
    const float sigma = 1.f;

    cv::RNG rng(0xffffffff);

    for(int attempt = 0; attempt < 500000; attempt++)
    {

        const float angle = rng.uniform(minAngle, maxAngle);
        const float scale = rng.uniform(minScale, maxScale);
        const float tx = rng.uniform(-maxTranslation, maxTranslation);
        const float ty = rng.uniform(-maxTranslation, maxTranslation);

        cv::Mat transform = cv::Mat::eye(3, 3, CV_32F);
        transform.at<float>(1,1) = transform.at<float>(0,0) = scale * std::cos(angle);
        transform.at<float>(0,1) = scale*std::sin(angle);
        transform.at<float>(1,0) = -transform.at<float>(0,1);
        transform.at<float>(0,2) = tx;
        transform.at<float>(1,2) = ty;

        cv::Mat points(3, pointsNumber, CV_32F);

        for(int i = 0; i < pointsNumber; ++i)
        {
            points.at<float>(0, i) = rng.uniform(0.f, 500.f);
            points.at<float>(1, i) = rng.uniform(0.f, 500.f);
            points.at<float>(2, i) = 1.f;
        }

        //std::cout << transform << std::endl;

        cv::Mat transformedPoints = transform * points;

        for(int i = 0; i < pointsNumber; i++)
        {
            transformedPoints.at<float>(0, i) += rng.gaussian(sigma);
            transformedPoints.at<float>(1, i) += rng.gaussian(sigma);

        }

        //std::cout << transformedPoints << std::endl;

        cv::Mat src = points.rowRange(0,2).t();
        cv::Mat dst = transformedPoints.rowRange(0,2).t();

        bool isOK = false;
        const cv::Mat calculatedTransform = estimator->estimate(src.reshape(2), dst.reshape(2), &isOK);
        //const cv::Mat testPoints = calculatedTransform * points;

        if(!isOK)
        {
            std::cout << "unable to calc" << std::endl;
        }
        else
        {
            const double norm = cv::norm(calculatedTransform - transform);

            //std::cout << "norm " << norm << std::endl;

            if(norm > 7.)
            {
                std::cout << "Error " << norm << std::endl;
                std::cout << transform - calculatedTransform << std::endl;
            }
        }
    }

}
