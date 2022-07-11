/**
 *
 * @fn unit_tests.cpp
 *
 *        Author:   Alessio Xompero
 *  Created Date:   2022/05/07
 *
 *
 ***********************************************************************
 * Contacts:
 *      Alessio Xompero:    a.xompero@qmul.ac.uk
 *
 * Centre for Intelligent Sensing, Queen Mary University of London, UK
 *
 ***********************************************************************
 *
 * Modified Date:   2022/05/17
 *
 *******************************************************************************
 MIT License

 Copyright (c) 2022 Alessio

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 *******************************************************************************
 */

// STL Libraries

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <DVision/DVision.h>

#include <utilities.h>

// --------------------------------------------------------------------------
void TestFundamentalMatrixDLib() {
	PrintHeading2(" Test Fundamental Matrix (DLib) ");

//	const double points1_raw[] = { 93.4400, 28.4400, 8.4400, 215.4400,
//				128.5600, 112.4400, 129.5600, 183.4400, 183.5600, 69.5600,
//				222.4400, 69.4400, 223.0600, 120.0600, 336.1900, 270.0600,
//				82.0600, 244.4400, 61.3100, 231.8100, 202.4400, 202.8100,
//				278.9400, 126.0600, 64.9400, 175.8100, 46.8100, 172.4400,
//				31.6900, 258.8100, 60.0600, 251.9400, 33.4400, 76.0600,
//				299.4400, 11.5600 };
//
//	const double points2_raw[] = { 115.9700, 45.9400, 32.7300, 229.3100,
//			149.7700, 127.1900, 150.5500, 197.4400, 203.9900, 84.4400, 243.1800,
//			84.4400, 241.7200, 134.5600, 361.1100, 287.6900, 239.9400, 231.6900,
//			231.0600, 217.5600, 286.3100, 200.5600, 300.3200, 140.1900, 49.9400,
//			175.6900, 105.6900, 198.5600, 276.0600, 173.0600, 295.8100,
//			173.0600, 50.3100, 98.8100, 318.3000, 25.8100 };
//
//	const size_t kNumPoints = 18;

	const double points1_raw[] = { 93.4400, 28.4400, 8.4400, 215.4400,
				128.5600, 112.4400, 129.5600, 183.4400, 183.5600, 69.5600,
				222.4400, 69.4400, 223.0600, 120.0600, 336.1900, 270.0600,
				278.9400, 126.0600, 299.4400, 11.5600 };

	const double points2_raw[] = { 115.9700, 45.9400, 32.7300, 229.3100,
			149.7700, 127.1900, 150.5500, 197.4400, 203.9900, 84.4400,
			243.1800,  84.4400, 241.7200, 134.5600, 361.1100, 287.6900,
			300.3200, 140.1900, 318.3000, 25.8100 };

	const size_t kNumPoints = 10;
	std::vector<cv::Point2f> points1(kNumPoints);
	std::vector<cv::Point2f> points2(kNumPoints);
	for (size_t i = 0; i < kNumPoints; ++i) {
		points1[i] = cv::Point2f(points1_raw[2 * i], points1_raw[2 * i + 1]);
		points2[i] = cv::Point2f(points2_raw[2 * i], points2_raw[2 * i + 1]);
	}

	// I verified that matched points are the same if using the same input image
	cv::Mat oldMat(points1.size(), 2, CV_32F, &points1[0]);
	cv::Mat curMat(points2.size(), 2, CV_32F, &points2[0]);

  cv::Mat hconcatmat;
  cv::hconcat(oldMat,curMat,hconcatmat);
  std::cout<< hconcatmat << std::endl;

	std::vector<uchar> inliers;
	/// To compute the fundamental matrix
	DVision::FSolver m_fsolver(400, 300);
	cv::Mat fundMat = m_fsolver.findFundamentalMat(oldMat, curMat, 2.0, 10,
			&inliers, true, 0.99, 2000);
	std::cout << fundMat << std::endl;


	  const double c = 400;
	  const double r = 300;

	  cv::Mat m_N = (cv::Mat_<double>(3,3) <<
	    1./c, 0, -0.5,
	    0, 1./r, -0.5,
	    0, 0, 1);

	  cv::Mat m_N_t = m_N.t();

	cv::Mat Fdlib = m_N * fundMat * m_N_t;
	std::cout << Fdlib << std::endl;

//  cv::Mat GT = (cv::Mat_<double>(3,3) << -0.217859, 0.419282, -0.0343075, -0.0717941, 0.0451643, 0.0216073, 0.248062, -0.429478, 0.0221019);
//  std::cout << GT << std::endl;

	cv::Mat GT = (cv::Mat_<double>(3,3) << 0.000004705974925, -0.000360910194893,  0.034827168188560,  0.000364450254288,  0.000003365362731, -0.093727763186923, -0.042551489201813,  0.099307562681452,  0.989164253900420);
  	GT /= GT.at<double>(2,2);
    std::cout << GT << std::endl;


	if (!fundMat.empty()) {
		std::vector<uchar> status;
		// Reference values obtained from Matlab.
		Fdlib /= Fdlib.at<double>(2,2);
		std::cout << Fdlib << std::endl;

		cv::Mat F_diff = Fdlib - GT;

		status.push_back(std::abs(F_diff.at<double>(0, 0)) < 1e-4);
		status.push_back(std::abs(F_diff.at<double>(0, 1)) < 1e-4);
		status.push_back(std::abs(F_diff.at<double>(0, 2)) < 1e-4);
		status.push_back(std::abs(F_diff.at<double>(1, 0)) < 1e-4);
		status.push_back(std::abs(F_diff.at<double>(1, 1)) < 1e-4);
		status.push_back(std::abs(F_diff.at<double>(1, 2)) < 1e-4);
		status.push_back(std::abs(F_diff.at<double>(2, 0)) < 1e-4);
		status.push_back(std::abs(F_diff.at<double>(2, 1)) < 1e-4);
		status.push_back(std::abs(F_diff.at<double>(2, 2)) < 1e-4);

		if (cv::countNonZero(status) == 9)
		      std::cout << "Fundamental matrix verification passed!" << std::endl;
		    else
		      std::cout << "Fundamental matrix verification not passed" << std::endl;
	} else {
		std::cerr << "Fundamental matrix verification not passed" << std::endl;
		std::cout << "Fundamental matrix verification not passed" << std::endl;
	}

	PrintClosing2(" Test Fundamental Matrix (DLib) ");
}

void TestFundamentalMatrixOpenCV() {
	PrintHeading2(" Test Fundamental Matrix (OpenCV) ");

	//	const double points1_raw[] = { 1.839035, 1.924743, 0.543582, 0.375221,
	//			0.473240, 0.142522, 0.964910, 0.598376, 0.102388, 0.140092,
	//			15.994343, 9.622164, 0.285901, 0.430055, 0.091150, 0.254594 };

//		const double points1_raw[] = { 93.4400, 28.4400, 8.4400, 215.4400,
//				128.5600, 112.4400, 129.5600, 183.4400, 183.5600, 69.5600,
//				222.4400, 69.4400, 223.0600, 120.0600, 336.1900, 270.0600,
//				82.0600, 244.4400, 61.3100, 231.8100, 202.4400, 202.8100,
//				278.9400, 126.0600, 64.9400, 175.8100, 46.8100, 172.4400,
//				31.6900, 258.8100, 60.0600, 251.9400, 33.4400, 76.0600,
//				299.4400, 11.5600 };

		const double points1_raw[] = { 93.4400, 28.4400, 8.4400, 215.4400,
						128.5600, 112.4400, 129.5600, 183.4400, 183.5600, 69.5600,
						222.4400, 69.4400, 223.0600, 120.0600, 336.1900, 270.0600};

		//	const double points2_raw[] = { 1.002114, 1.129644, 1.521742, 1.846002,
	//			1.084332, 0.275134, 0.293328, 0.588992, 0.839509, 0.087290,
	//			1.779735, 1.116857, 0.878616, 0.602447, 0.642616, 1.028681, };

//	const double points2_raw[] = { 115.9700, 45.9400, 32.7300, 229.3100,
//			149.7700, 127.1900, 150.5500, 197.4400, 203.9900, 84.4400, 243.1800,
//			84.4400, 241.7200, 134.5600, 361.1100, 287.6900, 239.9400, 231.6900,
//			231.0600, 217.5600, 286.3100, 200.5600, 300.3200, 140.1900, 49.9400,
//			175.6900, 105.6900, 198.5600, 276.0600, 173.0600, 295.8100,
//			173.0600, 50.3100, 98.8100, 318.3000, 25.8100 };

	const double points2_raw[] = { 115.9700, 45.9400, 32.7300, 229.3100,
			149.7700, 127.1900, 150.5500, 197.4400, 203.9900, 84.4400, 243.1800,
			84.4400, 241.7200, 134.5600, 361.1100, 287.6900};

	const size_t kNumPoints = 8;
	std::vector<cv::Point2f> points1(kNumPoints);
	std::vector<cv::Point2f> points2(kNumPoints);
	for (size_t i = 0; i < kNumPoints; ++i) {
		points1[i] = cv::Point2f(points1_raw[2 * i], points1_raw[2 * i + 1]);
		points2[i] = cv::Point2f(points2_raw[2 * i], points2_raw[2 * i + 1]);
	}

//	double f = 300;
//	double cx = 320;
//	double cy = 240;
//
//	std::vector<cv::Point2f> n_points1;
//	std::vector<cv::Point2f> n_points2;
//	for (auto pt : points1){
//	  cv::Point2f npt;
//	  npt.x = f * pt.x + cx;
//	  npt.y = f * pt.y + cy;
//	  n_points1.push_back(npt);
//	}
//	for (auto pt : points2){
//	    cv::Point2f npt;
//	    npt.x = f * pt.x + cx;
//	    npt.y = f * pt.y + cy;
//	    n_points2.push_back(npt);
//	  }
//	cv::Mat tmp1, tmp2;
//	CenterAndNormalizeImagePoints(points1, &n_points1, tmp1);
//	CenterAndNormalizeImagePoints(points2, &n_points2, tmp2);

	// I verified that matched points are the same if using the same input image
	cv::Mat oldMat(points1.size(), 2, CV_32F, &points1[0]);
	cv::Mat curMat(points2.size(), 2, CV_32F, &points2[0]);

	cv::Mat hconcatmat;
	cv::hconcat(oldMat,curMat,hconcatmat);
	std::cout<< hconcatmat << std::endl;

	// OpenCV fundamental matrix estimation
	cv::Mat Fcv = cv::findFundamentalMat(oldMat, curMat, cv::FM_8POINT, 2, 0.99);
//	cv::Mat Fcv = cv::findFundamentalMat(oldMat, curMat, cv::LMEDS, 2, 0.99);
//	cv::Mat Fcv = cv::findFundamentalMat(oldMat, curMat, cv::RANSAC, 2, 0.99);
	std::cout << Fcv << std::endl;

	// Reference values obtained from Matlab.
//	cv::Mat GT = (cv::Mat_<double>(3,3) << -0.217859, 0.419282, -0.0343075, -0.0717941, 0.0451643, 0.0216073, 0.248062, -0.429478, 0.0221019);
//	std::cout << GT << std::endl;

	cv::Mat GT = (cv::Mat_<double>(3,3) << 0.000004705974925, -0.000360910194893,  0.034827168188560,  0.000364450254288,  0.000003365362731, -0.093727763186923, -0.042551489201813,  0.099307562681452,  0.989164253900420);
  	GT /= GT.at<double>(2,2);
    std::cout << GT << std::endl;

	cv::Mat F_diff = Fcv - GT;

	if (!Fcv.empty()) {
		std::vector<uchar> status;
		// Reference values obtained from Matlab.
		status.push_back(std::abs(F_diff.at<double>(0, 0)) < 1e-4);
		status.push_back(std::abs(F_diff.at<double>(0, 1)) < 1e-4);
		status.push_back(std::abs(F_diff.at<double>(0, 2)) < 1e-4);
		status.push_back(std::abs(F_diff.at<double>(1, 0)) < 1e-4);
		status.push_back(std::abs(F_diff.at<double>(1, 1)) < 1e-4);
		status.push_back(std::abs(F_diff.at<double>(1, 2)) < 1e-4);
		status.push_back(std::abs(F_diff.at<double>(2, 0)) < 1e-4);
		status.push_back(std::abs(F_diff.at<double>(2, 1)) < 1e-4);
		status.push_back(std::abs(F_diff.at<double>(2, 2)) < 1e-4);

		if (cv::countNonZero(status) == 9)
		  std::cout << "Fundamental matrix verification passed!" << std::endl;
		else
		  std::cout << "Fundamental matrix verification not passed" << std::endl;
	} else {
		std::cerr << "Fundamental matrix verification not passed" << std::endl;
		std::cout << "Fundamental matrix verification not passed" << std::endl;
	}

	PrintClosing2(" Test Fundamental Matrix (OpenCV) ");
}



void TestFundamentalMatrixRANSACOpenCV() {
	PrintHeading2(" Test Fundamental Matrix with RANSAC (OpenCV) ");

		const double points1_raw[] = { 93.4400, 28.4400, 8.4400, 215.4400,
				128.5600, 112.4400, 129.5600, 183.4400, 183.5600, 69.5600,
				222.4400, 69.4400, 223.0600, 120.0600, 336.1900, 270.0600,
				82.0600, 244.4400, 61.3100, 231.8100, 202.4400, 202.8100,
				278.9400, 126.0600, 64.9400, 175.8100, 46.8100, 172.4400,
				31.6900, 258.8100, 60.0600, 251.9400, 33.4400, 76.0600,
				299.4400, 11.5600 };

	const double points2_raw[] = { 115.9700, 45.9400, 32.7300, 229.3100,
			149.7700, 127.1900, 150.5500, 197.4400, 203.9900, 84.4400, 243.1800,
			84.4400, 241.7200, 134.5600, 361.1100, 287.6900, 239.9400, 231.6900,
			231.0600, 217.5600, 286.3100, 200.5600, 300.3200, 140.1900, 49.9400,
			175.6900, 105.6900, 198.5600, 276.0600, 173.0600, 295.8100,
			173.0600, 50.3100, 98.8100, 318.3000, 25.8100 };

	const size_t kNumPoints = 18;
	std::vector<cv::Point2f> points1(kNumPoints);
	std::vector<cv::Point2f> points2(kNumPoints);
	for (size_t i = 0; i < kNumPoints; ++i) {
		points1[i] = cv::Point2f(points1_raw[2 * i], points1_raw[2 * i + 1]);
		points2[i] = cv::Point2f(points2_raw[2 * i], points2_raw[2 * i + 1]);
	}

	// I verified that matched points are the same if using the same input image
	cv::Mat oldMat(points1.size(), 2, CV_32F, &points1[0]);
	cv::Mat curMat(points2.size(), 2, CV_32F, &points2[0]);

	// OpenCV fundamental matrix estimation
	cv::Mat Fcv = cv::findFundamentalMat(oldMat, curMat, cv::RANSAC, 2, 0.99);
	std::cout << Fcv << std::endl;

	// Reference values obtained from Matlab.
	cv::Mat GT = (cv::Mat_<double>(3,3) << 0.000004705974925, -0.000360910194893,  0.034827168188560,  0.000364450254288,  0.000003365362731, -0.093727763186923, -0.042551489201813,  0.099307562681452,  0.989164253900420);
  	GT /= GT.at<double>(2,2);
    std::cout << GT << std::endl;

	cv::Mat F_diff = Fcv - GT;

	if (!Fcv.empty()) {
		std::vector<uchar> status;
		// Reference values obtained from Matlab.
		status.push_back(std::abs(F_diff.at<double>(0, 0)) < 1e-5);
		status.push_back(std::abs(F_diff.at<double>(0, 1)) < 1e-5);
		status.push_back(std::abs(F_diff.at<double>(0, 2)) < 1e-5);
		status.push_back(std::abs(F_diff.at<double>(1, 0)) < 1e-5);
		status.push_back(std::abs(F_diff.at<double>(1, 1)) < 1e-5);
		status.push_back(std::abs(F_diff.at<double>(1, 2)) < 1e-5);
		status.push_back(std::abs(F_diff.at<double>(2, 0)) < 1e-5);
		status.push_back(std::abs(F_diff.at<double>(2, 1)) < 1e-5);
		status.push_back(std::abs(F_diff.at<double>(2, 2)) < 1e-5);

		if (cv::countNonZero(status) == 9)
		  std::cout << "Fundamental matrix verification passed!" << std::endl;
		else
		  std::cout << "Fundamental matrix verification not passed" << std::endl;
	} else {
		std::cerr << "Fundamental matrix verification not passed" << std::endl;
		std::cout << "Fundamental matrix verification not passed" << std::endl;
	}

	PrintClosing2(" Test Fundamental Matrix (OpenCV) ");
}
