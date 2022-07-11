/**
 *
 * @fn utilities.h
 *
 *        Author:   Alessio Xompero
 *  Created Date:   2019/04/15
 *
 ***********************************************************************
 * Contacts:
 *      Alessio Xompero:    a.xompero@qmul.ac.uk
 *
 * Centre for Intelligent Sensing, Queen Mary University of London, UK
 *
 ***********************************************************************
 *
 * Modified Date:   2022/05/09
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

#ifndef INCLUDE_UTILITIES_H_
#define INCLUDE_UTILITIES_H_

// STL Libraries
#include <string>
#include <vector>

// OpenCV Libraries
#include <opencv2/opencv.hpp>

// Include ZeroMQ library for distributed networking
#include <zmq.h>

// DBoW2
#include <DBoW2/BowVector.h>
#include <DBoW2/FORB.h>
#include <DBoW2/QueryResults.h>

// DLoopDetector header
//#include <DLoopDetector.h>

//------------------------------------------------------------------------------
/**
 *
 */
void PrintHeading1(const std::string& heading);

/**
 *
 */
void PrintHeading2(const std::string& heading);

/**
 *
 */
void PrintClosing2(const std::string& heading);

/**
 * @brief The software logs errors to file. This function creates a error string
 * to output in the log file when called.
 */
void LogInfo(const std::string& ss);


/**
 * @brief Show the current ZMQ version installed and used by the software.
 */
void PrintZMQversion();

/**
 *
 */
int ShowHelp();

//------------------------------------------------------------------------------
/**
 *
 */
void ConvertMatDescToTDescriptorVec(
    const cv::Mat& M, std::vector<DBoW2::FORB::TDescriptor>& vDescs);

/**
 *
 */
DBoW2::BowVector ConvertBoWstring2BoWvec(std::string& s);

/**
 *
 */
void PrintBoWvec(const DBoW2::BowVector& bow_vec);

/******************************************************************************
 *
 * @brief Compute the Hamming distance between two 32-byte binary descriptors.
 *
 */
inline int DescriptorDistanceFAST(const cv::Mat &a, const cv::Mat &b) {
  const uchar *pa = a.ptr<uchar>();
  const uchar *pb = b.ptr<uchar>();

  CV_Assert(a.rows == b.rows && a.cols == b.cols);

  int desc_len = std::max(a.rows, a.cols);

  int dist = 0;
  for (int i = 0; i < desc_len; i++) {
    int axorb = pa[i] ^ pb[i];
    dist += __builtin_popcount(pa[i] ^ pb[i]);
  }
  return (dist);
}
;

/******************************************************************************
 *
 * @brief Read filw with list of VPR calls.where each row contains the index of
 * the processed frame and corresponding index of the query frame.
 *
 */
void ReadQueryFrameId(const std::string& vpr_filename,
                      std::vector<std::pair<int,int>>& quey_frame_id_vec);

/******************************************************************************
 *
 * @brief
 *
 */
void LoadImage(const int& agent_id, cv::VideoCapture &cam, cv::Mat& gray_img);

/******************************************************************************
 *
 * @brief
 *
 */
std::string getCurrentTimeAndDateString();



//------------------------------------------------------------
// Unit Tests

void TestFundamentalMatrixDLib();

void TestFundamentalMatrixOpenCV();

void TestFundamentalMatrixRANSACOpenCV();

void saveQueryResultsToFile(const DBoW2::QueryResults& qret, const int& agent_id,
		const int& query_id, const std::string& filename);


#endif /* INCLUDE_UTILITIES_H_ */
