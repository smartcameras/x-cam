/**
 *
 * @fn Frame.h
 *
 *        Author:   Alessio Xompero
 *  Created Date:   2021/12/24
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
 * Modified Date:   2021/12/24
 *
 *******************************************************************************
 MIT License

 Copyright (c) 2021 Alessio

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

#ifndef INCLUDE_FRAME_H_
#define INCLUDE_FRAME_H_

// STL Libraries
#include <sstream>
#include <stdio.h>

#include <iostream>
#include <fstream>

// OpenCV Libraries
#include <opencv2/opencv.hpp>

// include headers that implement a archive in simple text format
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>

// DBoW2
#include <DBoW2/BowVector.h>
#include <DBoW2/FeatureVector.h>
#include <DBoW2/FORB.h>
#include <DBoW2/TemplatedVocabulary.h>

#include <params.h>
#include <utilities.h>

BOOST_SERIALIZATION_SPLIT_FREE(cv::Mat)
namespace boost {
namespace serialization {

/*** Mat ***/
template<class Archive>
void save(Archive & ar, const cv::Mat& m, const unsigned int version) {
  size_t elemSize = m.elemSize(), elemType = m.type();

  ar & m.cols;
  ar & m.rows;
  ar & elemSize;
  ar & elemType;  // element type.
  size_t dataSize = m.cols * m.rows * m.elemSize();

  //cout << "Writing matrix data rows, cols, elemSize, type, datasize: (" << m.rows << "," << m.cols << "," << m.elemSize() << "," << m.type() << "," << dataSize << ")" << endl;

  for (size_t dc = 0; dc < dataSize; ++dc) {
    ar & m.data[dc];
  }
}

template<class Archive>
void load(Archive & ar, cv::Mat& m, const unsigned int version) {
  int cols, rows;
  size_t elemSize, elemType;

  ar & cols;
  ar & rows;
  ar & elemSize;
  ar & elemType;

  m.create(rows, cols, elemType);
  size_t dataSize = m.cols * m.rows * elemSize;

  //cout << "reading matrix data rows, cols, elemSize, type, datasize: (" << m.rows << "," << m.cols << "," << m.elemSize() << "," << m.type() << "," << dataSize << ")" << endl;

  for (size_t dc = 0; dc < dataSize; ++dc) {
    ar & m.data[dc];
  }
}

}
}

BOOST_SERIALIZATION_SPLIT_FREE(cv::KeyPoint)
namespace boost {
namespace serialization {

/*** KeyPoint ***/
template<class Archive>
void save(Archive & ar, const cv::KeyPoint& kp, const unsigned int version) {
  ar & kp.pt.x;
  ar & kp.pt.y;
}

template<class Archive>
void load(Archive & ar, cv::KeyPoint& kp, const unsigned int version) {
  float x, y;
  ar & x;
  ar & y;

  kp.pt.x = x;
  kp.pt.y = y;
}
}
}

typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> ORBVocabulary;

class Frame {
 public:
  Frame();

  Frame(const double &timeStamp, ORBVocabulary* voc);

  Frame(const double &timeStamp, ORBVocabulary* voc,
        const GlobalFeatType& gf_type, const int& gf_dim);

  // Copy constructor
  Frame(const Frame &f);

  /**
   *
   */
  std::string GetBoWstring();

  /**
   *
   */
  inline DBoW2::BowVector GetBoWvector() {
    return bow_vec;
  }

  /**
   *
   */
  inline cv::Mat GetGlobalDesc() {
    return global_desc.clone();
  }

  /**
   *
   */
  inline void SetGlobalDescDimensionality(const int& _dim) {
    global_desc_dim = _dim;
  }
  ;

  /**
   *
   */
  inline int GetNumberOfKeypoints() {
    return num_kps;
  }
  ;

  /**
   *
   */
  void ComputeLocalAndGlobalFeatures(const cv::Mat& rgb_img,
                                     const int& max_num_kps,
                                     const std::string& featpath);

  /**
   *
   */
  void SaveImageWithKeypoints(const cv::Mat& rgb_img, const int& agent_id);

 private:
  friend class boost::serialization::access;
  // When the class Archive corresponds to an output archive, the
  // & operator is defined similar to <<.  Likewise, when the class Archive
  // is a type of input archive the & operator is defined similar to >>.
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & frame_id;
    ar & timestamp;
    ar & num_kps;
    //ar & bow_vec;
    ar & kps;
    ar & descs;
    ar & global_desc;
  }

  /**
   *
   */
  void ReadNetVLADfeat(const std::string& filepath);

  /**
   *
   */
  void ReadDeepBitfeat(const std::string& filepath);

  /**
   *
   */
  void ComputeBoW();

  /**
   *
   */
  void ExtractORB(const cv::Mat& img, const int& max_num_kps);

 public:
  // Current and Next Frame id.
  long unsigned int frame_id;
  static long unsigned int next_frame_id;

  // Frame timestamp.
  double timestamp;

  ORBVocabulary* orb_voc;

  std::vector<cv::KeyPoint> kps;
  cv::Mat descs;

  GlobalFeatType global_fet_type;
  cv::Mat global_desc;
  int global_desc_dim;

 private:
  int num_kps;

  // Bag of words structure
  DBoW2::BowVector bow_vec;
  DBoW2::FeatureVector feat_vec;
};

#endif /* INCLUDE_FRAME_H_ */
