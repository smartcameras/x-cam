/**
 *
 * @fn messages.h
 *
 *        Author:   Alessio Xompero
 *  Created Date:   2021/12/22
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

#ifndef INCLUDE_MESSAGES_H_
#define INCLUDE_MESSAGES_H_

// Boost library
#include <boost/lexical_cast.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/map.hpp>

#include <Frame.h>

//class Frame;

/*******************************************************************************
 *
 */
class Msg {
 public:
  Msg(){
    agent_id = -1;
  };

  Msg(const int& _agent_id, const Frame& _f){
    agent_id = _agent_id;
    f = _f;
  };
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & agent_id;
    ar & f;
  }
 public:
  int agent_id;
  Frame f;
};


/*******************************************************************************
 *
 */
class SynchMsg {
 public:
  // Default constructor
  SynchMsg() {
    agent_id = -1;
    frame_id = -1;
  }
  ;

  // Constructor
  SynchMsg(const int& _agent_id, const int& _frame_id) {
    agent_id = _agent_id;
    frame_id = _frame_id;
  };

 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & agent_id;
    ar & frame_id;
  }

 public:
  int agent_id;
  int frame_id;
};

/*******************************************************************************
 *
 */
class AckMsg {
 public:
  // Default constructor (empty)
  AckMsg() {
    agent_id = -1;
    best_frame_idx = -1;
    best_score = 0;
    curr_img_proc = -1;
    num_matches = -1;
    num_inliers = -1;
    status = -1;
  }
  ;

  //
  AckMsg(int frame_id, double score) {
    best_frame_idx = frame_id;
    best_score = score;
    agent_id = -1;
    curr_img_proc = -1;
    num_matches = -1;
    num_inliers = -1;
    status = -1;
  }
  ;

  //
  AckMsg(int _agent_id, int _curr_img_proc, int frame_id, double score) {
    best_frame_idx = frame_id;
    best_score = score;
    agent_id = _agent_id;
    curr_img_proc = _curr_img_proc;
    num_matches = -1;
    num_inliers = -1;
    status = -1;
  }
  ;
 private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & agent_id;
    ar & curr_img_proc;
    ar & best_frame_idx;
    ar & best_score;
    ar & num_matches;
    ar & num_inliers;
    ar & status;
  }
 public:
  int agent_id;
  int best_frame_idx;
  double best_score;
  int curr_img_proc;
  int num_matches;
  int num_inliers;
  int status;
};


#endif /* INCLUDE_MESSAGES_H_ */
