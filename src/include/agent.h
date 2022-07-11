/**
 *
 * @fn agent.h
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
 * Modified Date:   2022/05/13
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

#ifndef INCLUDE_AGENT_H_
#define INCLUDE_AGENT_H_

// STL Libraries
#include <iostream>
#include <iomanip>
#include <vector>
#include <mutex>

// My headers
#include <Frame.h>
#include <messages.h>
#include <params.h>
#include <timer.h>
//#include <DBoW2/TemplatedVocabulary.h>

// DLoopDetector header
#include <DLoopDetector.h>

//class Frame;

enum SynchStatus {
  InSynch = 0,
  Behind = 1,
  Advanced = 2,
};

/**
 *
 */
class Agent {

 public:
  // Constructor
  Agent();

  //
  Agent(const Params& params);

  /**
   *
   */
  inline int GetAgentID() {
    return agent_id;
  }

  /**
   *
   */
  void Run();

 private:
  /**
   *
   */
  void listener();

  /**
   * @fn ProcessingSequenceCollaborative
   *
   * @brief Process the image sequence/video in a collaborative mode, sharing
   * data with other cameras.
   */
  void ProcessingSequenceCollaborative();

  /**
   * @fn RunCollaborativeWithFixedSharingInterval
   *
   * @brief Process the image sequence/video in a collaborative mode, sharing
   * data with other cameras.
   */
  void RunCollaborativeWithFixedSharingInterval(void *requester,
                                                std::ofstream &file_res);

  /**
   * @fn RunCollaborativeWithExternalSharingInterval
   *
   * @brief Process the image sequence/video in a collaborative mode, sharing
   * data with other cameras.
   */
  void RunCollaborativeWithExternalSharingInterval(void *requester,
                                                   std::ofstream &file_res);

  /**
   * @fn ProcessingSequenceMono
   *
   * @brief Process the video sequence as a independent agent that does not
   * communicate and collaborate with any other camera.
   */
  void ProcessingSequenceMono();

  /**
   *
   */
  void ReceiveACK(void* requester, const int& k, const int& place_id,
                  std::ofstream &file_res);

  /**
   *
   */
  void ReceiveSyncMsg(void* requester);

  /**
   *
   */
  AckMsg VisualPlaceRecognition(const Msg& msg_rcv);

  /**
   *
   */
  void LoadVocabulary(const std::string& strVocFile);

  /**
   *
   */
  template<class TDetector>
  void SetDetectorParameters(const int& imwidth, const int& imheight,
                             const float& frequency);



  /**
   *
   */
  void AddBowVectorToDatabase(const Frame& f);

  /**
   *
   */
  void PrintAgentInfo(const std::string& ss);

  /**
   *
   */
  void LoadVideoCaptureAndProperties(cv::VideoCapture &cam);

  /**
   *
   */
  void SendSyncMsg(void *requester, const int& place_id);

  /**
   *
   */
  void SendPlaceRecognitionMsg(void *requester, const Frame& place_frame,
                               const int& curr_frame);

  /**
   *
   */
  void SetPauseFrameProcessing();

  /**
   *
   */
  bool CheckPauseFrameProcessing();

  /**
   *
   */
  void UnsetPauseFrameProcessing();

  /**
   *
   */
  bool CheckListenerIsFinished();

  /**
   *
   */
  void SetListenerIsFinished();

  /**
   *
   */
  void UnsetListenerIsFinished();

  /**
   *
   */
  bool CheckRepReceived();

  /**
   *
   */
  void SetRepReceived();

  /**
   *
   */
  void UnsetRepReceived();

  /**
   *
   */
  void LoopToSynchFrameProcessing();

 private:
  int agent_id; /* Identity of the agent */

  int max_num_frames;
  int ack_sent;

  Timer timestamp;
  Timer rep_ms_clock;

  bool dbproc_finished;

  std::vector<Frame> FDB;  // Database with all the processed frames

  Params a_params;

  OrbLoopDetector *detector;
  ORBVocabulary* orb_voc;  // Pre-trained vocabulary with ORB features

  mutex mMutexPauseFP;
  bool mbPauseFrameProcessing;

  mutex mMutexListener;
  bool mbListenerFinished;

  mutex mMutexReceiveREP;
  bool mbREPReceived;

  SynchStatus synch_status;
};

#endif /* INCLUDE_AGENT_H_ */
