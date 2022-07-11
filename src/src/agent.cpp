/**
 *
 * @fn agent.cpp
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

// STL Libraries
#include <cassert>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <unistd.h>
#include <thread>

// Boost library
#include <boost/lexical_cast.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/map.hpp>

// Include ZeroMQ library for distributed networking
#include <zmq.h>

// OpenCV Libraries
#include <opencv2/opencv.hpp>

// My headers
#include <agent.h>

Timer agent_timestamp;

// Constructor
Agent::Agent() {
  agent_id = -1;

  max_num_frames = -1;

  ack_sent = 0;

  dbproc_finished = false;

  // Initialised the vocabulary and the loop closure detector class
  LoadVocabulary(a_params.GetORBVocabularyFile());

  detector = new OrbLoopDetector(*orb_voc);

  mbPauseFrameProcessing = false;
  mbListenerFinished = true;
  mbREPReceived = true;

  synch_status = SynchStatus::InSynch;
}

//
Agent::Agent(const Params& _params)
    : a_params(_params) {
  agent_id = a_params.GetAgentID();

  max_num_frames = -1;

  ack_sent = 0;

  dbproc_finished = false;

  // Initialised the vocabulary and the loop closure detector class
  LoadVocabulary(a_params.GetORBVocabularyFile());

  switch (_params.GetTypeGlobalFeature()) {
    case DBoW:
      std::cout << "DBoW" << std::endl;
      detector = new OrbLoopDetector(*orb_voc);
      break;
    case NetVLAD:
      std::cout << "NetVLAD" << std::endl;
      detector = new OrbLoopDetector(*orb_voc, QueryDist::QD_L2_NORM);
      break;
    case DeepBit:
      std::cout << "DeepBit" << std::endl;
      detector = new OrbLoopDetector(*orb_voc, QueryDist::QD_Hamming);
      break;
  }

  mbPauseFrameProcessing = false;
  mbListenerFinished = true;
  mbREPReceived = true;

  synch_status = SynchStatus::InSynch;
}

////////////////////////////////////////////////////////////////////////////////
//
void Agent::Run() {
  if (a_params.GetCollaborativeMode()) {
    // Start client and servers threads
    std::thread t1(&Agent::ProcessingSequenceCollaborative, this);  // Client thread where to process the sequence
    std::thread t2(&Agent::listener, this);  // Server thread
    t2.detach();
    t1.join();

    // Keep the agent alive for replying to other peers in server mode until it
    // does not longer receive requests for more than 30 seconds.
    std::cout << agent_timestamp.elapsed() << ": " << ack_sent << std::endl;

    int j = 1;
    while (agent_timestamp.elapsed() < 30 || ack_sent == 0
        || dbproc_finished == false) {
      PrintAgentInfo(": Sleep for 10 secs ... (" + std::to_string(j) + ")");
      sleep(10);

      j++;
    }

    agent_timestamp.print("Processing Agent");
  } else {
    std::thread t1(&Agent::ProcessingSequenceMono, this);
    t1.join();

    agent_timestamp.print("Processing Agent");
  }

  PrintAgentInfo(" has finished to operate!");
}

////////////////////////////////////////////////////////////////////////////////
//
template<class TDetector>
void Agent::SetDetectorParameters(const int& imwidth, const int& imheight,
                                  const float& frequency) {
  std::cout << "Set detector parameters!" << std::endl;

  // Set loop detector parameters
  typename TDetector::Parameters params(imheight, imwidth, frequency);

  // Set loop detector parameters
//	params.image_rows = imheight;
//	params.image_cols = imwidth;
  // Parameters given by default are:
  // use nss = true
  // alpha = 0.3
  // k = 3
  // geom checking = GEOM_DI
  // di levels = 0

  // We are going to change these values individually:
  params.use_nss = true;  // use normalized similarity score instead of raw score
  params.alpha = 0.3;  // nss threshold
  params.k = 1;  // a loop must be consistent with 1 previous matches
  params.geom_check = DLoopDetector::GEOM_DI;  // use direct index for geometrical checking
  params.di_levels = 2;  // use two direct index levels

  params.matching_th = a_params.GetMatchingThreshold();
  params.b_snn = a_params.GetBooleanUseSecondNearestNeighbour();
  params.max_neighbor_ratio = a_params.GetSecondNearestNeighbourThreshold();

  // To verify loops you can select one of the next geometrical checkings:
  // GEOM_EXHAUSTIVE: correspondence points are computed by comparing all
  //    the features between the two images.
  // GEOM_FLANN: as above, but the comparisons are done with a Flann structure,
  //    which makes them faster. However, creating the flann structure may
  //    be slow.
  // GEOM_DI: the direct index is used to select correspondence points between
  //    those features whose vocabulary node at a certain level is the same.
  //    The level at which the comparison is done is set by the parameter
  //    di_levels:
  //      di_levels = 0 -> features must belong to the same leaf (word).
  //         This is the fastest configuration and the most restrictive one.
  //      di_levels = l (l < L) -> node at level l starting from the leaves.
  //         The higher l, the slower the geometrical checking, but higher
  //         recall as well.
  //         Here, L stands for the depth levels of the vocabulary tree.
  //      di_levels = L -> the same as the exhaustive technique.
  // GEOM_NONE: no geometrical checking is done.
  //
  // In general, with a 10^6 vocabulary, GEOM_DI with 2 <= di_levels <= 4
  // yields the best results in recall/time.
  // Check the T-RO paper for more information.

  detector->setParameters(params);
  std::cout << "Parameters set!" << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
/**
 * SaveLoopClosureDetectionResults
 * @brief Append loop detection results into the file.
 *
 *  @param file_res
 *  @param result
 *
 *  0: LOOP_DETECTED
 *  1: CLOSE_MATCHES_ONLY
 *  2: NO_DB_RESULTS
 *  3: LOW_NSS_FACTOR
 *  4: LOW_SCORES
 *  5: NO_GROUPS
 *  6: NO_TEMPORAL_CONSISTENCY
 *  7: NO_GEOMETRICAL_CONSISTENCY
 *  8: PLACE_DETECTED:
 */
void SaveLoopClosureDetectionResults(
    std::ofstream& file_res, const DLoopDetector::DetectionResult& result) {
  file_res << result.query << ";";
  file_res << result.match << ";";
  file_res << result.num_matches << ";";
  file_res << result.status << "\n";
}

////////////////////////////////////////////////////////////////////////////////
//
void ShowPlaceRecognitionResults(const DLoopDetector::DetectionResult& result,
                                 bool flag) {
  //std::cout << "ShowPlaceRecognitionResults!!" << std::endl;

  if (result.detection() && !flag) {
    std::cout << "- Loop found with image " << result.match << "!" << std::endl;

  } else if (result.detection() && flag) {
    std::cout << "- Place found with image " << result.match << "!"
              << std::endl;
  } else {
    if (flag) {
      std::cout << "- No place: ";
    } else {
      std::cout << "- No loop: ";
    }
    switch (result.status) {
      case DLoopDetector::CLOSE_MATCHES_ONLY:
        std::cout << "All the images in the database are very recent"
                  << std::endl;
        break;

      case DLoopDetector::NO_DB_RESULTS:
        std::cout
            << "There are no matches against the database (few features in"
            " the image?)"
            << std::endl;
        break;

      case DLoopDetector::LOW_NSS_FACTOR:
        std::cout << "Little overlap between this image and the previous one"
                  << endl;
        break;

      case DLoopDetector::LOW_SCORES:
        std::cout << "No match reaches the score threshold (alpha: " << 0.3
                  << ")" << std::endl;
        break;

      case DLoopDetector::NO_GROUPS:
        std::cout << "Not enough close matches to create groups. "
                  << "Best candidate: " << result.match << std::endl;
        break;

      case DLoopDetector::NO_TEMPORAL_CONSISTENCY:
        std::cout << "No temporal consistency (k: " << 1 << "). "
                  << "Best candidate: " << result.match << std::endl;
        break;

      case DLoopDetector::NO_GEOMETRICAL_CONSISTENCY:
        std::cout << "No geometrical consistency. Best candidate: "
                  << result.match << std::endl;
        break;

      default:
        break;
    }
  }

  //std::cout << std::endl;
}

//
AckMsg Agent::VisualPlaceRecognition(const Msg& msg_rcv) {
  Timer t;

  PrintHeading2(" VISUAL PLACE RECOGNITION ");

  AckMsg ack;
  ack.agent_id = agent_id;
  int last = FDB.size();
  ack.curr_img_proc = FDB[last - 1].timestamp;

  DLoopDetector::DetectionResult result;

  switch (a_params.GetTypeGlobalFeature()) {
    case GlobalFeatType::DBoW:
      detector->detectPlaceFromExternCamera(msg_rcv, result, agent_id);
      break;
    case GlobalFeatType::NetVLAD:
      detector->detectPlaceFromExternCameraGlobalDesc(msg_rcv, result,
                                                      agent_id);
      break;
    case GlobalFeatType::DeepBit:
      detector->detectPlaceFromExternCameraGlobalDesc(msg_rcv, result,
                                                      agent_id);
      break;
    default:
      detector->detectPlaceFromExternCamera(msg_rcv, result, agent_id);
  }

  ShowPlaceRecognitionResults(result, true);

  ack.best_frame_idx = result.match;
  ack.num_matches = result.num_matches;
  ack.num_inliers = result.num_inliers;

  if (result.status < -1 || result.status > 8) {
    std::cerr << "Wrong status: " << result.status << std::endl;
    std::cout << "Wrong status: " << result.status << std::endl;
  }

  ack.status = result.status;

  if (result.detection()) {
    ack.best_score = 1;
  } else {
//		ack.best_frame_idx = -1;
    ack.best_score = 0;
  }

  t.print("VPR");

  std::string filename = "running_times_VPR_agent" + std::to_string(agent_id)
      + ".txt";
  t.savetofile(filename, last, 5);

  PrintClosing2("END VISUAL PLACE RECOGNITION");

  return ack;
}

////////////////////////////////////////////////////////////////////////////////
//
void Agent::listener() {

  // Read the socket number
  std::string call_to_port = "tcp://*:"
      + std::to_string(a_params.GetReceiverTcpPort());

  void *context = zmq_ctx_new();
  void *responder = zmq_socket(context, ZMQ_REP);
  int rc = zmq_bind(responder, call_to_port.c_str());
  assert(rc == 0);

  PrintAgentInfo(
      " initialised and listening on port "
          + std::to_string(a_params.GetReceiverTcpPort()) + " ...");

  int cnt_finish = 0;
  while (1) {
    PrintAgentInfo(" (server): Waiting to receive package..");

    char buf[1000000] = { };  // Initialise the array to empty -> this solved a problem with numbers remained in the buffer from previous messages
    int nbytes = zmq_recv(responder, buf, 1000000, 0);
    assert(nbytes != 0);

    std::string s(buf);

    PrintAgentInfo(
        " (server): Received message (" + std::to_string(s.length() / 1024)
            + " kBs)");

    if (s.length()  < 100) {
      SynchMsg msg_rcv;
      std::stringstream ss;
      ss << s;
      boost::archive::text_iarchive ia { ss };
      ia >> msg_rcv;

      PrintAgentInfo(
          " (server): Received frame #" + std::to_string(msg_rcv.frame_id)
              + " from agent " + std::to_string(msg_rcv.agent_id)
              + "! - Synch msg");

      // This portion keep synchronized the frames when no other processing is involved
      while (FDB.empty()) {
        sleep(0.001);
      }
      int last = FDB.size();
      int my_last_frame_id = FDB[last - 1].timestamp;

      PrintAgentInfo(
          " (server): Last frame: " + std::to_string(my_last_frame_id));

      while (msg_rcv.frame_id > my_last_frame_id
          && my_last_frame_id < max_num_frames - 1) {

        if (msg_rcv.frame_id == my_last_frame_id) {
          synch_status = SynchStatus::InSynch;
          break;
        }

        if (my_last_frame_id > max_num_frames - 1)
          break;

        synch_status = SynchStatus::Behind;

        //        LogInfo(
        //            std::to_string(msg_rcv.frame_id) + " vs "
        //                + std::to_string(FDB[last - 1].frame_id));
        if (CheckPauseFrameProcessing())
          UnsetPauseFrameProcessing();

        last = FDB.size();
        my_last_frame_id = FDB[last - 1].timestamp;
//        PrintAgentInfo(
//            "Database size: " + std::to_string(last) + " ("
//                + std::to_string(my_last_frame_id) + ")");

        sleep(0.0001);
      }

      PrintAgentInfo(
          " (server): Last frame (2): "
              + std::to_string(FDB[last - 1].timestamp));

      SynchMsg ack(agent_id, FDB[last - 1].timestamp);

      std::stringstream ss2;
      boost::archive::text_oarchive oa { ss2 };
      oa << ack;
      zmq_send(responder, ss2.str().c_str(), ss2.str().length(), 0);

      PrintAgentInfo(
          " (server): ACK message [" + std::to_string(ack.frame_id)
              + "] sent!");

      if (msg_rcv.frame_id == FDB[last - 1].timestamp) {
        synch_status = SynchStatus::InSynch;

        PrintAgentInfo("Set InSynch status");

        if (CheckPauseFrameProcessing()) {
          UnsetPauseFrameProcessing();
          PrintAgentInfo("Unset Pause Frame processing");
        }
      }
//      else if (msg_rcv.frame_id < FDB[last - 1].frame_id) {
//        if (!CheckPauseFrameProcessing())
//          SetPauseFrameProcessing();
//      }

      if (msg_rcv.frame_id < FDB[last - 1].timestamp) {
        synch_status = SynchStatus::Advanced;
        PrintAgentInfo("Set Advanced status");
        SetPauseFrameProcessing();
      }

      if (CheckPauseFrameProcessing())
        PrintAgentInfo("Frame processing pause");
      else
        PrintAgentInfo("Frame processing unpause");

    } else {
      // De-serialize the BoW vector with boost
      Msg msg_rcv;
      std::stringstream ss;
      ss << s;
      boost::archive::text_iarchive ia { ss };
      ia >> msg_rcv;

      PrintAgentInfo(
          " (server): Received frame #" + std::to_string(msg_rcv.f.timestamp)
              + " from agent " + std::to_string(msg_rcv.agent_id)
              + "! - VPR msg");

      int last = FDB.size();

      if (msg_rcv.f.timestamp > FDB[last - 1].timestamp) {
        while (msg_rcv.f.timestamp > FDB[last - 1].timestamp
            && FDB[last - 1].timestamp < max_num_frames - 1) {
          synch_status = SynchStatus::Behind;
          //        LogInfo(
          //            std::to_string(msg_rcv.frame_id) + " vs "
          //                + std::to_string(FDB[last - 1].frame_id));
          if (CheckPauseFrameProcessing())
            UnsetPauseFrameProcessing();

          sleep(0.0001);

          last = FDB.size();
//          PrintAgentInfo("Database size: " + std::to_string(last));

          if (msg_rcv.f.timestamp == FDB[last - 1].timestamp) {
            synch_status = SynchStatus::InSynch;

            break;
          }

          if (FDB[last - 1].timestamp > max_num_frames - 1)
            break;
        }
      }

      SetPauseFrameProcessing();

      // Find best frame
      AckMsg ack = VisualPlaceRecognition(msg_rcv);

//      sleep(0.100);

      std::stringstream ss2;
      boost::archive::text_oarchive oa { ss2 };
      oa << ack;
      zmq_send(responder, ss2.str().c_str(), ss2.str().length(), 0);

      PrintAgentInfo(
          " (server): ACK message [" + std::to_string(ack.best_frame_idx) + ","
              + std::to_string(ack.best_score) + "] sent!");

      if (msg_rcv.f.timestamp == FDB[last - 1].timestamp) {
        synch_status = SynchStatus::InSynch;

        PrintAgentInfo("Set InSynch status");

        if (CheckPauseFrameProcessing()) {
          UnsetPauseFrameProcessing();
          PrintAgentInfo("Unset Pause Frame processing");
        }
      }

      if (msg_rcv.f.frame_id < FDB[last - 1].timestamp) {
        synch_status = SynchStatus::Advanced;
        PrintAgentInfo("Set Advanced status");
        SetPauseFrameProcessing();
      }

      if (CheckPauseFrameProcessing())
        PrintAgentInfo("Frame processing pause");
      else
        PrintAgentInfo("Frame processing unpause");

//      else if (msg_rcv.f.frame_id < FDB[last - 1].frame_id) {
//        if (!CheckPauseFrameProcessing())
//          SetPauseFrameProcessing();
//      }

      ++ack_sent;
      agent_timestamp.reset();
    }

  }
}

////////////////////////////////////////////////////////////////////////////////
//
void Agent::ProcessingSequenceCollaborative() {
  PrintHeading2(" Processing sequence (collaborative) ");

  Timer t_seq;

  // Open file where to store found places
  std::ofstream file_res("agent" + std::to_string(agent_id) + "_vpr_res.csv");
  file_res << "ProcessFrameID" << ";";
  file_res << "QueryID" << ";";
  file_res << "AgentID" << ";";
  file_res << "ProcessFrameID2" << ";";
  file_res << "MatchID" << ";";
  file_res << "Status" << ";";
  file_res << "# matches" << ";";
  file_res << "# inliers" << "\n";

// Create ZMQ context and connections
  std::string call_to_port = "tcp://localhost:"
      + std::to_string(a_params.GetSenderTcpPort());

  void *context = zmq_ctx_new();
  void *requester = zmq_socket(context, ZMQ_REQ);
  zmq_connect(requester, call_to_port.c_str());

//
  if (a_params.GetFrequencyOfSharingFeatures() == -1) {
    RunCollaborativeWithExternalSharingInterval(requester, file_res);
  } else if (a_params.GetFrequencyOfSharingFeatures() > 0) {
    RunCollaborativeWithFixedSharingInterval(requester, file_res);
  }

  PrintAgentInfo(" (client): Video processing is finished!");

  file_res.close();

  zmq_close(requester);
  zmq_ctx_destroy(context);

  dbproc_finished = true;

  PrintAgentInfo(" (client): Closing sequence processing thread!");

  t_seq.print("Processing sequence");
}

////////////////////////////////////////////////////////////////////////////////
//
void Agent::RunCollaborativeWithFixedSharingInterval(void *requester,
                                                     std::ofstream &file_res) {
  std::string f_log_runtime = "running_times_agent" + std::to_string(agent_id)
      + ".txt";

  cv::VideoCapture cam;
  LoadVideoCaptureAndProperties(cam);

  cv::Mat gray_img;

  int k = 0;  // Frame index

  for (;;) {
    if (k == max_num_frames) {
      break;
    }

    Timer t;

    PrintAgentInfo(
        "Processing frame #" + std::to_string(k) + "/"
            + std::to_string(max_num_frames - 1));

    LoopToSynchFrameProcessing();

    LoadImage(agent_id, cam, gray_img);
    Frame f(k, orb_voc, a_params.GetTypeGlobalFeature(),
            a_params.GetGlobalFeatureDimensionality());

    f.ComputeLocalAndGlobalFeatures(gray_img,
                                    a_params.GetMaximumNumberKeypoints(),
                                    a_params.GetGlobalFeaturePathAndFilename());
    PrintAgentInfo("ComputeLocalAndGlobalFeature");

    if (a_params.GetTypeGlobalFeature() == NetVLAD
        || a_params.GetTypeGlobalFeature() == DeepBit)
      detector->addFrameGlobalFeat(f.GetGlobalDesc());

    // add image to the collection and check if there is some loop
    std::vector<DBoW2::FORB::TDescriptor> orb_descs;
    ConvertMatDescToTDescriptorVec(f.descs, orb_descs);
    detector->AddBowVectorToDB(f.kps, orb_descs);

    // Save image with keypoints if set
//		f.SaveImageWithKeypoints(gray_img, agent_id);
    //---------------------------------------------------

    FDB.push_back(f);
    // Send and received only if at least N frames have been processed
    if (k > a_params.GetInitialisationWindow()
        && k % a_params.GetFrequencyOfSharingFeatures() == 0
        && f.GetNumberOfKeypoints() > 12) {
      SendPlaceRecognitionMsg(requester, f, k);
      ReceiveACK(requester, k, f.timestamp, file_res);
    } else {
      PrintAgentInfo("SendSyncMsg");

      SendSyncMsg(requester, k);
      ReceiveSyncMsg(requester);
    }

    SetPauseFrameProcessing();

    t.print("Frame Processing");
    t.savetofile(f_log_runtime, k, 4);

    ++k;
  }
}

///////////////////////////////////////////////////////////////////////////////
void Agent::RunCollaborativeWithExternalSharingInterval(
    void *requester, std::ofstream &file_res) {
  std::string f_log_runtime = "running_times_agent" + std::to_string(agent_id)
      + ".txt";

  cv::VideoCapture cam;
  LoadVideoCaptureAndProperties(cam);

  cv::Mat gray_img;

  int count = 0;
  int k = 0;  // Frame index

  std::vector<std::pair<int, int>> query_frame_ids;
  std::pair<int, int> tmp_pair;
  int q = 0;  // Index for the vector with the indexes of the query frame

  ReadQueryFrameId(a_params.GetPredefinedScheduleSharingFeaturesFile(),
                   query_frame_ids);

  for (;;) {
    if (k == max_num_frames) {
      break;
    }

    Timer t;

    PrintAgentInfo(
        "Processing frame #" + std::to_string(k) + "/"
            + std::to_string(max_num_frames - 1));

    LoadImage(agent_id, cam, gray_img);

    Frame f(k, orb_voc, a_params.GetTypeGlobalFeature(),
            a_params.GetGlobalFeatureDimensionality());
    f.ComputeLocalAndGlobalFeatures(gray_img,
                                    a_params.GetMaximumNumberKeypoints(),
                                    a_params.GetGlobalFeaturePathAndFilename());
    detector->addFrameGlobalFeat(f.GetGlobalDesc());

    //---------------------------------------------------
    bool lcd = false;

    std::vector<DBoW2::FORB::TDescriptor> orb_descs;
    ConvertMatDescToTDescriptorVec(f.descs, orb_descs);

    if (!lcd) {
      // add image to the collection and check if there is some loop
      std::cout << "AddBowVectorToDB.." << std::endl;
      detector->AddBowVectorToDB(f.kps, orb_descs);
    } else {
      // Detect loop
      std::cout << "Detect loop.." << std::endl;
      DLoopDetector::DetectionResult result;
      detector->detectLoop(f.kps, orb_descs, result);
    }

    // Save image with keypoints if set
    //		f.SaveImageWithKeypoints(gray_img, agent_id);
    //---------------------------------------------------

    tmp_pair = query_frame_ids[q];

    if (k == tmp_pair.first) {
      SendPlaceRecognitionMsg(requester, FDB[tmp_pair.second], k);
      ReceiveACK(requester, k, FDB[tmp_pair.second].timestamp, file_res);

      FDB.push_back(f);

      ++q;
    } else {
      FDB.push_back(f);
      SendSyncMsg(requester, k);
      ReceiveSyncMsg(requester);
    }

    t.print("Frame Processing");
    t.savetofile(f_log_runtime, k, 4);

    ++k;
  }
}

////////////////////////////////////////////////////////////////////////////////
//
void Agent::ProcessingSequenceMono() {
  PrintHeading1("Processing Sequence Mono");

  Timer t_seq;

  // Open file where to store found places
  std::ofstream file_res("agent" + std::to_string(agent_id) + "_vpr_res.txt");
  file_res << "QueryID" << ";";
  file_res << "MatchID" << ";";
  file_res << "# matches" << ";";
  file_res << "Status" << "\n";

  std::string f_log_runtime = "running_times_agent" + std::to_string(agent_id)
      + ".txt";

  cv::VideoCapture cam;
  LoadVideoCaptureAndProperties(cam);

  cv::Mat gray_img;

  int k = 0;  // Frame index

  for (;;) {
    if (k == max_num_frames) {
      break;
    }

    Timer t;

    PrintAgentInfo(
        "Processing frame #" + std::to_string(k) + "/"
            + std::to_string(max_num_frames - 1));

    LoadImage(agent_id, cam, gray_img);
    Frame f(k, orb_voc, a_params.GetTypeGlobalFeature(),
            a_params.GetGlobalFeatureDimensionality());

    f.ComputeLocalAndGlobalFeatures(gray_img,
                                    a_params.GetMaximumNumberKeypoints(),
                                    a_params.GetGlobalFeaturePathAndFilename());
    PrintAgentInfo("ComputeLocalAndGlobalFeature");

    if (a_params.GetTypeGlobalFeature() == NetVLAD
        || a_params.GetTypeGlobalFeature() == DeepBit)
      detector->addFrameGlobalFeat(f.GetGlobalDesc());

    //---------------------------------------------------
    // Detect loop
    std::vector<DBoW2::FORB::TDescriptor> orb_descs;
    ConvertMatDescToTDescriptorVec(f.descs, orb_descs);
    DLoopDetector::DetectionResult result;
    detector->detectLoop(f.kps, orb_descs, result);

    ShowPlaceRecognitionResults(result, true);

    SaveLoopClosureDetectionResults(file_res, result);

    // Save image with keypoints if set
    //		f.SaveImageWithKeypoints(gray_img, agent_id);
    //---------------------------------------------------

    FDB.push_back(f);

    t.print("Frame Processing");
    t.savetofile(f_log_runtime, k, 4);

    ++k;
  }

  PrintAgentInfo(" (client): Video processing is finished!");

  file_res.close();

  dbproc_finished = true;

  PrintAgentInfo(" (client): Closing sequence processing thread!");

  t_seq.print("Processing sequence");
}

////////////////////////////////////////////////////////////////////////////////
//
void Agent::PrintAgentInfo(const std::string& ss) {
  std::cout << std::endl;
  std::cout << "Agent " + std::to_string(agent_id) + ": ";
  std::cout << ss << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
//
void Agent::LoadVideoCaptureAndProperties(cv::VideoCapture &cam) {
  cam = cv::VideoCapture(a_params.GetVideoPath() + "%06d.png");
  if (!cam.isOpened()) {
    return;
  }

  if (a_params.GetNumFrames() == -1) {
    max_num_frames = cam.get(cv::CAP_PROP_FRAME_COUNT);
  } else {
    max_num_frames = std::min(a_params.GetNumFrames(),
                              (int) cam.get(cv::CAP_PROP_FRAME_COUNT));
  }

//  LogInfo("The number of frames is " + std::to_string(max_num_frames));

  int img_width = cam.get(cv::CAP_PROP_FRAME_WIDTH);
  int img_height = cam.get(cv::CAP_PROP_FRAME_HEIGHT);

  SetDetectorParameters<OrbLoopDetector>(img_width, img_height, a_params.GetFrequency());
}

////////////////////////////////////////////////////////////////////////////////
//
void Agent::LoadVocabulary(const std::string& strVocFile) {
  std::cout << std::endl << "Loading ORB Vocabulary. This could take a while..."
            << std::endl;
  orb_voc = new ORBVocabulary();

  bool bVocLoad = orb_voc->loadFromTextFile(strVocFile);
  if (!bVocLoad) {
    std::cerr << "Wrong path to vocabulary. " << std::endl;
    std::cerr << "Failed to open at: " << strVocFile << std::endl;
    exit(-1);
  }
  std::cout << "Vocabulary loaded!" << std::endl << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
//
void Agent::SendSyncMsg(void *requester, const int& place_id) {
  SynchMsg snd_msg(agent_id, place_id);
  std::stringstream ss;
  boost::archive::text_oarchive oa { ss };
  oa << snd_msg;

  if (ss.str().length() < 1024) {
    PrintAgentInfo(
        " (client): Sending message (" + std::to_string(ss.str().length())
            + " bytes) of frame " + std::to_string(place_id) + " ");
  } else {
    PrintAgentInfo(
        " (client): Sending message ("
            + std::to_string(ss.str().length() / 1024) + " kBs) of frame "
            + std::to_string(place_id) + " ");
  }

  LogInfo(
      getCurrentTimeAndDateString() + " - SND: frame "
          + std::to_string(place_id) + " (Sync)");

  zmq_send(requester, ss.str().c_str(), ss.str().length(), 0);
}

////////////////////////////////////////////////////////////////////////////////
//
void Agent::SendPlaceRecognitionMsg(void *requester, const Frame& place_frame,
                                    const int& curr_frame) {

//	std::cout << "Sending place recognition message ..." << std::endl;
//	std::cout << place_frame.descs.rowRange(0,10) << std::endl;
//	int j = 0;
//	for (auto a : place_frame.kps) {
//		if (j > 9)
//			break;
//		std::cout << a.pt << std::endl;
//		j++;
//	}

  Msg snd_msg(agent_id, place_frame);

  std::stringstream ss;
  boost::archive::text_oarchive oa { ss };
  oa << snd_msg;

  PrintAgentInfo(
      " (client): Sending message (" + std::to_string(ss.str().length() / 1024)
          + " kBs) of frame " + std::to_string(curr_frame) + " - VPR msg ");

  LogInfo(
      getCurrentTimeAndDateString() + " - SND: frame "
          + std::to_string(curr_frame) + " (VPR)");

  zmq_send(requester, ss.str().c_str(), ss.str().length(), 0);

//  ReceiveACK(requester, curr_frame, place_frame.frame_id, file_res);
}

////////////////////////////////////////////////////////////////////////////////
//
void Agent::ReceiveACK(void* requester, const int& k, const int& place_id,
                       std::ofstream &file_res) {

  UnsetRepReceived();

// Read reply
  char buffer[1024];
  zmq_recv(requester, buffer, 1024, 0);

  AckMsg ack;
  std::stringstream ss;
  ss << buffer;
  boost::archive::text_iarchive ia { ss };
  ia >> ack;

  PrintAgentInfo(
      " (client) --> ACK: " + std::to_string(ack.best_frame_idx) + ","
          + std::to_string(ack.best_score));

  std::cout << k << " | " << place_id << " | " << ack.agent_id << " | "
            << ack.curr_img_proc << " | " << ack.best_frame_idx << " | "
            << ack.num_matches << std::endl;

  // Write ACK in log file
  file_res << k << ";";
  file_res << place_id << ";";
  file_res << ack.agent_id << ";";
  file_res << ack.curr_img_proc << ";";
  file_res << ack.best_frame_idx << ";";
  file_res << ack.status << ";";
  file_res << ack.num_matches << ";";
  file_res << ack.num_inliers << "\n";

  LogInfo(
      getCurrentTimeAndDateString() + " - RCV: frame "
          + std::to_string(ack.curr_img_proc) + " (VPR)");

  SetRepReceived();
}

////////////////////////////////////////////////////////////////////////////////
//
void Agent::ReceiveSyncMsg(void* requester) {

  UnsetRepReceived();

// Read reply
  char buffer[1024];
  zmq_recv(requester, buffer, 1024, 0);

  SynchMsg ack;
  std::stringstream ss;
  ss << buffer;
  boost::archive::text_iarchive ia { ss };
  ia >> ack;

  PrintAgentInfo(" (client) --> ACK: " + std::to_string(ack.frame_id));
  LogInfo(
      getCurrentTimeAndDateString() + " - RCV: frame "
          + std::to_string(ack.frame_id) + " (Sync)");

  SetRepReceived();
}

void Agent::SetPauseFrameProcessing() {
  unique_lock<mutex> lock(mMutexPauseFP);
  mbPauseFrameProcessing = true;
}

bool Agent::CheckPauseFrameProcessing() {
  unique_lock<mutex> lock(mMutexPauseFP);
  return mbPauseFrameProcessing;
}

void Agent::UnsetPauseFrameProcessing() {
  unique_lock<mutex> lock(mMutexPauseFP);
  mbPauseFrameProcessing = false;
}

////////////////////////////////////////////////////////////////////////////////
bool Agent::CheckListenerIsFinished() {
  unique_lock<mutex> lock(mMutexListener);
  return mbListenerFinished;
}

void Agent::SetListenerIsFinished() {
  unique_lock<mutex> lock(mMutexListener);
  mbListenerFinished = false;
}

void Agent::UnsetListenerIsFinished() {
  unique_lock<mutex> lock(mMutexListener);
  mbListenerFinished = true;
}

////////////////////////////////////////////////////////////////////////////////
bool Agent::CheckRepReceived() {
  unique_lock<mutex> lock(mMutexReceiveREP);
  return mbREPReceived;
}

void Agent::SetRepReceived() {
  unique_lock<mutex> lock(mMutexReceiveREP);
  mbREPReceived = true;
}

void Agent::UnsetRepReceived() {
  unique_lock<mutex> lock(mMutexReceiveREP);
  mbREPReceived = false;
  rep_ms_clock.reset();
}

void Agent::LoopToSynchFrameProcessing() {
  while (CheckPauseFrameProcessing()) {
    sleep(0.0001);

//		switch (synch_status) {
//		case SynchStatus::InSynch:
////          PrintAgentInfo("Cameras in synch");
//			break;
//		case SynchStatus::Advanced:
////          PrintAgentInfo("ADVANCED");
//			break;
//		case SynchStatus::Behind:
////          PrintAgentInfo("BEHIND");
//			break;
//		}

    if (!CheckRepReceived()) {
      PrintAgentInfo("Waiting reply message");
      if (rep_ms_clock.elapsed() > 0.5)
        UnsetPauseFrameProcessing();
    }

    if (CheckRepReceived() && synch_status == SynchStatus::InSynch)
      UnsetPauseFrameProcessing();

//      if (CheckRepReceived() && synch_status == SynchStatus::Advanced
//          && t.elapsed() > 2)
//        UnsetPauseFrameProcessing();
  }
}
