/**
 *
 * @fn params.h
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

#ifndef INCLUDE_PARAMS_H_
#define INCLUDE_PARAMS_H_

#include <iostream>
#include <string>
#include <utilities.h>

enum GlobalFeatType {
  DBoW = 0,
  NetVLAD = 1,
  DeepBit = 2
};

class Params {
 public:
  Params() {
    agent_id = 1;

    num_max_frames = -1;

    max_num_kps = 1000;
    matching_th = -1;
    b_snn = false;
    snn_th = 0.6;

    gf_type = GlobalFeatType::DBoW;
    gf_dim = -1;

    init_wnd = 30;
    feat_sharing_freq = 5;
    frequency = 1;
    synch_mode = false;
    collaborative_mode = true;

    rcv_tcp_port = 5555;
    snd_tcp_port = 5556;

    videopath = "";
    respath = "../results/";
    strVocFile = "./Vocabulary/ORBvoc.txt";
    vprcallfilename = "";
    featpath = "../feats/backyard/view1/netvlad/netvlad_feats.txt";
  }
  ;

  // Copy Constructor
  Params(const Params &a) {
    agent_id = a.agent_id;

    num_max_frames = a.num_max_frames;

    max_num_kps = a.max_num_kps;
    matching_th = a.matching_th;
    b_snn = a.b_snn;
    snn_th = a.snn_th;

    gf_type = a.gf_type;
    gf_dim = a.gf_dim;

    init_wnd = a.init_wnd;
    feat_sharing_freq = a.feat_sharing_freq;
    frequency = a.frequency;
    synch_mode = a.synch_mode;
    collaborative_mode = a.collaborative_mode;

    rcv_tcp_port = a.rcv_tcp_port;
    snd_tcp_port = a.snd_tcp_port;

    videopath = a.videopath;
    respath = a.respath;
    strVocFile = a.strVocFile;
    vprcallfilename = a.vprcallfilename;
    featpath = a.featpath;
  }

  ~Params() {
  }
  ;

  void ParseInputParams(int argc, char *argv[]) {
    agent_id = atoi(argv[1]);
    rcv_tcp_port = atoi(argv[2]);
    snd_tcp_port = atoi(argv[3]);

    videopath = argv[4];
    respath = argv[5];
    strVocFile = argv[6];

    num_max_frames = atoi(argv[7]);
    max_num_kps = atoi(argv[8]);
    matching_th = atoi(argv[9]);
    b_snn = atoi(argv[10])==1;
    snn_th = atof(argv[11]);

    gf_type = static_cast<GlobalFeatType>(atoi(argv[12]));
    gf_dim = atoi(argv[13]);
    featpath = argv[14];

    collaborative_mode = atoi(argv[15]) == 1;
    init_wnd = atoi(argv[16]);
    feat_sharing_freq = atoi(argv[17]);
    synch_mode = atoi(argv[18]) == 1;
    vprcallfilename = argv[19];

    frequency = atoi(argv[20]);
  }
  ;

  void PrintParams() {
    //PrintHeading2("Parameters setting");

    std::cout << "Agent ID: " << agent_id << std::endl;

    std::cout << "TCP socket port of the receiver: " << rcv_tcp_port
              << std::endl;
    std::cout << "TCP socket port of the sender: " << snd_tcp_port << std::endl;

    std::cout << "Maximum number of Frames: " << num_max_frames << std::endl;
    std::cout << "Matching threshold: " << matching_th << std::endl;
    std::cout << "Use SNN: " << (int) b_snn << std::endl;
    std::cout << "SNN threshold: " << snn_th << std::endl;
    std::cout << "Global feature dim: " << gf_dim << std::endl;
    std::cout << "Maximum target number of keypoints: " << max_num_kps << std::endl;

    switch (gf_type) {
      case DBoW:
        std::cout << "Global feature type: DBoW" << std::endl;
        break;
      case NetVLAD:
        std::cout << "Global feature type: NetVLAD" << std::endl;
        break;
      case DeepBit:
        std::cout << "Global feature type: DeepBit" << std::endl;
        break;
    }

    std::cout << "Number of frames for initialisation window: " << init_wnd
              << std::endl;
    std::cout << "Frequency of sharing features: " << feat_sharing_freq
              << std::endl;
    std::cout << "Frequency: " << frequency << std::endl;
    std::cout << "Synchronisation mode: " << (int) synch_mode << std::endl;

    std::cout << "Collaborative mode: " << (int) collaborative_mode << std::endl;

    std::cout << "Video path: " << videopath << std::endl;
    std::cout << "Results path: " << respath << std::endl;
    std::cout << "Feature path: " << featpath << std::endl;
  }
  ;

  //----------------------------------------------------------------------------
  // Get properties of the class
  inline int GetAgentID() const {
    return agent_id;
  }
  ;

  inline int GetReceiverTcpPort() const {
    return rcv_tcp_port;
  }
  ;

  inline int GetSenderTcpPort() const {
    return snd_tcp_port;
  }
  ;

  inline int GetNumFrames() const {
    return num_max_frames;
  }
  ;

  inline int GetMaximumNumberKeypoints() const {
      return max_num_kps;
    }
    ;

  inline int GetMatchingThreshold() const {
    return matching_th;
  }
  ;

  inline float GetFrequency() const {
     return frequency;
   }
   ;

  inline bool GetBooleanUseSecondNearestNeighbour() const {
    return b_snn;
  }
  ;

  inline float GetSecondNearestNeighbourThreshold() const {
    return snn_th;
  }
  ;

  inline GlobalFeatType GetTypeGlobalFeature() const {
    return gf_type;
  }
  ;

  inline int GetGlobalFeatureDimensionality() const {
    return gf_dim;
  }
  ;

  inline int GetInitialisationWindow() const {
    return init_wnd;
  }
  ;

  inline int GetFrequencyOfSharingFeatures() const {
    return feat_sharing_freq;
  }
  ;

  inline bool GetCollaborativeMode() const {
    return collaborative_mode;
  }
  ;

  inline bool GetSynchronizationMode() const {
    return synch_mode;
  }
  ;

  inline std::string GetVideoPath() const {
    return videopath;
  }
  ;

  inline std::string GetResultsPath() const {
    return respath;
  }
  ;

  inline std::string GetORBVocabularyFile() const {
    return strVocFile;
  }
  ;

  inline std::string GetPredefinedScheduleSharingFeaturesFile() const {
    return vprcallfilename;
  }
  ;

  inline std::string GetGlobalFeaturePathAndFilename() const {
    return featpath;
  }
  ;

 public:

 protected:
  int agent_id;  // Set the agent ID

  int num_max_frames;  // Maximum number of frames to process in the sequence (default: -1, i.e., process all frames)
  int matching_th;  // Threshold for matching binary features (e.g., 50 for 256-dimensionality features, such as ORB or BRIEF)
  bool b_snn;  // Boolean to use the Lowe's ratio test or second nearest neighbour
  float snn_th;  // Threshold for Lowe's ratio test or second nearest neighbour (e.g., 0.6 or 0.8)

  int max_num_kps; // Maximum number of keypoints (e.g., FAST/ORB corner points, ) to localise/detect for each image

  GlobalFeatType gf_type;  // Type of global feature to use (i.e., DBoW, NetVLAD, DeepBit)
  int gf_dim;  // Dimensionality of the global feature

  int init_wnd;  // Initialisation window: Minimum number of frames before sharing features (default value: 30)
  int feat_sharing_freq;  // Frequency to share the features for cross-camera place recognition
  float frequency;

  bool collaborative_mode;
  bool synch_mode;  // Boolean variable to set the per-frame synchronisation mode between the two cameras

  int rcv_tcp_port;
  int snd_tcp_port;

  std::string videopath;
  std::string respath;
  std::string strVocFile;
  std::string vprcallfilename;
  std::string featpath;
};

#endif /* INCLUDE_PARAMS_H_ */
