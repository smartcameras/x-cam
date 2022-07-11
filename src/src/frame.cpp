/**
 *
 * @fn frame.cpp
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

#include <Frame.h>

long unsigned int Frame::next_frame_id = 0;

// Empty constructor
Frame::Frame() {
  num_kps = -1;
  frame_id = next_frame_id++;
  timestamp = -1;
  orb_voc = NULL;
  global_desc_dim = -1;
  global_fet_type = DBoW;
}

// Constructor
Frame::Frame(const double &timeStamp, ORBVocabulary* voc) {
  num_kps = -1;
  frame_id = next_frame_id++;
  timestamp = timeStamp;
  orb_voc = voc;

  global_fet_type = DBoW;
  SetGlobalDescDimensionality(global_desc_dim);
}

//Copy constructor
Frame::Frame(const Frame&f)
    : num_kps(f.num_kps),
      frame_id(f.frame_id),
      timestamp(f.timestamp),
      kps(f.kps),
      descs(f.descs),
      orb_voc(f.orb_voc),
      global_desc_dim(f.global_desc_dim),
      global_desc(f.global_desc),
      global_fet_type(f.global_fet_type) {
}

Frame::Frame(const double &timeStamp, ORBVocabulary* voc,
             const GlobalFeatType& gf_type, const int& gf_dim) {
  num_kps = -1;
  frame_id = next_frame_id++;
  timestamp = timeStamp;
  orb_voc = voc;

  global_fet_type = gf_type;
  SetGlobalDescDimensionality(gf_dim);
}

////////////////////////////////////////////////////////////////////////////////
//
void Frame::ComputeLocalAndGlobalFeatures(const cv::Mat& img,
                                     const int& max_num_kps,
                                     const std::string& featpath=NULL) {

  ExtractORB(img, max_num_kps);
  ComputeBoW();

  switch (global_fet_type) {
    case NetVLAD:
      ReadNetVLADfeat(featpath);
      break;
    case DeepBit:
      ReadDeepBitfeat(featpath);
      break;
    case DBoW:
      break;
  }

}

///////////////////////////////////////////////////////////////////////////////
//
void Frame::ExtractORB(const cv::Mat& rgb_img, const int& max_num_kps) {
  if (rgb_img.empty()) {
    std::cerr << "!!! Failed imread(): image not found !!!" << std::endl;
    return;
  }

  // Convert RGB to gray image
  cv::Mat mImGray = rgb_img;
  if (mImGray.channels() == 3) {
    cvtColor(mImGray, mImGray, cv::COLOR_RGB2GRAY);
  }

  cv::Ptr<cv::ORB> orb = cv::ORB::create(max_num_kps);
  std::vector<std::vector<cv::Mat> > feats;

  cv::Mat mask;

  orb->detectAndCompute(mImGray, mask, kps, descs);

  num_kps = kps.size();

//  std::cout << "# of detected keypoints: " << num_kps << std::endl;

  if (descs.type() != CV_8UC1) {
    descs.convertTo(descs, CV_8UC1);
  }
}

//
void Frame::ComputeBoW() {
  if (bow_vec.empty()) {
    std::vector<cv::Mat> vCurrentDesc;
    vCurrentDesc.reserve(descs.rows);
    for (int j = 0; j < descs.rows; j++)
      vCurrentDesc.push_back(descs.row(j));

    orb_voc->transform(vCurrentDesc, bow_vec, feat_vec, 4);  //TODO: This value 4 should be stored somewhere else
  }
}

//
std::string Frame::GetBoWstring() {
  DBoW2::BowVector::const_iterator bit;

  std::string sbow = "";

  unsigned int j = 0;
  const unsigned int N = bow_vec.size();
  for (bit = bow_vec.begin(); bit != bow_vec.end(); ++bit, ++j) {
    sbow += std::to_string(bit->first);
    sbow += "<" + std::to_string(bit->first) + ", "
        + std::to_string(bit->second) + ">";

    if (j < N - 1)
      sbow += ", ";
  }
  return sbow;
}

//------------------------------------------------------------------------------

void Frame::ReadNetVLADfeat(const std::string& filepath) {
  global_desc.create(1, global_desc_dim, CV_32FC1);

  std::ostringstream ss;
  ss << std::setw(4) << std::setfill('0') << timestamp + 1;
  std::string filename = "netvlad_" + ss.str() + ".txt";
  std::string fullfile = filepath + "/netvlad/" + filename;

  // Open the key-point file provided by Heinly.
  std::ifstream feat_file(fullfile.c_str());

  if (!feat_file.is_open()) {
    std::string msg = "Error opening file" + filename;
    perror(msg.c_str());
    return;
  }

  std::string line;  //Container for the line to read

  float val;
  int j = 0;
  while (getline(feat_file, line)) {
    if (j >= global_desc_dim) {
      break;
    }

    std::istringstream in(line);
    in >> val;

    global_desc.at<float>(j) = val;

    j++;
  }

  feat_file.close();
}

void Frame::ReadDeepBitfeat(const std::string& filepath) {
  global_desc.create(1, global_desc_dim, CV_8UC1);

  std::ostringstream ss;
  ss << std::setw(4) << std::setfill('0') << timestamp + 1;
  std::string filename = "deepbit_feats.txt";
  std::string fullfile = filepath + "/deepbit/" + filename;

  // Open the key-point file provided by Heinly.
  std::ifstream feat_file(fullfile.c_str());

  if (!feat_file.is_open()) {
    std::string msg = "Error opening file" + filename;
    perror(msg.c_str());
    return;
  }

  std::string line;  //Container for the line to read

  // We assume DeepBit 32-bit version
  int val1, val2, val3, val4, j;
  while (getline(feat_file, line)) {
    std::istringstream in(line);
    in >> j;

    if (j < timestamp) {
      continue;
    }

    in >> val1;
    global_desc.at<uchar>(0, 0) = val1;
    in >> val2;
    global_desc.at<uchar>(0, 1) = val2;
    in >> val3;
    global_desc.at<uchar>(0, 2) = val3;
    in >> val4;
    global_desc.at<uchar>(0, 3) = val4;

    break;
  }

  feat_file.close();
}


void Frame::SaveImageWithKeypoints(const cv::Mat& gray_img,
		const int& agent_id) {
	if (gray_img.empty()) {
		std::cerr << "!!! Failed imread(): image not found !!!" << std::endl;
		return;
	}

	// Convert RGB to gray image
	cv::Mat mImGray = gray_img;
	if (mImGray.channels() == 3) {
		cvtColor(mImGray, mImGray, cv::COLOR_RGB2GRAY);
	}

	if (mImGray.channels() < 3) {  //this should be always true
	    cvtColor(mImGray, mImGray, cv::COLOR_GRAY2BGR);
	  }

	const float r = 5;
	for (int j = 0; j < num_kps; j++) {
		cv::Point2f pt1, pt2;
		pt1.x = kps[j].pt.x - r;
		pt1.y = kps[j].pt.y - r;
		pt2.x = kps[j].pt.x + r;
		pt2.y = kps[j].pt.y + r;

		cv::rectangle(mImGray, pt1, pt2, cv::Scalar(0, 255, 0));
		cv::circle(mImGray, kps[j].pt, 2, cv::Scalar(0, 255, 0), -1);
	}

	// Save the frame into a file
	int n_zero = 4;
	std::string old_str = std::to_string((int)timestamp);
	auto new_str = std::string(n_zero - std::min(n_zero, (int)old_str.length()), '0') + old_str;
	std::cout << new_str << std::endl;
	std::string outfilename = "agent" + std::to_string(agent_id) + "_" + new_str + "_kps.png";
	cv::imwrite(outfilename, mImGray);
}

