/**
 *
 * @fn utilites.cpp
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

#include <vector>
#include <fstream>
#include <chrono>
#include <ctime>   // localtime
#include <sstream> // stringstream
#include <iomanip> // put_time
#include <string>  // string

// OpenCV Libraries
#include <opencv2/opencv.hpp>

// STL Libraries
#include <utilities.h>

// Boost library
#include <boost/lexical_cast.hpp>


//------------------------------------------------------------------------------
//
void PrintHeading1(const std::string& heading) {
  std::cout << std::endl << std::string(78, '=') << std::endl;
  std::cout << heading << std::endl;
  std::cout << std::string(78, '=') << std::endl << std::endl;
}

//
void PrintHeading2(const std::string& heading) {
  int num_symb = (78 - heading.length()) / 2;
  std::cout << std::endl << std::string(78, '#') << std::endl;
  std::cout << std::string(num_symb, '#') << heading
            << std::string(num_symb, '#') << std::endl;
}

//
void PrintClosing2(const std::string& heading) {
  int num_symb = (78 - heading.length()) / 2;
  std::cout << std::string(num_symb, '#') << heading
            << std::string(num_symb, '#') << std::endl;
  std::cout << std::string(78, '#') << std::endl << std::endl;
}

// The software logs errors to file. This function creates a error string
// to output in the log file when called.
void LogInfo(const std::string& ss) {
  std::cerr << ss << std::endl;
}


void PrintZMQversion() {
  int major, minor, patch;
  zmq_version(&major, &minor, &patch);
  printf("Current ØMQ version is %d.%d.%d\n", major, minor, patch);
}


int ShowHelp() {
  PrintHeading1(
      "Synchronized cross-camera video processing - Testing");

  std::cout << "Usage:" << std::endl;
  std::cout
      << "  ./Agent [agent_id] [TCP Port Server] [TCP Port Client] [videopath] [configuration file] [options]"
      << std::endl;
  std::cout << "options:" << std::endl;
  std::cout << "  --Maximum number of frames [default: all the available (-1)]"
            << std::endl;
  std::cout << std::endl;

  std::cout << "Examples:" << std::endl;
  std::cout << " ./Agent 1 5555 5556 ../sequence/ example/config.yaml"
            << std::endl;
  std::cout << " ./Agent 1 5555 5556 ../sequence/ example/config.yaml 5"
            << std::endl;
  std::cout << " ./Agent 1 5555 5556 ../sequence/ example/config.yaml 5 50"
            << std::endl;
  std::cout << std::endl;

  return EXIT_SUCCESS;
}

//------------------------------------------------------------------------------
//
void ConvertMatDescToTDescriptorVec(
    const cv::Mat& M, std::vector<DBoW2::FORB::TDescriptor>& vDescs) {
  int n_rows = M.rows;
  for (size_t r = 0; r < n_rows; ++r) {
    vDescs.push_back(M.row(r));
  }
  assert(vDescs.size() == n_rows);
}

/**
 *
 */
DBoW2::BowVector ConvertBoWstring2BoWvec(std::string& s) {
  // De-serialize the BoW vector
  DBoW2::BowVector feat_vec;

  std::cout << s << std::endl;

  std::string delimiter = ">";
  std::string delimiter2 = ",";

  size_t pos = 0, pos2;
  std::string token, tkn1, tkn2;

  while ((pos = s.find(delimiter)) != std::string::npos) {
    token = s.substr(1, pos);
    s.erase(0, pos + delimiter.length());

    std::cout << token << std::endl;

    pos2 = token.find(delimiter2);

    tkn1 = token.substr(0, pos2);
    tkn2 = token.erase(0, pos2 + delimiter.length());

    std::cout << tkn1 << std::endl;
    std::cout << tkn2 << std::endl;

    uint bow_id = boost::lexical_cast<uint>(tkn1);
    double bow_freq = boost::lexical_cast<double>(tkn2);

    feat_vec.addIfNotExist(bow_id, bow_freq);

    std::cin.get();
  }
  std::cout << s << std::endl;

  return feat_vec;
}

/**
 *
 */
void PrintBoWvec(const DBoW2::BowVector& bow_vec) {
  int cnt = 0;
  for (auto& it1 : bow_vec) {
    if (cnt >= 5)
      break;

    std::cout << "<" << it1.first << " - " << it1.second << ">";
    ++cnt;
  }
  std::cout << std::endl;
}



//
void ReadQueryFrameId(const std::string& vpr_filename,
                      std::vector<std::pair<int,int>>& quey_frame_id_vec) {
  quey_frame_id_vec.empty();

  // Open the key-point file provided by Heinly.
  std::ifstream vpr_file(vpr_filename.c_str());

  if (!vpr_file.is_open()) {
    std::string msg = "Error opening file" + vpr_filename;
    perror(msg.c_str());
    return;
  }

  std::string line;  //Container for the line to read

  int curr_frame_id;
  int query_fram_id;

  while (getline(vpr_file, line)) {
    std::istringstream in(line);
    in >> curr_frame_id;
    in >> query_fram_id;

    quey_frame_id_vec.push_back(std::make_pair(curr_frame_id,query_fram_id));
  }

  vpr_file.close();
}


//
void LoadImage(const int& agent_id, cv::VideoCapture &cam, cv::Mat& gray_img) {
//  std::cout << "Loading Image" << std::endl;
  cv::Mat rgb_img;

  cam >> rgb_img;  // Get a new frame from the video

  if (rgb_img.empty()) {
    std::cerr
        << "Agent " + std::to_string(agent_id)
            + ": !!! Failed imread(): image not found !!!"
        << std::endl;
    return;
  }

  // Convert RGB to gray image
  if (rgb_img.channels() == 3) {
    cvtColor(rgb_img, gray_img, cv::COLOR_RGB2GRAY);
  } else {
    gray_img = rgb_img.clone();
  }
//  std::cout << "Image loaded!" << std::endl;
}




std::string getCurrentTimeAndDateString() {
  auto now = std::chrono::high_resolution_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);

  auto fine = std::chrono::time_point_cast<std::chrono::nanoseconds>(now);

  double param = fine.time_since_epoch().count() * 1e-9;
  double fractpart, intpart;
  fractpart = modf (param , &intpart);

  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %T");
  return ss.str() + std::to_string(fractpart).substr(1);
}


void saveQueryResultsToFile(const DBoW2::QueryResults& qret, const int& agent_id,
		const int& query_id, const std::string& filename) {
	std::ofstream file_res(
			"agent" + std::to_string(agent_id) + "_" + filename + ".txt",
			std::ios::app);

	file_res << "query_id:" << query_id << ";" << "qret_size:" <<(int)qret.size() << ";";
	for (DBoW2::QueryResults::const_iterator qit = qret.begin(); qit != qret.end(); ++qit) {
		file_res << qit->Id << ":" << qit->Score << ";";
	}
	file_res << "\n";
	file_res.close();
}


