/*
 * test.cpp
 *
 *  Created on: 19 Apr 2019
 *      Author: alessioxompero
 */


// Boost library
#include <boost/lexical_cast.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/map.hpp>

#include <Frame.h>

/**
 *
 */
void Serialization_TEST() {
  std::cout << "Serialization test" << std::endl;
  DBoW2::BowVector bow1;
  bow1.addWeight(0, 0.3);
  bow1.addWeight(1, 0.75);
  bow1.addWeight(2, 0.28);
  bow1.addWeight(3, 0.34);
  bow1.addWeight(4, 0.59);

  std::stringstream ss;
  boost::archive::text_oarchive oarch(ss);
  oarch << bow1;
  DBoW2::BowVector bow2;
  boost::archive::text_iarchive iarch(ss);
  iarch >> bow2;
  std::cout << (bow1 == bow2) << std::endl;

  std::cout << bow1 << std::endl;
  std::cout << bow2 << std::endl;

  sleep(30);
}

/**
 *
 */
void FrameSerialization_TEST() {
  std::cout << "Serialization test" << std::endl;

  cv::Mat img = cv::imread("examples/img_test.png", 0);

  Frame f(0);
  f.ExtractORB(img);

  std::stringstream ss;
  boost::archive::text_oarchive oarch(ss);
  oarch << f;

  Frame f2;
  boost::archive::text_iarchive iarch(ss);
  iarch >> f2;

  std::cout << f.kps[0].pt << std::endl;
  std::cout << f2.kps[0].pt << std::endl;

  std::cout << f.descs.row(0) << std::endl;
  std::cout << f2.descs.row(0) << std::endl;

  cv::Mat ddiff = f.descs.row(0) - f2.descs.row(0);

  if ((cv::norm(f.kps[0].pt - f2.kps[0].pt) == 0) && (ddiff.dot(ddiff) == 0)) {
    std::cout << "Test passed!" << std::endl;
  } else {
    std::cerr << "Test failed!" << std::endl;
  }
}

