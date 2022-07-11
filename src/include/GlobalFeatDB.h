/*
 * GlobalFeatDB.h
 *
 *  Created on: 7 Dec 2020
 *      Author: alessio
 */

#ifndef INCLUDE_GLOBALFEATDB_H_
#define INCLUDE_GLOBALFEATDB_H_

#include <vector>
#include <map>

#include <opencv2/opencv.hpp>
#include <utilities.h>

#include <DBoW2/QueryResults.h>

namespace DBoW2 {

/// Scoring type
enum QueryDist {
  QD_L2_NORM,
  QD_Hamming
};

class GlobalFeatDatabase {
 protected:
  std::map<EntryId, cv::Mat> feats_db;

  QueryDist query_dist;

  int m_nentries;

 public:
  /**
   * Constructor
   */
  GlobalFeatDatabase(void);

  GlobalFeatDatabase(QueryDist _query_dist);

  // Destructor
  ~GlobalFeatDatabase() {
  }
  ;

  EntryId add(const cv::Mat& global_feat);

  void query(const cv::Mat& global_feat, QueryResults &ret, int max_results = 1,
             int max_id = -1) const;

  void queryL2(const cv::Mat& global_feat, QueryResults &ret, int max_results =
                   1,
               int max_id = -1) const;

  void queryHamming(const cv::Mat& global_feat, QueryResults &ret,
                    int max_results = 1, int max_id = -1) const;

  inline unsigned int size() const {
    return m_nentries;
  }
};

}

#endif /* INCLUDE_GLOBALFEATDB_H_ */
