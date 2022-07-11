/*
 * GlobalFeatDB.cpp
 *
 *  Created on: 5 Jan 2022
 *      Author: alessio
 */

#include <GlobalFeatDB.h>

using namespace DBoW2;

//////////////////

GlobalFeatDatabase::GlobalFeatDatabase() {
	query_dist = QueryDist::QD_L2_NORM;
	m_nentries = 0;
}

// ---------------------------------------------------------------------------

GlobalFeatDatabase::GlobalFeatDatabase(QueryDist _query_dist) {
	query_dist = _query_dist;
	m_nentries = 0;
}

// ---------------------------------------------------------------------------

EntryId GlobalFeatDatabase::add(const cv::Mat& global_feat) {
	EntryId entry_id = m_nentries++;

	feats_db[entry_id] = global_feat.clone();

	return entry_id;
}

// ---------------------------------------------------------------------------

void GlobalFeatDatabase::query(const cv::Mat& global_feat, QueryResults &ret,
		int max_results, int max_id) const {

	std::cout << "Query the database.." << std::endl;

	ret.resize(0);

	switch (query_dist) {
	case QD_L2_NORM:
		queryL2(global_feat, ret, max_results, max_id);
		break;
	case QD_Hamming:
		queryHamming(global_feat, ret, max_results, max_id);
		break;
	}
}

// ---------------------------------------------------------------------------

void GlobalFeatDatabase::queryL2(const cv::Mat& global_feat, QueryResults &ret,
		int max_results, int max_id) const {

	std::map<EntryId, double> pairs;
	std::map<EntryId, double>::iterator pit;

	if (feats_db.empty() || feats_db.size() == 0) {
		std::cout << "Global Feature Database is empty!" << std::endl;
		return;
	}

//  std::cout << "Global Feature Database size: " << feats_db.size() << std::endl;

	for (auto it : feats_db) {
		EntryId feat_entry_id = it.first;
//    std::cout << "DB - " << feat_entry_id << ": " << it.second << std::endl;

		if ((int) feat_entry_id <= max_id || max_id == -1) {
			// Compute squared Euclidean distance between NetVLAD features
			double dist = cv::norm(global_feat, it.second);
			double ss = cv::exp(-dist);

			pit = pairs.lower_bound(feat_entry_id);
			pairs.insert(pit,
					std::map<EntryId, double>::value_type(feat_entry_id, ss));
		}
	}

	// move to vector
	ret.reserve(pairs.size());

	for (pit = pairs.begin(); pit != pairs.end(); ++pit) {
		ret.push_back(Result(pit->first, pit->second));
	}
	// sort vector in descending order of score
	std::sort(ret.rbegin(), ret.rend());

	// cut vector from 0 to max_results
	if (max_results > 0 && (int) ret.size() > max_results)
		ret.resize(max_results);
}

// ---------------------------------------------------------------------------

void GlobalFeatDatabase::queryHamming(const cv::Mat& global_feat,
		QueryResults &ret, int max_results, int max_id) const {

	// Find the max dist
	double max_dist = std::max(global_feat.rows, global_feat.cols) * 8;

	std::map<EntryId, double> pairs;
	std::map<EntryId, double>::iterator pit;

	if (feats_db.empty() || feats_db.size() == 0) {
		std::cout << "Global Feature Database is empty!" << std::endl;
		return;
	}

	for (auto it : feats_db) {
		EntryId feat_entry_id = it.first;

		if ((int) feat_entry_id <= max_id || max_id == -1) {
			// Compute squared Euclidean distance between NetVLAD features
			int dist = DescriptorDistanceFAST(global_feat, it.second);
			double ss = 1 - dist / max_dist;

//			std::cout << feat_entry_id << ": " << dist << " -> " << ss
//					<< std::endl;

			pit = pairs.lower_bound(feat_entry_id);
			pairs.insert(pit,
					std::map<EntryId, double>::value_type(feat_entry_id, dist));

		}
	}

	// move to vector
	ret.reserve(pairs.size());

	for (pit = pairs.begin(); pit != pairs.end(); ++pit) {
		ret.push_back(Result(pit->first, pit->second));
	}

	// sort vector in ascending order of score
	std::sort(ret.rbegin(), ret.rend());

	// cut vector
	if (max_results > 0 && (int) ret.size() > max_results)
		ret.resize(max_results);
}
