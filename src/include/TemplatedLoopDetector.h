/**
 * File: TemplatedLoopDetector
 * Date: March 2011
 * Author: Dorian Galvez-Lopez
 * Description: templated loop detector
 * License: see the LICENSE.txt file
 *
 */

#ifndef __D_T_TEMPLATED_LOOP_DETECTOR__
#define __D_T_TEMPLATED_LOOP_DETECTOR__

#include <vector>
#include <numeric>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>

#include <DBoW2/TemplatedVocabulary.h>
#include <DBoW2/TemplatedDatabase.h>
#include <DBoW2/QueryResults.h>
#include <DBoW2/BowVector.h>

#include <DUtils/DUtils.h>
#include <DUtilsCV/DUtilsCV.h>
#include <DVision/DVision.h>

#include <messages.h>
#include <utilities.h>
#include <GlobalFeatDB.h>

using namespace std;
using namespace DUtils;
using namespace DBoW2;

namespace DLoopDetector {

struct PlaceFound {
	int query_agent_id;
	int train_agent_id;
	int query_frame_id;
	int query_place_id;
	int place_found_id;
	std::vector<cv::Point2f> query_pts;
	std::vector<cv::Point2f> train_pts;
	std::vector<uchar> inliers;
	std::vector<double> distances;
};

/// Geometrical checking methods
enum GeometricalCheck {
	/// Exhaustive search
	GEOM_EXHAUSTIVE,
	/// Use direct index
	GEOM_DI,
	/// Use a Flann structure
	GEOM_FLANN,
	/// Do not perform geometrical checking
	GEOM_NONE
};

/// Reasons for dismissing loops
enum DetectionStatus {
	/// Loop correctly detected
	LOOP_DETECTED,
	/// All the matches are very recent
	CLOSE_MATCHES_ONLY,
	/// No matches against the database
	NO_DB_RESULTS,
	/// Score of current image against previous one too low
	LOW_NSS_FACTOR,
	/// Scores (or NS Scores) were below the alpha threshold
	LOW_SCORES,
	/// Not enough matches to create groups
	NO_GROUPS,
	/// Not enough temporary consistent matches (k)
	NO_TEMPORAL_CONSISTENCY,
	/// The geometrical consistency failed
	NO_GEOMETRICAL_CONSISTENCY,
	/// Loop correctly detected
	PLACE_DETECTED
};

/// Result of a detection
struct DetectionResult {
	/// Detection status. LOOP_DETECTED iff loop detected
	DetectionStatus status;
	/// Query id
	EntryId query;
	/// Matched id if loop detected, otherwise, best candidate
	EntryId match;

	// Number of putative matches
	int num_matches;

	// Number of verified matches (inliers)
	int num_inliers;

	/**
	 * Checks if the loop was detected
	 * @return true iff a loop was detected
	 */
	inline bool detection() const {
		return status == LOOP_DETECTED || status == PLACE_DETECTED;
	}
};

/// TDescriptor: class of descriptor
/// F: class of descriptor functions
template<class TDescriptor, class F>
/// Generic Loop detector
class TemplatedLoopDetector {
public:

	/// Parameters to create a loop detector
	struct Parameters {
		/// Height of expected images
		int image_rows;
		/// Width of expected images
		int image_cols;

		// Main loop detector parameters

		/// Use normalized similarity score?
		bool use_nss;
		/// Alpha threshold
		float alpha;
		/// Min consistent matches to pass the temporal check
		int k;
		/// Geometrical check
		GeometricalCheck geom_check;
		/// If using direct index for geometrical checking, direct index levels
		int di_levels;

		// These are less deciding parameters of the system

		/// Distance between entries to be consider a match
		int dislocal;
		/// Max number of results from db queries to consider
		int max_db_results;
		/// Min raw score between current entry and previous one to consider a match
		float min_nss_factor;
		/// Min number of close matches to consider some of them
		int min_matches_per_group;
		/// Max separation between matches to consider them of the same group
		int max_intragroup_gap;
		/// Max separation between groups of matches to consider them consistent
		int max_distance_between_groups;
		/// Max separation between two queries to consider them consistent
		int max_distance_between_queries;

		// These are for the RANSAC to compute the F

		/// Min number of inliers when computing a fundamental matrix
		int min_Fpoints;
		/// Max number of iterations of RANSAC
		int max_ransac_iterations;
		/// Success probability of RANSAC
		double ransac_probability;
		/// Max reprojection error of fundamental matrices
		double max_reprojection_error;

		// This is to compute correspondences

		/// Max value of the neighbour-ratio of accepted correspondences
		double max_neighbor_ratio;
		// Threshold on the similarity (Hamming) distance (-1: no threshold)
		int matching_th;
		/// Use the second nearest neighbour or distance ratio test
		bool b_snn;

		/**
		 * Creates parameters by default
		 */
		Parameters();

		/**
		 * Creates parameters by default
		 * @param height image height
		 * @param width image width
		 * @param frequency set the value of some parameters according to the
		 *   expected working frequency (Hz)
		 * @param nss use normalized similarity score
		 * @param _alpha alpha parameter
		 * @param _k k parameter (number of temporary consistent matches)
		 * @param geom type of geometrical check
		 * @param dilevels direct index levels when geom == GEOM_DI
		 */
		Parameters(int height, int width, float frequency = 1, bool nss = true,
				float _alpha = 0.3, int _k = 3, GeometricalCheck geom = GEOM_DI,
				int dilevels = 0);

	private:
		/**
		 * Sets some parameters according to the frequency
		 * @param frequency
		 */
		void set(float frequency);
	};

public:

	/**
	 * Empty constructor
	 */
	TemplatedLoopDetector(const Parameters &params = Parameters());

	/**
	 * Creates a loop detector with the given parameters and with a BoW2 database
	 * with the given vocabulary
	 * @param voc vocabulary
	 * @param params loop detector parameters
	 */
	TemplatedLoopDetector(const TemplatedVocabulary<TDescriptor, F> &voc,
			const Parameters &params = Parameters());

	/**
	 * Creates a loop detector with a copy of the given database, but clearing
	 * its contents
	 * @param db database to copy
	 * @param params loop detector parameters
	 */
	TemplatedLoopDetector(const TemplatedDatabase<TDescriptor, F> &db,
			const Parameters &params = Parameters());

	/**
	 *
	 */
	TemplatedLoopDetector(const TemplatedVocabulary<TDescriptor, F> &voc,
			const QueryDist &query_dist, const Parameters &params =
					Parameters());

	/**
	 * Creates a loop detector with a copy of the given database, but clearing
	 * its contents
	 * @param db database to copy
	 * @param params loop detector parameters
	 */
	TemplatedLoopDetector(const TemplatedDatabase<TDescriptor, F> &db,
			const QueryDist &query_dist, const Parameters &params =
					Parameters());

	/**
	 * Creates a loop detector with a copy of the given database, but clearing
	 * its contents
	 * @param T class derived from TemplatedDatabase
	 * @param db database to copy
	 * @param params loop detector parameters
	 */
	template<class T>
	TemplatedLoopDetector(const T &db, const Parameters &params = Parameters());

	/**
	 * Destructor
	 */
	virtual ~TemplatedLoopDetector(void);

	/**
	 * Retrieves a reference to the database used by the loop detector
	 * @return const reference to database
	 */
	inline const TemplatedDatabase<TDescriptor, F>& getDatabase() const;

	/**
	 * Retrieves a reference to the vocabulary used by the loop detector
	 * @return const reference to vocabulary
	 */
	inline const TemplatedVocabulary<TDescriptor, F>& getVocabulary() const;

	/**
	 * Sets the database to use. The contents of the database and the detector
	 * entries are cleared
	 * @param T class derived from TemplatedDatabase
	 * @param db database to copy
	 */
	template<class T>
	void setDatabase(const T &db);

	/**
	 * Sets a new DBoW2 database created from the given vocabulary
	 * @param voc vocabulary to copy
	 */
	void setVocabulary(const TemplatedVocabulary<TDescriptor, F>& voc);

	/**
	 *
	 */
	void setParameters(const Parameters &params);

	/**
	 * Allocates some memory for the first entries
	 * @param nentries number of expected entries
	 * @param nkeys number of keypoints per image expected
	 */
	void allocate(int nentries, int nkeys = 0);

	/**
	 * Adds the given tuple <keys, descriptors, current_t> to the database
	 * and returns the match if any
	 * @param keys keypoints of the image
	 * @param descriptors descriptors associated to the given keypoints
	 * @param match (out) match or failing information
	 * @return true iff there was match
	 */
	bool detectLoop(const std::vector<cv::KeyPoint> &keys,
			const std::vector<TDescriptor> &descriptors,
			DetectionResult &match);

	/**
	 * Adds the given tuple <keys, descriptors, current_t> to the database
	 * and returns the match if any
	 * @param keys keypoints of the image
	 * @param descriptors descriptors associated to the given keypoints
	 * @param match (out) match or failing information
	 * @return true iff there was match
	 */
	//  bool detectPlaceFromExternCamera(const std::vector<cv::KeyPoint> &keys,
	//                                   const std::vector<TDescriptor> &descriptors,
	//                                   DetectionResult &match);
	bool detectPlaceFromExternCamera(const Msg& msg_rcv, DetectionResult &match,
			const int& agent_id);

	bool detectPlaceFromExternCameraGlobalDesc(const Msg& msg_rcv,
			DetectionResult &match, const int& agent_id);

	/**
	 * Resets the detector and clears the database, such that the next entry
	 * will be 0 again
	 */
	inline void clear();

	inline void addFrameGlobalFeat(const cv::Mat& global_feat) {
		m_gf_database->add(global_feat);
	}
	;

	void AddBowVectorToDB(const std::vector<cv::KeyPoint> &keys,
			const std::vector<DBoW2::FORB::TDescriptor> &orb_descs);

protected:

	/// Matching island
	struct tIsland {
		/// Island starting entry
		EntryId first;
		/// Island ending entry
		EntryId last;
		/// Island score
		double score;  // score of island

		/// Entry of the island with the highest score
		EntryId best_entry;  // id and score of the entry with the highest score
		/// Highest single score in the island
		double best_score;  // in the island

		/**
		 * Creates an empty island
		 */
		tIsland() {
		}

		/**
		 * Creates an island
		 * @param f first entry
		 * @param l last entry
		 */
		tIsland(EntryId f, EntryId l) :
				first(f), last(l) {
		}

		/**
		 * Creates an island
		 * @param f first entry
		 * @param l last entry
		 * @param s island score
		 */
		tIsland(EntryId f, EntryId l, double s) :
				first(f), last(l), score(s) {
		}

		/**
		 * Says whether this score is less than the score of another island
		 * @param b
		 * @return true iff this score < b.score
		 */
		inline bool operator <(const tIsland &b) const {
			return this->score < b.score;
		}

		/**
		 * Says whether this score is greater than the score of another island
		 * @param b
		 * @return true iff this score > b.score
		 */
		inline bool operator >(const tIsland &b) const {
			return this->score > b.score;
		}

		/**
		 * Returns true iff a > b
		 * This function is used to sort in descending order
		 * @param a
		 * @param b
		 * @return a > b
		 */
		static inline bool gt(const tIsland &a, const tIsland &b) {
			return a.score > b.score;
		}

		/**
		 * Returns true iff entry ids of a are less then those of b.
		 * Assumes there is no overlap between the islands
		 * @param a
		 * @param b
		 * @return a.first < b.first
		 */
		static inline bool ltId(const tIsland &a, const tIsland &b) {
			return a.first < b.first;
		}

		/**
		 * Returns the length of the island
		 * @return length of island
		 */
		inline int length() const {
			return last - first + 1;
		}

		/**
		 * Returns a printable version of the island
		 * @return printable island
		 */
		std::string toString() const {
			stringstream ss;
			ss << "[" << first << "-" << last << ": " << score << " | best: <"
					<< best_entry << ": " << best_score << "> ] ";
			return ss.str();
		}
	};

	/// Temporal consistency window
	struct tTemporalWindow {
		/// Island matched in the last query
		tIsland last_matched_island;
		/// Last query id
		EntryId last_query_id;
		/// Number of consistent entries in the window
		int nentries;

		/**
		 * Creates an empty temporal window
		 */
		tTemporalWindow() :
				nentries(0) {
		}
	};

protected:
	/**
	 * Print on screen the results of the islands as frame intervals, island score,
	 * best score within the island, frame ID of the best score in the island.
	 * @param islands vector with the computed islands
	 */
	void PrintIslandsResults(const std::vector<tIsland>& islands) const;

	/**
	 * Print on screen the results of the query as frame ID and corresponding score.
	 * @param qret results from query
	 */
	void PrintCandidateResults(const QueryResults& qret) const;

	/**
	 * Removes from q those results whose score is lower than threshold
	 * (that should be alpha * ns_factor)
	 * @param q results from query
	 * @param threshold min value of the resting results
	 */
	void removeLowScores(QueryResults &q, double threshold) const;

	/**
	 * Returns the islands of the given matches in ascending order of entry ids
	 * @param q
	 * @param islands (out) computed islands
	 */
	void computeIslands(QueryResults &q, vector<tIsland> &islands) const;

	/**
	 * Returns the score of the island composed of the entries of q whose indices
	 * are in [i_first, i_last] (both included)
	 * @param q
	 * @param i_first first index of q of the island
	 * @param i_last last index of q of the island
	 * @return island score
	 */
	double calculateIslandScore(const QueryResults &q, unsigned int i_first,
			unsigned int i_last) const;

	/**
	 * Updates the temporal window by adding the given match <island, id>, such
	 * that the window will contain only those islands which are consistent
	 * @param matched_island
	 * @param entry_id
	 */
	void updateTemporalWindow(const tIsland &matched_island, EntryId entry_id);

	/**
	 * Returns the number of consistent islands in the temporal window
	 * @return number of temporal consistent islands
	 */
	inline int getConsistentEntries() const {
		return m_window.nentries;
	}

	/**
	 * Check if an old entry is geometrically consistent (by calculating a
	 * fundamental matrix) with the given set of keys and descriptors
	 * @param old_entry entry id of the stored image to check
	 * @param keys current keypoints
	 * @param descriptors current descriptors associated to the given keypoints
	 * @param curvec feature vector of the current entry
	 */
	bool isGeometricallyConsistent_DI(EntryId old_entry,
			const std::vector<cv::KeyPoint> &keys,
			const std::vector<TDescriptor> &descriptors,
			const FeatureVector &curvec) const;

	// Overloaded
	bool isGeometricallyConsistent_DI(EntryId old_entry, const Msg& msg_rcv,
			const std::vector<TDescriptor> &descriptors,
			const FeatureVector &bowvec, const int& agent_id,
			int& num_matches, int& num_inliers) const;

	/**
	 * Checks if an old entry is geometrically consistent (by using FLANN and
	 * computing an essential matrix by using the neighbour ratio 0.6)
	 * with the given set of keys and descriptors
	 * @param old_entry entry id of the stored image to check
	 * @param keys current keypoints
	 * @param descriptors current descriptors
	 * @param flann_structure flann structure with the descriptors of the current entry
	 */
	bool isGeometricallyConsistent_Flann(EntryId old_entry,
			const std::vector<cv::KeyPoint> &keys,
			const std::vector<TDescriptor> &descriptors,
			cv::FlannBasedMatcher &flann_structure) const;

	/**
	 * Creates a flann structure from a set of descriptors to perform queries
	 * @param descriptors
	 * @param flann_structure (out) flann matcher
	 */
	void getFlannStructure(const std::vector<TDescriptor> &descriptors,
			cv::FlannBasedMatcher &flann_structure) const;

	/**
	 * Check if an old entry is geometrically consistent (by calculating a
	 * fundamental matrix from left-right correspondences) with the given set
	 * of keys and descriptors,
	 * without using the direct index
	 * @param old_keys keys of old entry
	 * @param old_descriptors descriptors of old keys
	 * @param cur_keys keys of current entry
	 * @param cur_descriptors descriptors of cur keys
	 */
	bool isGeometricallyConsistent_Exhaustive(
			const std::vector<cv::KeyPoint> &old_keys,
			const std::vector<TDescriptor> &old_descriptors,
			const std::vector<cv::KeyPoint> &cur_keys,
			const std::vector<TDescriptor> &cur_descriptors) const;

	/**
	 * Calculate the matches between the descriptors A[i_A] and the descriptors
	 * B[i_B]. Applies a left-right matching without neighbour ratio
	 * @param A set A of descriptors
	 * @param i_A only descriptors A[i_A] will be checked
	 * @param B set B of descriptors
	 * @param i_B only descriptors B[i_B] will be checked
	 * @param i_match_A (out) indices of descriptors matched (s.t. A[i_match_A])
	 * @param i_match_B (out) indices of descriptors matched (s.t. B[i_match_B])
	 */
	void getMatches_neighratio(const std::vector<TDescriptor> &A,
			const vector<unsigned int> &i_A, const vector<TDescriptor> &B,
			const vector<unsigned int> &i_B, vector<unsigned int> &i_match_A,
			vector<unsigned int> &i_match_B) const;

	/**
	 * Calculate the matches between the descriptors A[i_A] and the descriptors
	 * B[i_B]. Applies a left-right matching without neighbour ratio
	 * @param A set A of descriptors
	 * @param i_A only descriptors A[i_A] will be checked
	 * @param B set B of descriptors
	 * @param i_B only descriptors B[i_B] will be checked
	 * @param i_match_A (out) indices of descriptors matched (s.t. A[i_match_A])
	 * @param i_match_B (out) indices of descriptors matched (s.t. B[i_match_B])
	 * @param i_match_d (out) distances of the matched descriptors
	 */
	void getMatches_neighratio(const vector<TDescriptor> &A,
			const vector<unsigned int> &i_A, const vector<TDescriptor> &B,
			const vector<unsigned int> &i_B, vector<unsigned int> &i_match_A,
			vector<unsigned int> &i_match_B, vector<double> &D) const;

	/**
	 * Calculate the matches between the descriptors A[i_A] and the descriptors
	 * B[i_B]. Applies a left-right matching without neighbour ratio
	 * @param A set A of descriptors
	 * @param i_A only descriptors A[i_A] will be checked
	 * @param B set B of descriptors
	 * @param i_B only descriptors B[i_B] will be checked
	 * @param i_match_A (out) indices of descriptors matched (s.t. A[i_match_A])
	 * @param i_match_B (out) indices of descriptors matched (s.t. B[i_match_B])
	 */
	void getMatches(const std::vector<TDescriptor> &A,
			const vector<unsigned int> &i_A, const vector<TDescriptor> &B,
			const vector<unsigned int> &i_B, vector<unsigned int> &i_match_A,
			vector<unsigned int> &i_match_B) const;

	/**
	 *
	 */
	void SaveMatchedPointsOfPlace(PlaceFound pt) const;

protected:

	/// Database
	// The loop detector stores its own copy of the database
	TemplatedDatabase<TDescriptor, F> *m_database;

	/// KeyPoints of images
	vector<vector<cv::KeyPoint> > m_image_keys;

	/// Descriptors of images
	vector<vector<TDescriptor> > m_image_descriptors;

	/// Last bow vector added to database
	BowVector m_last_bowvec;

	/// Temporal consistency window
	tTemporalWindow m_window;

	/// Parameters of loop detector
	Parameters m_params;

	/// To compute the fundamental matrix
	DVision::FSolver m_fsolver;

	/// Database of the global features (NetVLAD or DeepBit)
	GlobalFeatDatabase *m_gf_database;
};

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TemplatedLoopDetector<TDescriptor, F>::Parameters::Parameters() :
		use_nss(true), alpha(0.3), k(4), geom_check(GEOM_DI), di_levels(0) {
	set(1);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TemplatedLoopDetector<TDescriptor, F>::Parameters::Parameters(int height,
		int width, float frequency, bool nss, float _alpha, int _k,
		GeometricalCheck geom, int dilevels) :
		image_rows(height), image_cols(width), use_nss(nss), alpha(_alpha), k(
				_k), geom_check(geom), di_levels(dilevels) {
	set(frequency);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedLoopDetector<TDescriptor, F>::Parameters::set(float f) {
	dislocal = 20 * f;
	max_db_results = 50 * f;
	min_nss_factor = 0.005;
	min_matches_per_group = f;
	max_intragroup_gap = 3 * f;
	max_distance_between_groups = 3 * f;
	max_distance_between_queries = 2 * f;

	min_Fpoints = 12;
	max_ransac_iterations = 500;
	ransac_probability = 0.99;
	max_reprojection_error = 2.0;

	max_neighbor_ratio = 0.6;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TemplatedLoopDetector<TDescriptor, F>::TemplatedLoopDetector(
		const Parameters &params) :
		m_database(NULL), m_gf_database(NULL), m_params(params) {
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TemplatedLoopDetector<TDescriptor, F>::TemplatedLoopDetector(
		const TemplatedVocabulary<TDescriptor, F> &voc,
		const Parameters &params) :
		m_params(params), m_gf_database(NULL) {
	m_database = new TemplatedDatabase<TDescriptor, F>(voc,
			params.geom_check == GEOM_DI, params.di_levels);

	m_fsolver.setImageSize(params.image_cols, params.image_rows);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TemplatedLoopDetector<TDescriptor, F>::TemplatedLoopDetector(
		const TemplatedVocabulary<TDescriptor, F> &voc,
		const QueryDist &query_dist, const Parameters &params) :
		m_params(params) {
	m_database = new TemplatedDatabase<TDescriptor, F>(voc,
			params.geom_check == GEOM_DI, params.di_levels);

	m_gf_database = new GlobalFeatDatabase(query_dist);

	m_fsolver.setImageSize(params.image_cols, params.image_rows);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedLoopDetector<TDescriptor, F>::setVocabulary(
		const TemplatedVocabulary<TDescriptor, F>& voc) {
	delete m_database;
	m_database = new TemplatedDatabase<TDescriptor, F>(voc,
			m_params.geom_check == GEOM_DI, m_params.di_levels);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedLoopDetector<TDescriptor, F>::setParameters(
		const Parameters &params) {
	m_params = params;

	m_fsolver.setImageSize(params.image_cols, params.image_rows);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TemplatedLoopDetector<TDescriptor, F>::TemplatedLoopDetector(
		const TemplatedDatabase<TDescriptor, F> &db, const Parameters &params) :
		m_params(params), m_gf_database(NULL) {
	m_database = new TemplatedDatabase<TDescriptor, F>(db.getVocabulary(),
			params.geom_check == GEOM_DI, params.di_levels);

	m_fsolver.setImageSize(params.image_cols, params.image_rows);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TemplatedLoopDetector<TDescriptor, F>::TemplatedLoopDetector(
		const TemplatedDatabase<TDescriptor, F> &db,
		const QueryDist &query_dist, const Parameters &params) :
		m_params(params) {
	m_database = new TemplatedDatabase<TDescriptor, F>(db.getVocabulary(),
			params.geom_check == GEOM_DI, params.di_levels);

	m_gf_database = new GlobalFeatDatabase(query_dist);

	m_fsolver.setImageSize(params.image_cols, params.image_rows);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
template<class T>
TemplatedLoopDetector<TDescriptor, F>::TemplatedLoopDetector(const T &db,
		const Parameters &params) :
		m_params(params), m_gf_database(NULL) {
	m_database = new T(db);
	m_database->clear();

	m_fsolver.setImageSize(params.image_cols, params.image_rows);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
template<class T>
void TemplatedLoopDetector<TDescriptor, F>::setDatabase(const T &db) {
	delete m_database;
	m_database = new T(db);
	clear();
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TemplatedLoopDetector<TDescriptor, F>::~TemplatedLoopDetector(void) {
	delete m_database;
	m_database = NULL;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedLoopDetector<TDescriptor, F>::allocate(int nentries, int nkeys) {
	const int sz = (const int) m_image_keys.size();

	if (sz < nentries) {
		m_image_keys.resize(nentries);
		m_image_descriptors.resize(nentries);
	}

	if (nkeys > 0) {
		for (int i = sz; i < nentries; ++i) {
			m_image_keys[i].reserve(nkeys);
			m_image_descriptors[i].reserve(nkeys);
		}
	}

	m_database->allocate(nentries, nkeys);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
inline const TemplatedDatabase<TDescriptor, F>&
TemplatedLoopDetector<TDescriptor, F>::getDatabase() const {
	return *m_database;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
inline const TemplatedVocabulary<TDescriptor, F>&
TemplatedLoopDetector<TDescriptor, F>::getVocabulary() const {
	return m_database->getVocabulary();
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
bool TemplatedLoopDetector<TDescriptor, F>::detectLoop(
		const std::vector<cv::KeyPoint> &keys,
		const std::vector<TDescriptor> &descriptors, DetectionResult &match) {
	EntryId entry_id = m_database->size();
	match.query = entry_id;

	BowVector bowvec;
	FeatureVector featvec;

	match.match = -1;

	if (m_params.geom_check == GEOM_DI)
		m_database->getVocabulary()->transform(descriptors, bowvec, featvec,
				m_params.di_levels);
	else
		m_database->getVocabulary()->transform(descriptors, bowvec);

	if ((int) entry_id <= m_params.dislocal) {
		// only add the entry to the database and finish
		m_database->add(bowvec, featvec);
		match.status = CLOSE_MATCHES_ONLY;
	} else {
		int max_id = (int) entry_id - m_params.dislocal;

		QueryResults qret;
		m_database->query(bowvec, qret, m_params.max_db_results, max_id);

		// update database
		m_database->add(bowvec, featvec);  // returns entry_id

		if (!qret.empty()) {
			// factor to compute normalized similarity score, if necessary
			double ns_factor = 1.0;

			if (m_params.use_nss) {
				ns_factor = m_database->getVocabulary()->score(bowvec,
						m_last_bowvec);
			}

			if (!m_params.use_nss || ns_factor >= m_params.min_nss_factor) {
				// scores in qret must be divided by ns_factor to obtain the
				// normalized similarity score, but we can
				// speed this up by moving ns_factor to alpha's

				// remove those scores whose nss is lower than alpha
				// (ret is sorted in descending score order now)
				removeLowScores(qret, m_params.alpha * ns_factor);

				if (!qret.empty()) {
					// the best candidate is the one with highest score by now
					match.match = qret[0].Id;

					// compute islands
					vector<tIsland> islands;
					computeIslands(qret, islands);
					// this modifies qret and changes the score order

					// get best island
					if (!islands.empty()) {
						const tIsland& island = *std::max_element(
								islands.begin(), islands.end());

						// check temporal consistency of this island
						updateTemporalWindow(island, entry_id);

						// get the best candidate (maybe match)
						match.match = island.best_entry;

						if (getConsistentEntries() > m_params.k) {
							// candidate loop detected
							// check geometry
							bool detection;

							if (m_params.geom_check == GEOM_DI) {
								// all the DI stuff is implicit in the database
								detection = isGeometricallyConsistent_DI(
										island.best_entry, keys, descriptors,
										featvec);
							} else if (m_params.geom_check == GEOM_FLANN) {
								cv::FlannBasedMatcher flann_structure;
								getFlannStructure(descriptors, flann_structure);

								detection = isGeometricallyConsistent_Flann(
										island.best_entry, keys, descriptors,
										flann_structure);
							} else if (m_params.geom_check == GEOM_EXHAUSTIVE) {
								detection =
										isGeometricallyConsistent_Exhaustive(
												m_image_keys[island.best_entry],
												m_image_descriptors[island.best_entry],
												keys, descriptors);
							} else  // GEOM_NONE, accept the match
							{
								detection = true;
							}

							if (detection) {
								match.status = LOOP_DETECTED;
							} else {
								match.status = NO_GEOMETRICAL_CONSISTENCY;
							}

						}  // if enough temporal matches
						else {
							match.status = NO_TEMPORAL_CONSISTENCY;
						}

					}  // if there is some island
					else {
						match.status = NO_GROUPS;
					}
				}  // if !qret empty after removing low scores
				else {
					match.status = LOW_SCORES;
				}
			}  // if (ns_factor > min normal score)
			else {
				match.status = LOW_NSS_FACTOR;
			}
		}  // if(!qret.empty())
		else {
			match.status = NO_DB_RESULTS;
		}
	}

	// update record
	// m_image_keys and m_image_descriptors have the same length
	if (m_image_keys.size() == entry_id) {
		m_image_keys.push_back(keys);
		m_image_descriptors.push_back(descriptors);
	} else {
		m_image_keys[entry_id] = keys;
		m_image_descriptors[entry_id] = descriptors;
	}

	// store this bowvec if we are going to use it in next iteratons
	if (m_params.use_nss && (int) entry_id + 1 > m_params.dislocal) {
		m_last_bowvec = bowvec;
	}

	return match.detection();
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
bool TemplatedLoopDetector<TDescriptor, F>::detectPlaceFromExternCamera(
		const Msg& msg_rcv, DetectionResult &match, const int& agent_id) {
	PrintHeading2(" Detect Overlap From External Camera ");


	std::vector<TDescriptor> descriptors;
	ConvertMatDescToTDescriptorVec(msg_rcv.f.descs, descriptors);

	EntryId entry_id = m_database->size();
	match.query = entry_id;

	int num_matches = -1;
	int num_inliers = -1;
	match.num_matches = num_matches;
	match.num_inliers = num_inliers;

	match.match = -1;
	match.status = NO_DB_RESULTS;

	BowVector bowvec;
	FeatureVector featvec;

	if (m_params.geom_check == GEOM_DI)
		m_database->getVocabulary()->transform(descriptors, bowvec, featvec,
				m_params.di_levels);
	else
		m_database->getVocabulary()->transform(descriptors, bowvec);

//	if ((int) entry_id <= m_params.dislocal) {
//		match.status = CLOSE_MATCHES_ONLY;
//	} else {
		//		int max_id = (int) entry_id - m_params.dislocal;

		QueryResults qret;
		m_database->query(bowvec, qret, m_params.max_db_results,
				(int) entry_id);
		saveQueryResultsToFile(qret, agent_id, entry_id, "queryresults");

		if (!qret.empty()) {
			// factor to compute normalized similarity score, if necessary
			double ns_factor = 1.0;

			if (m_params.use_nss) {
				ns_factor = m_database->getVocabulary()->score(bowvec,
						m_last_bowvec);
			}

			if (!m_params.use_nss || ns_factor >= m_params.min_nss_factor) {
				// scores in qret must be divided by ns_factor to obtain the
				// normalized similarity score, but we can
				// speed this up by moving ns_factor to alpha's

				// remove those scores whose nss is lower than alpha
				// (ret is sorted in descending score order now)
				removeLowScores(qret, m_params.alpha * ns_factor);

				if (!qret.empty()) {
					// the best candidate is the one with highest score by now
					match.match = qret[0].Id;

					// Print qret results
					//PrintCandidateResults(qret);

					// compute islands
					vector<tIsland> islands;
					computeIslands(qret, islands);
					// this modifies qret and changes the score order

					saveQueryResultsToFile(qret, agent_id, entry_id, "queryresults_islands");

					//PrintCandidateResults(qret);
					//PrintIslandResults(islands);

					// get best island
					if (!islands.empty()) {
						const tIsland& island = *std::max_element(
								islands.begin(), islands.end());

						// get the best candidate (maybe match)
						match.match = island.best_entry;

						bool detection;

						if (m_params.geom_check == GEOM_DI) {
							// all the DI stuff is implicit in the database
							detection = isGeometricallyConsistent_DI(
									island.best_entry, msg_rcv, descriptors,
									featvec, agent_id, num_matches, num_inliers);

							match.num_matches = num_matches;
							match.num_inliers = num_inliers;
						} else if (m_params.geom_check == GEOM_FLANN) {
							cv::FlannBasedMatcher flann_structure;
							getFlannStructure(descriptors, flann_structure);

							detection = isGeometricallyConsistent_Flann(
									island.best_entry, msg_rcv.f.kps,
									descriptors, flann_structure);
						} else if (m_params.geom_check == GEOM_EXHAUSTIVE) {
							detection = isGeometricallyConsistent_Exhaustive(
									m_image_keys[island.best_entry],
									m_image_descriptors[island.best_entry],
									msg_rcv.f.kps, descriptors);
						} else  // GEOM_NONE, accept the match
						{
							detection = true;
						}

						if (detection) {
							match.status = PLACE_DETECTED;
						} else {
							match.status = NO_GEOMETRICAL_CONSISTENCY;
						}
					}  // if there is some island
					else {
						match.status = NO_GROUPS;
					}
				}  // if !qret empty after removing low scores
				else {
					match.status = LOW_SCORES;
				}
			}  // if (ns_factor > min normal score)
			else {
				match.status = LOW_NSS_FACTOR;
			}
		}  // if(!qret.empty())
		else {
			match.status = NO_DB_RESULTS;
		}
//	}

	return match.detection();
}

template<class TDescriptor, class F>
bool TemplatedLoopDetector<TDescriptor, F>::detectPlaceFromExternCameraGlobalDesc(
		const Msg& msg_rcv, DetectionResult &match, const int& agent_id) {
	PrintHeading2(" detectPlaceFromExternCameraGlobalDesc ");

	std::vector<TDescriptor> descriptors;
	ConvertMatDescToTDescriptorVec(msg_rcv.f.descs, descriptors);

	EntryId entry_id = m_gf_database->size();
	match.query = entry_id;

	int num_matches = -1;
	int num_inliers = -1;
	match.num_matches = num_matches;
	match.num_inliers = num_inliers;

	match.match = -1;
	match.status = NO_DB_RESULTS;

	cv::Mat global_desc = msg_rcv.f.global_desc.clone();

	BowVector bowvec;
	FeatureVector featvec;

	if (m_params.geom_check == GEOM_DI)
		m_database->getVocabulary()->transform(descriptors, bowvec, featvec,
				m_params.di_levels);
	else
		m_database->getVocabulary()->transform(descriptors, bowvec);

//	if ((int) entry_id <= m_params.dislocal) {
//		match.status = CLOSE_MATCHES_ONLY;
//	} else {
		// int max_id = (int) entry_id - m_params.dislocal;

		QueryResults qret;

//		if () {
//			m_database->query(bowvec, qret, m_params.max_db_results,
//					(int) entry_id);
//		} else {
		//Create query functions for DeepBit/NetVLAD global features
		m_gf_database->query(global_desc, qret, m_params.max_db_results,
				(int) entry_id);
//		}
		saveQueryResultsToFile(qret, agent_id, entry_id, "queryresults");

		if (!qret.empty()) {
			// factor to compute normalized similarity score, if necessary
			double ns_factor = 1.0;

			if (m_params.use_nss) {
				ns_factor = m_database->getVocabulary()->score(bowvec,
						m_last_bowvec);
			}

			if (!m_params.use_nss || ns_factor >= m_params.min_nss_factor) {
				// scores in qret must be divided by ns_factor to obtain the
				// normalized similarity score, but we can
				// speed this up by moving ns_factor to alpha's

				// remove those scores whose nss is lower than alpha
				// (ret is sorted in descending score order now)
				//        std::cout << m_params.alpha << ", " << ns_factor << std::endl;
				//        removeLowScores(qret, m_params.alpha * ns_factor);
				//        std::cout << "Number of query results: " << (int) qret.size() << std::endl;

				if (!qret.empty()) {
					// the best candidate is the one with highest score by now
					match.match = qret[0].Id;

					// compute islands
					vector<tIsland> islands;
					computeIslands(qret, islands);
					// this modifies qret and changes the score order
					saveQueryResultsToFile(qret, agent_id, entry_id, "queryresults_islands");

					// get best island
					if (!islands.empty()) {
						const tIsland& island = *std::max_element(
								islands.begin(), islands.end());

						// check temporal consistency of this island
						//updateTemporalWindow(island, entry_id);

						// get the best candidate (maybe match)
						match.match = island.best_entry;

						//if (getConsistentEntries() > m_params.k) {
						// candidate loop detected
						// check geometry
						bool detection;

						if (m_params.geom_check == GEOM_DI) {
							// all the DI stuff is implicit in the database
							detection = isGeometricallyConsistent_DI(
									island.best_entry, msg_rcv, descriptors,
									featvec, agent_id, num_matches, num_inliers);

							match.num_matches = num_matches;
							match.num_inliers = num_inliers;
						} else if (m_params.geom_check == GEOM_FLANN) {
							cv::FlannBasedMatcher flann_structure;
							getFlannStructure(descriptors, flann_structure);

							detection = isGeometricallyConsistent_Flann(
									island.best_entry, msg_rcv.f.kps,
									descriptors, flann_structure);
						} else if (m_params.geom_check == GEOM_EXHAUSTIVE) {
							detection = isGeometricallyConsistent_Exhaustive(
									m_image_keys[island.best_entry],
									m_image_descriptors[island.best_entry],
									msg_rcv.f.kps, descriptors);
						} else  // GEOM_NONE, accept the match
						{
							detection = true;
						}

						if (detection) {
							match.status = PLACE_DETECTED;
						} else {
							match.status = NO_GEOMETRICAL_CONSISTENCY;
						}

						//            }  // if enough temporal matches
						//            else {
						//              match.status = NO_TEMPORAL_CONSISTENCY;
						//            }

					}  // if there is some island
					else {
						match.status = NO_GROUPS;
					}
				}  // if !qret empty after removing low scores
				else {
					match.status = LOW_SCORES;
				}
			}  // if (ns_factor > min normal score)
			else {
				match.status = LOW_NSS_FACTOR;
			}
		}  // if(!qret.empty())
		else {
			match.status = NO_DB_RESULTS;
		}
//	}

	return match.detection();
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
inline void TemplatedLoopDetector<TDescriptor, F>::clear() {
	m_database->clear();
	m_window.nentries = 0;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedLoopDetector<TDescriptor, F>::computeIslands(QueryResults &q,
		vector<tIsland> &islands) const {
	islands.clear();

	if (q.size() == 1) {
		islands.push_back(
				tIsland(q[0].Id, q[0].Id, calculateIslandScore(q, 0, 0)));
		islands.back().best_entry = q[0].Id;
		islands.back().best_score = q[0].Score;
	} else if (!q.empty()) {
		// sort query results in ascending order of ids
		std::sort(q.begin(), q.end(), Result::ltId);

		// create long enough islands
		QueryResults::const_iterator dit = q.begin();
		int first_island_entry = dit->Id;
		int last_island_entry = dit->Id;

		// these are indices of q
		unsigned int i_first = 0;
		unsigned int i_last = 0;

		double best_score = dit->Score;
		EntryId best_entry = dit->Id;

		++dit;
		for (unsigned int idx = 1; dit != q.end(); ++dit, ++idx) {
			if ((int) dit->Id - last_island_entry
					< m_params.max_intragroup_gap) {
				// go on until find the end of the island
				last_island_entry = dit->Id;
				i_last = idx;
				if (dit->Score > best_score) {
					best_score = dit->Score;
					best_entry = dit->Id;
				}
			} else {
				// end of island reached
				int length = last_island_entry - first_island_entry + 1;
				if (length >= m_params.min_matches_per_group) {
					islands.push_back(
							tIsland(first_island_entry, last_island_entry,
									calculateIslandScore(q, i_first, i_last)));

					islands.back().best_score = best_score;
					islands.back().best_entry = best_entry;
				}

				// prepare next island
				first_island_entry = last_island_entry = dit->Id;
				i_first = i_last = idx;
				best_score = dit->Score;
				best_entry = dit->Id;
			}
		}
		// add last island
		if (last_island_entry - first_island_entry + 1
				>= m_params.min_matches_per_group) {
			islands.push_back(
					tIsland(first_island_entry, last_island_entry,
							calculateIslandScore(q, i_first, i_last)));

			islands.back().best_score = best_score;
			islands.back().best_entry = best_entry;
		}
	}

}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
double TemplatedLoopDetector<TDescriptor, F>::calculateIslandScore(
		const QueryResults &q, unsigned int i_first,
		unsigned int i_last) const {
	// get the sum of the scores
	double sum = 0;
	while (i_first <= i_last)
		sum += q[i_first++].Score;
	return sum;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedLoopDetector<TDescriptor, F>::updateTemporalWindow(
		const tIsland &matched_island, EntryId entry_id) {
	// if m_window.nentries > 0, island > m_window.last_matched_island and
	// entry_id > m_window.last_query_id hold

	if (m_window.nentries == 0
			|| int(entry_id - m_window.last_query_id)
					> m_params.max_distance_between_queries) {
		m_window.nentries = 1;
	} else {
		EntryId a1 = m_window.last_matched_island.first;
		EntryId a2 = m_window.last_matched_island.last;
		EntryId b1 = matched_island.first;
		EntryId b2 = matched_island.last;

		bool fit = (b1 <= a1 && a1 <= b2) || (a1 <= b1 && b1 <= a2);

		if (!fit) {
			int d1 = (int) a1 - (int) b2;
			int d2 = (int) b1 - (int) a2;
			int gap = (d1 > d2 ? d1 : d2);

			fit = (gap <= m_params.max_distance_between_groups);
		}

		if (fit)
			m_window.nentries++;
		else
			m_window.nentries = 1;
	}

	m_window.last_matched_island = matched_island;
	m_window.last_query_id = entry_id;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
bool TemplatedLoopDetector<TDescriptor, F>::isGeometricallyConsistent_DI(
		EntryId old_entry, const std::vector<cv::KeyPoint> &keys,
		const std::vector<TDescriptor> &descriptors,
		const FeatureVector &bowvec) const {
	const FeatureVector &oldvec = m_database->retrieveFeatures(old_entry);

	// for each word in common, get the closest descriptors

	vector<unsigned int> i_old, i_cur;

	FeatureVector::const_iterator old_it, cur_it;
	const FeatureVector::const_iterator old_end = oldvec.end();
	const FeatureVector::const_iterator cur_end = bowvec.end();

	old_it = oldvec.begin();
	cur_it = bowvec.begin();

	while (old_it != old_end && cur_it != cur_end) {
		if (old_it->first == cur_it->first) {
			// compute matches between
			// features old_it->second of m_image_keys[old_entry] and
			// features cur_it->second of keys
			vector<unsigned int> i_old_now, i_cur_now;

			      getMatches_neighratio(m_image_descriptors[old_entry], old_it->second,
			                            descriptors, cur_it->second, i_old_now, i_cur_now);
			//getMatches(m_image_descriptors[old_entry], old_it->second,
			//		descriptors, cur_it->second, i_old_now, i_cur_now);

			i_old.insert(i_old.end(), i_old_now.begin(), i_old_now.end());
			i_cur.insert(i_cur.end(), i_cur_now.begin(), i_cur_now.end());

			// move old_it and cur_it forward
			++old_it;
			++cur_it;
		} else if (old_it->first < cur_it->first) {
			// move old_it forward
			old_it = oldvec.lower_bound(cur_it->first);
			// old_it = (first element >= cur_it.id)
		} else {
			// move cur_it forward
			cur_it = bowvec.lower_bound(old_it->first);
			// cur_it = (first element >= old_it.id)
		}
	}

	// calculate now the fundamental matrix
	if ((int) i_old.size() >= m_params.min_Fpoints) {
		vector<cv::Point2f> old_points, cur_points;

		// add matches to the vectors to calculate the fundamental matrix
		vector<unsigned int>::const_iterator oit, cit;
		oit = i_old.begin();
		cit = i_cur.begin();

		for (; oit != i_old.end(); ++oit, ++cit) {
			const cv::KeyPoint &old_k = m_image_keys[old_entry][*oit];
			const cv::KeyPoint &cur_k = keys[*cit];

			old_points.push_back(old_k.pt);
			cur_points.push_back(cur_k.pt);
		}

		cv::Mat oldMat(old_points.size(), 2, CV_32F, &old_points[0]);
		cv::Mat curMat(cur_points.size(), 2, CV_32F, &cur_points[0]);

		//		return m_fsolver.checkFundamentalMat(oldMat, curMat,
//				m_params.max_reprojection_error, m_params.min_Fpoints,
//				m_params.ransac_probability, m_params.max_ransac_iterations);

		std::vector<uchar> inliers;

		cv::Mat fundMat = m_fsolver.findFundamentalMat(oldMat, curMat,
				m_params.max_reprojection_error, m_params.min_Fpoints, &inliers,
				true, m_params.ransac_probability,
				m_params.max_ransac_iterations);

//		 std::cout << fundMat << std::endl;

		int num_matches = (int) inliers.size();

		return !fundMat.empty();
	}

	return false;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
bool TemplatedLoopDetector<TDescriptor, F>::isGeometricallyConsistent_DI(
		EntryId old_entry, const Msg& msg_rcv,
		const std::vector<TDescriptor> &descriptors,
		const FeatureVector &bowvec, const int& agent_id,
		int& num_matches, int& num_inliers) const {

	std::cout << "isGeometricallyConsistent_DI" << std::endl;

	const FeatureVector &oldvec = m_database->retrieveFeatures(old_entry);
	num_matches = -1;
	num_inliers = -1;

	// for each word in common, get the closest descriptors
	vector<unsigned int> i_old, i_cur;
	vector<double> distances;

	FeatureVector::const_iterator old_it, cur_it;
	const FeatureVector::const_iterator old_end = oldvec.end();
	const FeatureVector::const_iterator cur_end = bowvec.end();

	old_it = oldvec.begin();
	cur_it = bowvec.begin();

	while (old_it != old_end && cur_it != cur_end) {
		if (old_it->first == cur_it->first) {
			// compute matches between
			// features old_it->second of m_image_keys[old_entry] and
			// features cur_it->second of keys
			vector<unsigned int> i_old_now, i_cur_now;
			vector<double> distances_now;

			if (old_it->second[0] < m_image_descriptors[old_entry].size()) {

				getMatches_neighratio(m_image_descriptors[old_entry],
						old_it->second, descriptors, cur_it->second, i_old_now,
						i_cur_now, distances_now);
				//getMatches(m_image_descriptors[old_entry],
				//						old_it->second, descriptors, cur_it->second, i_old_now,
				//						i_cur_now);

				i_old.insert(i_old.end(), i_old_now.begin(), i_old_now.end());
				i_cur.insert(i_cur.end(), i_cur_now.begin(), i_cur_now.end());
				distances.insert(distances.end(), distances_now.begin(), distances_now.end());
			}
			// move old_it and cur_it forward
			++old_it;
			++cur_it;
		} else if (old_it->first < cur_it->first) {
			// move old_it forward
			old_it = oldvec.lower_bound(cur_it->first);
			// old_it = (first element >= cur_it.id)
		} else {
			// move cur_it forward
			cur_it = bowvec.lower_bound(old_it->first);
			// cur_it = (first element >= old_it.id)
		}
	}

	num_matches = (int) i_old.size();

	std::vector<cv::KeyPoint> keys = msg_rcv.f.kps;

	// calculate now the fundamental matrix
	if ((int) i_old.size() >= m_params.min_Fpoints) {
		vector<cv::Point2f> old_points, cur_points;

		// add matches to the vectors to calculate the fundamental matrix
		vector<unsigned int>::const_iterator oit, cit;
		oit = i_old.begin();
		cit = i_cur.begin();

		for (; oit != i_old.end(); ++oit, ++cit) {
			const cv::KeyPoint &old_k = m_image_keys[old_entry][*oit];
			const cv::KeyPoint &cur_k = keys[*cit];

			old_points.push_back(old_k.pt);
			cur_points.push_back(cur_k.pt);
		}

		// I verified that matched points are the same if using the same input image
		cv::Mat oldMat(old_points.size(), 2, CV_32F, &old_points[0]);
		cv::Mat curMat(cur_points.size(), 2, CV_32F, &cur_points[0]);

		std::vector<uchar> inliers;
//		DVision::FSolver mysolver(1280,720);
		cv::Mat fundMat = m_fsolver.findFundamentalMat(oldMat, curMat,
				m_params.max_reprojection_error, m_params.min_Fpoints, &inliers,
				true, m_params.ransac_probability,
				m_params.max_ransac_iterations);

		num_matches = (int) inliers.size();
		num_inliers = cv::countNonZero(inliers);

		PlaceFound place;
		place.train_agent_id = agent_id;
		place.query_agent_id = msg_rcv.agent_id;
		place.query_frame_id = msg_rcv.f.timestamp;
		place.query_place_id = msg_rcv.f.timestamp;
		place.query_pts = cur_points;
		place.train_pts = old_points;
		place.place_found_id = old_entry;
		place.inliers = inliers;
		place.distances = distances;

		SaveMatchedPointsOfPlace(place);

		return !fundMat.empty();
	}

	return false;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
bool TemplatedLoopDetector<TDescriptor, F>::isGeometricallyConsistent_Exhaustive(
		const std::vector<cv::KeyPoint> &old_keys,
		const std::vector<TDescriptor> &old_descriptors,
		const std::vector<cv::KeyPoint> &cur_keys,
		const std::vector<TDescriptor> &cur_descriptors) const {
	vector<unsigned int> i_old, i_cur;
	vector<unsigned int> i_all_old, i_all_cur;

	i_all_old.reserve(old_keys.size());
	i_all_cur.reserve(cur_keys.size());

	for (unsigned int i = 0; i < old_keys.size(); ++i) {
		i_all_old.push_back(i);
	}

	for (unsigned int i = 0; i < cur_keys.size(); ++i) {
		i_all_cur.push_back(i);
	}

	getMatches_neighratio(old_descriptors, i_all_old, cur_descriptors,
			i_all_cur, i_old, i_cur);

	if ((int) i_old.size() >= m_params.min_Fpoints) {
		// add matches to the vectors to calculate the fundamental matrix
		vector<unsigned int>::const_iterator oit, cit;
		oit = i_old.begin();
		cit = i_cur.begin();

		vector<cv::Point2f> old_points, cur_points;
		old_points.reserve(i_old.size());
		cur_points.reserve(i_cur.size());

		for (; oit != i_old.end(); ++oit, ++cit) {
			const cv::KeyPoint &old_k = old_keys[*oit];
			const cv::KeyPoint &cur_k = cur_keys[*cit];

			old_points.push_back(old_k.pt);
			cur_points.push_back(cur_k.pt);
		}

		cv::Mat oldMat(old_points.size(), 2, CV_32F, &old_points[0]);
		cv::Mat curMat(cur_points.size(), 2, CV_32F, &cur_points[0]);

		return m_fsolver.checkFundamentalMat(oldMat, curMat,
				m_params.max_reprojection_error, m_params.min_Fpoints,
				m_params.ransac_probability, m_params.max_ransac_iterations);
	}

	return false;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedLoopDetector<TDescriptor, F>::getFlannStructure(
		const std::vector<TDescriptor> &descriptors,
		cv::FlannBasedMatcher &flann_structure) const {
	vector<cv::Mat> features(1);
	F::toMat32F(descriptors, features[0]);

	flann_structure.clear();
	flann_structure.add(features);
	flann_structure.train();
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
bool TemplatedLoopDetector<TDescriptor, F>::isGeometricallyConsistent_Flann(
		EntryId old_entry, const std::vector<cv::KeyPoint> &keys,
		const std::vector<TDescriptor> &descriptors,
		cv::FlannBasedMatcher &flann_structure) const {
	vector<unsigned int> i_old, i_cur;  // indices of correspondences

	const vector<cv::KeyPoint>& old_keys = m_image_keys[old_entry];
	const vector<TDescriptor>& old_descs = m_image_descriptors[old_entry];
	const vector<cv::KeyPoint>& cur_keys = keys;

	vector<cv::Mat> queryDescs_v(1);
	F::toMat32F(old_descs, queryDescs_v[0]);

	vector<vector<cv::DMatch> > matches;

	flann_structure.knnMatch(queryDescs_v[0], matches, 2);

	for (int old_idx = 0; old_idx < (int) matches.size(); ++old_idx) {
		if (!matches[old_idx].empty()) {
			int cur_idx = matches[old_idx][0].trainIdx;
			float dist = matches[old_idx][0].distance;

			bool ok = true;
			if (matches[old_idx].size() >= 2) {
				float dist_ratio = dist / matches[old_idx][1].distance;
				ok = dist_ratio <= m_params.max_neighbor_ratio;
			}

			if (ok) {
				vector<unsigned int>::iterator cit = std::find(i_cur.begin(),
						i_cur.end(), cur_idx);

				if (cit == i_cur.end()) {
					i_old.push_back(old_idx);
					i_cur.push_back(cur_idx);
				} else {
					int idx = i_old[cit - i_cur.begin()];
					if (dist < matches[idx][0].distance) {
						i_old[cit - i_cur.begin()] = old_idx;
					}
				}
			}
		}
	}

	if ((int) i_old.size() >= m_params.min_Fpoints) {
		// add matches to the vectors for calculating the fundamental matrix
		vector<unsigned int>::const_iterator oit, cit;
		oit = i_old.begin();
		cit = i_cur.begin();

		vector<cv::Point2f> old_points, cur_points;
		old_points.reserve(i_old.size());
		cur_points.reserve(i_cur.size());

		for (; oit != i_old.end(); ++oit, ++cit) {
			const cv::KeyPoint &old_k = old_keys[*oit];
			const cv::KeyPoint &cur_k = cur_keys[*cit];

			old_points.push_back(old_k.pt);
			cur_points.push_back(cur_k.pt);
		}

		cv::Mat oldMat(old_points.size(), 2, CV_32F, &old_points[0]);
		cv::Mat curMat(cur_points.size(), 2, CV_32F, &cur_points[0]);

		return m_fsolver.checkFundamentalMat(oldMat, curMat,
				m_params.max_reprojection_error, m_params.min_Fpoints,
				m_params.ransac_probability, m_params.max_ransac_iterations);
	}

	return false;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedLoopDetector<TDescriptor, F>::getMatches_neighratio(
		const vector<TDescriptor> &A, const vector<unsigned int> &i_A,
		const vector<TDescriptor> &B, const vector<unsigned int> &i_B,
		vector<unsigned int> &i_match_A,
		vector<unsigned int> &i_match_B) const {
	i_match_A.resize(0);
	i_match_B.resize(0);
	i_match_A.reserve(min(i_A.size(), i_B.size()));
	i_match_B.reserve(min(i_A.size(), i_B.size()));

	vector<unsigned int>::const_iterator ait, bit;
	unsigned int i, j;
	i = 0;
	for (ait = i_A.begin(); ait != i_A.end(); ++ait, ++i) {
		int best_j_now = -1;
		double best_dist_1 = 1e9;
		double best_dist_2 = 1e9;

		j = 0;
		for (bit = i_B.begin(); bit != i_B.end(); ++bit, ++j) {
			if (*ait >= A.size()) {
				std::cout << "Error! A[*ait] does not exist!" << std::endl;
				std::cout << (int) A.size() << ", " << *ait << "(" << i_A.size()
						<< ")" << std::endl;
			}
			if (*bit >= B.size()) {
				std::cout << "Error! B[*bit] does not exist!" << std::endl;
				std::cout << (int) B.size() << ", " << *bit << std::endl;
			}
			double d = F::distance(A[*ait], B[*bit]);

			// in i
			if (d < best_dist_1) {
				best_j_now = j;
				best_dist_2 = best_dist_1;
				best_dist_1 = d;
			} else if (d < best_dist_2) {
				best_dist_2 = d;
			}
		}

		if (best_dist_1 / best_dist_2 <= m_params.max_neighbor_ratio) {
			if (best_j_now >= i_B.size()) {
				std::cout << "Error! i_B[*best_j_now] does not exist!"
						<< std::endl;
			}
			unsigned int idx_B = i_B[best_j_now];
			bit = find(i_match_B.begin(), i_match_B.end(), idx_B);

			if (bit == i_match_B.end()) {
				i_match_B.push_back(idx_B);
				i_match_A.push_back(*ait);
			} else {
				if (bit - i_match_B.begin() >= i_match_A.size()) {
					std::cout
							<< "Error! i_match_A[bit - i_match_B.begin()] does not exist!"
							<< std::endl;
				}
				unsigned int idx_A = i_match_A[bit - i_match_B.begin()];

				if (idx_A >= A.size()) {
					std::cout << "Error! A[idx_A] does not exist!" << std::endl;
				}
				if (idx_B >= B.size()) {
					std::cout << "Error! B[idx_B] does not exist!" << std::endl;
				}
				double d = F::distance(A[idx_A], B[idx_B]);

				if (best_dist_1 < d) {
					i_match_A[bit - i_match_B.begin()] = *ait;
				}
			}

		}
	}
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedLoopDetector<TDescriptor, F>::getMatches_neighratio(
		const vector<TDescriptor> &A, const vector<unsigned int> &i_A,
		const vector<TDescriptor> &B, const vector<unsigned int> &i_B,
		vector<unsigned int> &i_match_A, vector<unsigned int> &i_match_B,
		vector<double> &i_match_d) const {
	i_match_A.resize(0);
	i_match_B.resize(0);
	i_match_d.resize(0);
	i_match_A.reserve(min(i_A.size(), i_B.size()));
	i_match_B.reserve(min(i_A.size(), i_B.size()));
	i_match_d.reserve(min(i_A.size(), i_B.size()));

	vector<unsigned int>::const_iterator ait, bit;
	unsigned int i, j;
	i = 0;
	for (ait = i_A.begin(); ait != i_A.end(); ++ait, ++i) {
		int best_j_now = -1;
		double best_dist_1 = 1e9;
		double best_dist_2 = 1e9;

		j = 0;
		for (bit = i_B.begin(); bit != i_B.end(); ++bit, ++j) {
			if (*ait >= A.size()) {
				std::cout << "Error! A[*ait] does not exist!" << std::endl;
				std::cout << (int) A.size() << ", " << *ait << "(" << i_A.size()
						<< ")" << std::endl;
			}
			if (*bit >= B.size()) {
				std::cout << "Error! B[*bit] does not exist!" << std::endl;
				std::cout << (int) B.size() << ", " << *bit << std::endl;
			}
			double d = F::distance(A[*ait], B[*bit]);

			// in i
			if (d < best_dist_1) {
				best_j_now = j;
				best_dist_2 = best_dist_1;
				best_dist_1 = d;
			} else if (d < best_dist_2) {
				best_dist_2 = d;
			}
		}

		if (best_dist_1 / best_dist_2 <= m_params.max_neighbor_ratio) {
			if (best_j_now >= i_B.size()) {
				std::cout << "Error! i_B[*best_j_now] does not exist!"
						<< std::endl;
			}
			unsigned int idx_B = i_B[best_j_now];
			bit = find(i_match_B.begin(), i_match_B.end(), idx_B);

			if (bit == i_match_B.end()) {
				i_match_B.push_back(idx_B);
				i_match_A.push_back(*ait);
				i_match_d.push_back(best_dist_1);
			} else {
				if (bit - i_match_B.begin() >= i_match_A.size()) {
					std::cout
							<< "Error! i_match_A[bit - i_match_B.begin()] does not exist!"
							<< std::endl;
				}
				unsigned int idx_A = i_match_A[bit - i_match_B.begin()];

				if (idx_A >= A.size()) {
					std::cout << "Error! A[idx_A] does not exist!" << std::endl;
				}
				if (idx_B >= B.size()) {
					std::cout << "Error! B[idx_B] does not exist!" << std::endl;
				}
				double d = F::distance(A[idx_A], B[idx_B]);

				if (best_dist_1 < d) {
					i_match_A[bit - i_match_B.begin()] = *ait;
				}
			}

		}
	}
}

// --------------------------------------------------------------------------

// Driver function to sort the vector elements
// by second element of pairs
bool inline SortByMatchValue(const cv::DMatch &a, const cv::DMatch &b) {
	return (a.distance < b.distance);
}
;

template<class TDescriptor, class F>
void TemplatedLoopDetector<TDescriptor, F>::getMatches(
		const vector<TDescriptor> &A, const vector<unsigned int> &i_A,
		const vector<TDescriptor> &B, const vector<unsigned int> &i_B,
		vector<unsigned int> &i_match_A,
		vector<unsigned int> &i_match_B) const {
	i_match_A.resize(0);
	i_match_B.resize(0);
	i_match_A.reserve(min(i_A.size(), i_B.size()));
	i_match_B.reserve(min(i_A.size(), i_B.size()));

	vector<cv::DMatch> list_matches, list_matches_tmp;
	vector<unsigned int>::const_iterator ait, bit;
	unsigned int i, j;
	i = 0;
	for (ait = i_A.begin(); ait != i_A.end(); ++ait, ++i) {
		j = 0;
		for (bit = i_B.begin(); bit != i_B.end(); ++bit, ++j) {
			double d = F::distance(A[*ait], B[*bit]);

			if (m_params.matching_th == -1 || d < m_params.matching_th) {
				cv::DMatch match(i, j, d);
				list_matches_tmp.push_back(match);
			}
		}

		if (list_matches_tmp.size() > 1 && m_params.b_snn) {
			std::sort(list_matches_tmp.begin(), list_matches_tmp.end(),
					SortByMatchValue);

			if (list_matches_tmp[0].distance
					< m_params.max_neighbor_ratio
							* list_matches_tmp[1].distance) {
				list_matches.push_back(list_matches_tmp[0]);
			}
		} else {
			list_matches.insert(list_matches.end(), list_matches_tmp.begin(),
					list_matches_tmp.end());
		}
	}

	std::sort(list_matches.begin(), list_matches.end(), SortByMatchValue);

	std::map<int, bool> query_matched, train_matched;

	std::vector<cv::DMatch>::iterator iti, ite;
	iti = list_matches.begin();
	ite = list_matches.end();

	//std::cout << "Matches:" << std::endl;
	for (; iti != ite; ++iti) {
		if (!query_matched[iti->queryIdx] && !train_matched[iti->trainIdx]) {
			query_matched[iti->queryIdx] = true;
			train_matched[iti->trainIdx] = true;

			i_match_B.push_back(iti->trainIdx);
			i_match_A.push_back(iti->queryIdx);
		}
	}

}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedLoopDetector<TDescriptor, F>::removeLowScores(QueryResults &q,
		double threshold) const {
	// remember scores in q are in descending order now
	//QueryResults::iterator qit =
	//  lower_bound(q.begin(), q.end(), threshold, Result::geqv);

	Result aux(0, threshold);
	QueryResults::iterator qit = lower_bound(q.begin(), q.end(), aux,
			Result::geq);

	// qit = first element < m_alpha_minus || end

	if (qit != q.end()) {
		int valid_entries = qit - q.begin();
		q.resize(valid_entries);
	}
}

// --------------------------------------------------------------------------
template<class TDescriptor, class F>
void TemplatedLoopDetector<TDescriptor, F>::AddBowVectorToDB(
		const std::vector<cv::KeyPoint> &keys,
		const std::vector<DBoW2::FORB::TDescriptor> &orb_descs) {
	EntryId entry_id = m_database->size();

	BowVector bowvec;
	FeatureVector featvec;

	if (m_params.geom_check == GEOM_DI)
		m_database->getVocabulary()->transform(orb_descs, bowvec, featvec,
				m_params.di_levels);
	else
		m_database->getVocabulary()->transform(orb_descs, bowvec);

	// update database
	m_database->add(bowvec, featvec);  // returns entry_id

	// update record
	// m_image_keys and m_image_descriptors have the same length
	if (m_image_keys.size() == entry_id) {
		m_image_keys.push_back(keys);
		m_image_descriptors.push_back(orb_descs);
	} else {
		m_image_keys[entry_id] = keys;
		m_image_descriptors[entry_id] = orb_descs;
	}

	// store this bowvec if we are going to use it in next iteratons
	m_last_bowvec = bowvec;
//	if (m_params.use_nss && (int) entry_id + 1 > m_params.dislocal) {
//		m_last_bowvec = bowvec;
//	}
}

// --------------------------------------------------------------------------

// TODO: save also distances
template<class TDescriptor, class F>
void TemplatedLoopDetector<TDescriptor, F>::SaveMatchedPointsOfPlace(
		PlaceFound pt) const {
	std::cout << pt.query_pts.size() << " <-> " << pt.train_pts.size()
	    <<  " <-> " << pt.distances.size() <<  " <-> " << pt.inliers.size()
			<< std::endl;
	assert(pt.query_pts.size() == pt.train_pts.size());

	std::string filename = "agent" + std::to_string(pt.train_agent_id)
			+ "_kps_F" + std::to_string(pt.place_found_id) + "_Q"
			+ std::to_string(pt.query_frame_id) + ".txt";

	int N = pt.train_pts.size();

	FILE* fout = fopen(filename.c_str(), "w");

	fprintf(fout, "%d %d\n", pt.query_agent_id, pt.query_place_id);
	fprintf(fout, "%d %d\n", pt.train_agent_id, pt.place_found_id);

	for (int j = 0; j < N; ++j) {
		//if (pt.inliers[j]) {
		//  fprintf(fout, "%f %f %f %f\n", pt.query_pts[j].x, pt.query_pts[j].y,
		//          pt.train_pts[j].x, pt.train_pts[j].y);
		//}
		fprintf(fout, "%f %f %f %f %d %.6f\n", pt.query_pts[j].x, pt.query_pts[j].y,
				pt.train_pts[j].x, pt.train_pts[j].y, (int) pt.inliers[j], pt.distances[j]);
	}
	fclose(fout);
}

// --------------------------------------------------------------------------
// Print island results
template<class TDescriptor, class F>
void TemplatedLoopDetector<TDescriptor, F>::PrintIslandsResults(
		const std::vector<tIsland>& islands) const {
	std::cout << "[" << std::endl;
	for (auto &isl : islands)
		std::cout << "Island: " << isl.first << "-" << isl.last << " ("
				<< isl.score << "), " << "Best: " << isl.best_entry << " ("
				<< isl.best_score << ")" << std::endl;
	std::cout << "]" << std::endl;
}

// --------------------------------------------------------------------------
// Print candidate results
template<class TDescriptor, class F>
void TemplatedLoopDetector<TDescriptor, F>::PrintCandidateResults(
		const QueryResults& qret) const {
	std::cout << "[";
	for (auto id : qret) {
		if (id == qret.size()) {
			std::cout << id.Id << " (" << id.Score << ")";
		} else {
			std::cout << id.Id << " (" << id.Score << "), ";
		}
	}
	std::cout << "]" << std::endl;
}


}    // namespace DLoopDetector

#endif
