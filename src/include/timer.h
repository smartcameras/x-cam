/*
 * timer.h
 *
 *  Created on: 26 Nov 2018
 *      Author: alessioxompero
 */

#ifndef INCLUDE_TIMER_H_
#define INCLUDE_TIMER_H_

#include <chrono>

class Timer {
 private:
  // Type aliases to make accessing nested type easier
  using clock_t = std::chrono::high_resolution_clock;
  using second_t = std::chrono::duration<double, std::ratio<1> >;

  std::chrono::time_point<clock_t> m_beg;

 public:
  Timer()
      : m_beg(clock_t::now()) {
  }

  void reset() {
    m_beg = clock_t::now();
  }

  double elapsed() const {
    return std::chrono::duration_cast<second_t>(clock_t::now() - m_beg).count();
  }

  void print(const std::string& heading) {
    std::cout << "***Running time for " << heading << ": ";
    if (elapsed() < 1) {
      std::cout << elapsed() * pow(10, 3) << " milliseconds\n";
    } else if (elapsed() < 60) {
      std::cout << elapsed() << " seconds\n";
    } else {
      std::cout << elapsed() / 60 << " minutes\n";
    }

  }

  /**
   * modes:
   *  1. Local feature extraction
   *  2. Feature tracking
   *  3. Re-detection
   *  4. Frame processing
   *  5. VPR_Processing
   */
  void savetofile(const std::string& filename, const int& frame_index, const int& mode) {
    FILE* fout = fopen(filename.c_str(), "a");
    fprintf(fout, "%d %.6f %d\n", frame_index, elapsed() * pow(10, 3), mode);
    fclose(fout);
  }

};

#endif /* INCLUDE_TIMER_H_ */
