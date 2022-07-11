/*
 * run_all_tests.cpp
 *
 *  Created on: 19 Apr 2019
 *      Author: alessioxompero
 */

#include <iostream>

#include "tests/tests.h"

int main(int argc, char *argv[]) {

  Serialization_TEST();

  FrameSerialization_TEST();

  return EXIT_SUCCESS;
}

