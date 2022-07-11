/**
 *
 * @fn main.cpp
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
#include <iostream>
#include <string>

// My headers
#include <agent.h>
#include <utilities.h>

//
int main(int argc, char *argv[]) {
  if (argc < 7 || argc > 21) {
    std::cout << "Wrong number of input!" << std::endl;
    return ShowHelp();
  }

  PrintZMQversion();

//  // Run Unit Tests
//  TestFundamentalMatrixDLib();
//  TestFundamentalMatrixOpenCV();
//  TestFundamentalMatrixRANSACOpenCV();
//
//  return EXIT_SUCCESS;

  // Parse input
  Params params;
  params.ParseInputParams(argc, argv);
  params.PrintParams();


  Agent camera(params);
  camera.Run();

  return EXIT_SUCCESS;
}
