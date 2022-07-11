cmake_minimum_required(VERSION 2.8)
project(XCPR)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-ignored-attributes -Wno-deprecated-declarations -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-attributes -Wno-deprecated-declarations -O3 -march=native")

################################################################################
# Check C++11 or C++0x support

include(CheckCXXCompilerFlag)

CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fext-numeric-literals")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

################################################################################
# Find packages and libraries
################################################################################

# ZeroMQ
find_library(ZeroMQ
        NAMES zmq
        PATHS "/usr/local/zmq"
        )

if(NOT ZeroMQ)
   message(FATAL_ERROR "ZeroMQ not found.")
else()
	message(STATUS "ZeroMQ found.")
endif()

# OpenCV
set(OpenCV_DIR "/usr/local/opencv/opencv-3.4.1")
find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR} NO_DEFAULT_PATH)

if(NOT OpenCV_FOUND)
   message(FATAL_ERROR "OpenCV > 3.x not found.")
else()
	message(STATUS "OpenCV found. " ${OpenCV_DIR})
endif()

# Boost
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost 1.65 COMPONENTS serialization REQUIRED)
if(NOT Boost_FOUND)
  message(FATAL_ERROR "Boost > 1.65 not found.")
else()
	message(STATUS "Boost found.")
endif()


# Find DLib
set(DLib_DIR "${PROJECT_SOURCE_DIR}/extern/DBoW2/build/dependencies/src/DLib-build")
find_package(DLib REQUIRED PATHS ${DLib_DIR} NO_DEFAULT_PATH)

if(NOT DLib_FOUND)
   message(FATAL_ERROR "DLib not found.")
else()
	message(STATUS "DLib found.")
endif()

set(DLib_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/extern/DBoW2/build/dependencies/src/DLib/include")

# DBoW2
set(DBoW2_DIR "${PROJECT_SOURCE_DIR}/extern/DBoW2/build")
find_package(DBoW2 REQUIRED PATHS ${DBoW2_DIR} NO_DEFAULT_PATH)

if(NOT DBoW2_FOUND)
   message(FATAL_ERROR "DDBoW2 not found.")
else()
	message(STATUS "DBoW2 found.")
endif()

set(DBoW2_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/extern/DBoW2/include")
set(DBoW2_LIBS "${PROJECT_SOURCE_DIR}/extern/DBoW2/build")


find_library(DBoW2_LIB DBoW2 PATH "${PROJECT_SOURCE_DIR}/extern/DBoW2/build")

set(HDRS
include/agent.h
include/DLoopDetector.h         
include/TemplatedLoopDetector.h
include/GlobalFeatDB.h
include/Frame.h
include/messages.h
include/params.h
include/timer.h
include/utilities.h
)


# Include directories
include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${DBoW2_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}/extern/DBoW2/build/dependencies/install/include
${DLib_INCLUDE_DIRS}
${ZeroMQ_INCLUDE_DIR}
${OpenCV_INCLUDE_DIRS}
${Boost_INCLUDE_DIRS}
)


add_library(${PROJECT_NAME}
src/agent.cpp
src/frame.cpp
src/GlobalFeatDB.cpp
src/utilities.cpp
src/unit_tests.cpp
)


################################################################################
# Build examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

add_executable(Agent
${PROJECT_SOURCE_DIR}/src/main.cpp)

target_link_libraries(Agent
${PROJECT_NAME} 
${OpenCV_LIBS}
${DBoW2_LIB}
${DLib_LIBS}
-lboost_serialization
-lpthread 
-lzmq
)

