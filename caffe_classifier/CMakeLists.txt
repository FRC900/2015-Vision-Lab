cmake_minimum_required(VERSION 2.8)
set(CMAKE_LEGACY_CYGWIN_WIN32 0) # Remove when CMake >= 2.8.4 is required
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -march=native -flto")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -Ofast -march=native -flto")

project( CaffePrediction )
find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED )
find_package( Boost COMPONENTS system REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} /home/kjaget/caffe_bvlc/caffe/install/include ${Boost_INCLUDE_DIR} ../nonmax_suppression "../Scale Factor" )

find_library (LibCaffe caffe PATH  /home/kjaget/caffe_bvlc/caffe/install/lib )
find_library (LibProto proto PATH  /home/kjaget/caffe_bvlc/caffe/install/lib )
find_library (LibGLOG  glog)
find_library (LibProtobuf protobuf)

cuda_add_executable( CaffeBatchPrediction ../nonmax_suppression/fast_nms.cpp main.cpp )
target_link_libraries( CaffeBatchPrediction ${OpenCV_LIBS} ${LibCaffe} ${LibProto} ${Boost_LIBRARIES} ${LibGLOG} ${LibProtobuf} )
