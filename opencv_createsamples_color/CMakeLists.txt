cmake_minimum_required(VERSION 2.8)
project( opencv_createsamples_color )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( opencv_createsamples_color cvboost.cpp cvcommon.cpp cvhaarclassifier.cpp cvhaartraining.cpp cvsamples.cpp cvsamplesoutput.cpp createsamples.cpp )
target_link_libraries( opencv_createsamples_color ${OpenCV_LIBS} )

