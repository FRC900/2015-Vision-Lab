cmake_minimum_required(VERSION 2.8)
set(CMAKE_LEGACY_CYGWIN_WIN32 0) # Remove when CMake >= 2.8.4 is required
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -flto")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Ofast -flto")
project( scalefactor )
find_package( OpenCV REQUIRED )
add_executable( scalefactor scalefactor.cpp)
target_link_libraries( scalefactor ${OpenCV_LIBS} )
