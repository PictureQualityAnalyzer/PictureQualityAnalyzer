cmake_minimum_required(VERSION 2.8)
project( analyze )
find_package( OpenCV REQUIRED )
add_executable( analyze analyze.cpp)
target_link_libraries( analyze ${OpenCV_LIBS} )
