cmake_minimum_required(VERSION 3.1)

project(VecMat VERSION 0.1.0)
enable_language(CXX)

set(CMAKE_CXX_STANDARD 14) 
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
set(CMAKE_CXX_EXTENSIONS OFF)


if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
   message(   "Clang is used" )
   set(CMAKE_CXX_FLAGS "-O2  -Wall -Wextra ")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
   message(   "gcc is used" )
   set(CMAKE_CXX_FLAGS "-O2  -march=native -Wall -Wextra -pedantic -Wshadow ")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
   message(   "intel is used" )
#elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
# using Visual Studio C++
else()
   message( FATAL_ERROR  "Not supported" )
endif()

set(CMAKE_CXX_FLAGS_DEBUG -DDEBUG)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PRETTYPRINT})

# if  boost is necessary
#find_package(Boost  REQUIRED COMPONENTS program_options)
#include_directories(${Boost_INCLUDE_DIRS})




add_executable(demo demo.cpp)
add_executable(speed speed_test.cpp)

# if  boost is necessary
#target_link_libraries(cxx_test Boost::program_options) 

##for blas
set(NO_BLASSUPPORT ${NO_BLASSUPORT})
if(NO_BLASSUPPORT)
add_definitions(-DNO_vecNd_BLAS)
else(NO_BLASSUPPORT)
message("BLAS will be used")
add_definitions(-DvecNd_BLAS)

if(MKL)
message("MKL to be used")
set(BLA_VENDOR "Intel10_64lp" )
find_package(BLAS ) 
find_package(LAPACK) 

target_link_libraries(demo BLAS::BLAS) 
target_link_libraries(speed BLAS::BLAS) 

else(MKL)
find_package(BLAS ) 
target_link_libraries(demo BLAS::BLAS) 
target_link_libraries(demo lapacke) 
target_link_libraries(speed BLAS::BLAS) 
target_link_libraries(speed lapacke) 
endif(MKL)


endif(NO_BLASSUPPORT)


# TODO Make test in test dir.
enable_testing()
# サブディレクトリを指定．指定したディレクトリ内のCMakeLists.txt が実行される
add_subdirectory(test)


