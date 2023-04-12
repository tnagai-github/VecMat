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

include_directories(~/src/cxx-prettyprint)
include_directories(${PRETTYPRINT})

# if  boost is necessary
#find_package(Boost  REQUIRED COMPONENTS program_options)
#include_directories(${Boost_INCLUDE_DIRS})




add_executable(demo demo.cpp)
add_executable(speed speed_test.cpp)

# if  boost is necessary
#target_link_libraries(cxx_test Boost::program_options) 

##for blas
#set(BLASSUPPORT ${BLASSUPORT})
set(BLASSUPPORT true)
if(BLASSUPPORT)
message("BLAS will be used")
add_definitions(-DvecNd_BLAS)
##set(BLA_VENDOR "Intel10_64ilp_seq" )
#set ( ENV{BLA_VENDOR} "Intel10_64ilp_seq" )
#find_package(LAPACK) 
#find_package(BLAS ) 
target_link_libraries(demo lapacke) 
target_link_libraries(demo blas) 

target_link_libraries(speed lapacke) 
target_link_libraries(speed blas) 

#target_link_libraries(demo BLAS::BLAS ${LAPACK_LIBRARIES})
#target_link_libraries(demo mkl_intel_ilp64 mkl_sequential mkl_core pthread m dl ) 
#target_link_libraries(demo mkl_intel_ilp64 mkl_intel_thread mkl_core iomp5 pthread m dl ) 
#target_link_libraries(speed mkl_intel_ilp64 mkl_intel_thread mkl_core iomp5 pthread m dl ) 

endif(BLASSUPPORT)


# TODO Make test in test dir.
#enable_testing()
# サブディレクトリを指定．指定したディレクトリ内のCMakeLists.txt が実行される
#add_subdirectory(test)


