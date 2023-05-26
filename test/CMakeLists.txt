
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)



add_executable(test001 test001.cpp)
target_link_libraries(
	test001
	GTest::gtest_main
)
target_link_libraries(test001 lapacke) 
target_link_libraries(test001 blas) 

include(GoogleTest)
gtest_discover_tests(test001)


