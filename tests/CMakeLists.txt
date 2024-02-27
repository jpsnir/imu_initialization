# tests for imu initialization code.

find_package(GTest REQUIRED)
include_directories(${GTest_INCLUDE_DIRS})
add_executable(test_experiment02 ./test_experiment02.cpp)
target_link_libraries(test_experiment02 ${GTest_LIBRARIES})
