cmake_minimum_required(VERSION 3.10)
project(JetsonBigBrother)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(HEADERS include/Person.h include/Process.h include/thread_utility.h)
set(SOURCES src/Person.cpp src/Process.cpp src/thread_utility.cpp main.cpp)
find_package(OpenCV 4.3 REQUIRED)

add_executable(jetbb ${SOURCES})
target_include_directories(jetbb PUBLIC
        "${PROJECT_BINARY_DIR}"
        "${PROJECT_SOURCE_DIR}/include"
        )
target_link_libraries(jetbb ${OpenCV_LIBRARIES} -pthread)
