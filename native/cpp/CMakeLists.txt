cmake_minimum_required(VERSION 3.18)
project(GFPCMTest)

# Use C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
find_package(Eigen3 REQUIRED NO_MODULE)

find_package(pybind11 REQUIRED)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Add the executable and link Eigen
# add_executable(gfpcm_test main.cpp GFPCM.cpp STPFCM.cpp Metrics.cpp MultivariateParamTester.cpp CentroidStrategy.cpp DatasetLoader.cpp)
# target_link_libraries(gfpcm_test Eigen3::Eigen)

# Add pybind11 module for STPFCM
pybind11_add_module(_algofuzz
    STPFCM.cpp
    CentroidStrategy.cpp
    DatasetLoader.cpp
    Metrics.cpp
    algofuzz_pybind.cpp
)
target_link_libraries(_algofuzz PRIVATE Eigen3::Eigen)
install(TARGETS _algofuzz DESTINATION ${SKBUILD_PROJECT_NAME})