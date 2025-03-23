cmake_minimum_required(VERSION 3.16)
project(KalaCL LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)

# Enable AVX and FMA optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mfma")

# Add the library (shared)
add_library(kalacl SHARED src/kala_cell.cpp)

# Include the header files
target_include_directories(kalacl PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Create an executable for testing
add_executable(KalaCL src/main.cpp
        src/dataset_loader.cpp
        include/dataset_loader.h
        include/kala_rnn.h
        src/kala_network.cpp
        src/kala_rnn.cpp)

# Link the library to the test program
target_link_libraries(KalaCL PRIVATE kalacl)
