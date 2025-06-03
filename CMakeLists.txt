cmake_minimum_required(VERSION 3.26)
project(model_confidence LANGUAGES CXX)
set(TARGET model_confidence)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(${TARGET} main.cpp)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp)

# include llama.h and llama-cpp.h
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/llama.cpp/include)

target_link_libraries(${TARGET} PRIVATE llama)
