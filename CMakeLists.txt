cmake_minimum_required(VERSION 3.10)
project(DBSCANFramework CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP encontrado")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP NO encontrado: modo omp no funcionará")
endif()

include_directories(include)

file(GLOB_RECURSE SOURCES
    src/*.cpp
    src/neighbors/*.cpp
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(dbscan_app ${SOURCES})

if(OpenMP_CXX_FOUND)
    target_link_libraries(dbscan_app PUBLIC OpenMP::OpenMP_CXX)
endif()
