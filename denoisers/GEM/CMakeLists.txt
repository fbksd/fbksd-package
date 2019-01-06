cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)

project(GEM)

set(CMAKE_CXX_STANDARD 14)

find_package(fbksd REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


set(SRC src/bandwidth.cpp
        src/denoiser.cpp
        src/film.cpp
        src/filter.cpp
        src/gaussian.cpp
        src/geometry.cpp
        src/kernel2d.cpp
        src/montecarlo.cpp
        src/random.cpp
        src/rng.cpp
        src/sampler.cpp
        src/smooth.cpp
        src/stratified.cpp
        src/memory.cpp
        src/parallel.cpp
        src/bandwidth.h
        src/denoiser.h
        src/film.h
        src/filter.h
        src/gaussian.h
        src/geometry.h
        src/kernel2d.h
        src/montecarlo.h
        src/random.h
        src/rng.h
        src/sampler.h
        src/smooth.h
        src/stratified.h
        src/memory.h
        src/parallel.h
        src/error.h
)

add_library(GEM_core ${SRC})
target_link_libraries(GEM_core fbksd::client)

add_executable(GEM src/main.cpp)
target_link_libraries(GEM GEM_core)
install(TARGETS GEM RUNTIME DESTINATION .)
install(FILES info.json DESTINATION .)

add_executable(GEM-sampler src/main_sampler.cpp)
target_link_libraries(GEM-sampler GEM_core)
