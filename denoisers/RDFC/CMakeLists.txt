cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)

project(RDFC)

set(CMAKE_CXX_STANDARD 14)

find_package(fbksd REQUIRED)

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# OpenEXR
include_directories(/usr/include/OpenEXR)
set(OPENEXR_LIBRARIES IlmImf Half)

# CUDA
find_package(CUDA REQUIRED)
# fixes some compile errors (probably Cuda bugs on Ubuntu 16.04)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES")

include_directories(src)
set(SRC src/main.cpp
        src/featurefilter.h
        src/featurefilter.cu
        src/box.cpp
        src/box.h
        src/film.cpp
        src/film.h
        src/filter.cpp
        src/filter.h
        src/gaussian.cpp
        src/gaussian.h
        src/geometry.cpp
        src/geometry.h
        src/memory.cpp
        src/memory.h
        src/montecarlo.cpp
        src/montecarlo.h
        src/parallel.cpp
        src/parallel.h
        src/random.cpp
        src/random.h
        src/rng.cpp
        src/rng.h
        src/sampler.cpp
        src/sampler.h
        src/multifilm.cpp
        src/multifilm.h
        src/multisampler.cpp
        src/multisampler.h
        src/pbrt.h
        src/stdafx.h
        src/stdafx.cpp
        src/imageio.h
        src/imageio.cpp
)

cuda_add_executable(RDFC ${SRC})
target_link_libraries(RDFC ${OPENEXR_LIBRARIES} fbksd::client)
install(TARGETS RDFC RUNTIME DESTINATION .)
install(FILES info.json DESTINATION .)
