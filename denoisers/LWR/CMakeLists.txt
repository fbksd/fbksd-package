cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)

project(LWR)

set(CMAKE_CXX_STANDARD 14)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "7.0.0")
        message(FATAL_ERROR "gcc version greather than 6 not supported by cuda.")
    endif()
endif()

find_package(fbksd REQUIRED)

# OpenEXR
# FIXME: OpenEXR can be in a different path
include_directories(/usr/include/OpenEXR)

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# CUDA
find_package(CUDA REQUIRED)
# fixes some compile errors (probably Cuda bugs on Ubuntu 16.04)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES")

include_directories(src)
set(SRC src/lwr_film.cpp
        src/lwr_film.h
        src/lwrr.cpp
        src/lwrr.h
        src/lwrr_fit.cu
        src/lwrr_fit.h
        src/svd.cuh
        src/lwrr_mem.h
        src/lwrr_setting.h
        src/lwr_sampler.cpp
        src/lwr_sampler.h
        src/sampler.cpp
        src/sampler.h
        src/geometry.cpp
        src/geometry.h
        src/rng.cpp
        src/rng.h
        src/film.cpp
        src/film.h
        src/filter.cpp
        src/filter.h
        src/gaussian.cpp
        src/gaussian.h
        src/memory.cpp
        src/memory.h
        src/parallel.cpp
        src/parallel.h
        src/montecarlo.cpp
        src/montecarlo.h
        src/stdafx.h
)

set(LWR_SRC src/main.cpp)
set(LWR_MF_SRC src/main-mf.cpp)

cuda_add_library(lwr-core STATIC ${SRC})
cuda_add_cublas_to_target(lwr-core)
target_link_libraries(lwr-core fbksd::client)

cuda_add_executable(LWR ${LWR_SRC})
target_link_libraries(LWR lwr-core)
install(TARGETS LWR RUNTIME DESTINATION .)
install(FILES info.json DESTINATION .)

cuda_add_executable(LWR-mf ${LWR_MF_SRC})
target_link_libraries(LWR-mf lwr-core IlmImf)
install(TARGETS LWR-mf RUNTIME DESTINATION .)
