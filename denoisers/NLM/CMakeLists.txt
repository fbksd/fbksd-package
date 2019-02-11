cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)

project(NLM)

set(CMAKE_CXX_STANDARD 14)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "7.0.0")
        message(FATAL_ERROR "gcc version greather than 6 not supported by cuda.")
    endif()
endif()

find_package(fbksd REQUIRED)
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
set(SRC src/main.cpp
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
        src/image.cpp
        src/image.h
        src/memory.cpp
        src/memory.h
        src/montecarlo.cpp
        src/montecarlo.h
        src/parallel.cpp
        src/parallel.h
        src/rng.cpp
        src/rng.h
        src/sampler.cpp
        src/sampler.h
        src/nlmdenoiser.cpp
        src/nlmdenoiser.h
        src/dualfilm.cpp
        src/dualfilm.h
        src/dualsampler.cpp
        src/dualsampler.h
        src/kernel2d.cpp
        src/kernel2d.h
        src/nlmkernel.cu
        src/nlmkernel.h
        src/pbrt.h
)

cuda_add_executable(NLM ${SRC})
target_link_libraries(NLM fbksd::client)
install(TARGETS NLM RUNTIME DESTINATION .)
install(FILES info.json DESTINATION .)
