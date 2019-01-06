cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)

project(SBF)

set(CMAKE_CXX_STANDARD 14)

find_package(fbksd REQUIRED)

#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# OpenEXR
include_directories(/usr/include/OpenEXR)

set(SRC
    CrossBilateralFilter.cpp
    CrossBilateralFilter.h
    CrossNLMFilter.cpp
    CrossNLMFilter.h
    film.cpp
    film.h
    filter.cpp
    filter.h
    fmath.hpp
    geometry.cpp
    geometry.h
    memory.cpp
    memory.h
    montecarlo.cpp
    montecarlo.h
    pbrt.h
    ReconstructionFilter.h
    rng.cpp
    rng.h
    sampler.cpp
    sampler.h
    sbfsampler.cpp
    sbfsampler.h
    sbf.cpp
    sbf.h
    SBFCommon.h
    stdafx.cpp
    stdafx.h
    TwoDArray.h
    VectorNf.h
    sbfimage.cpp
    sbfimage.h
    imageio.h
    imageio.cpp

    filters/box.cpp
    filters/gaussian.cpp
    filters/mitchell.cpp
    filters/sinc.cpp
    filters/triangle.cpp
    filters/box.h
    filters/gaussian.h
    filters/mitchell.h
    filters/sinc.h
    filters/triangle.h
)

set(SBF_SRC main.cpp)
set(SBF_MF_SRC main-mf.cpp)

add_library(sbf-core STATIC ${SRC})
target_link_libraries(sbf-core PUBLIC IlmImf Half fbksd::client)
target_include_directories(sbf-core PUBLIC ${CMAKE_SOURCE_DIR})

add_executable(SBF ${SBF_SRC})
target_link_libraries(SBF sbf-core)
install(TARGETS SBF RUNTIME DESTINATION .)
install(FILES info.json DESTINATION .)

add_executable(SBF-mf ${SBF_MF_SRC})
target_link_libraries(SBF-mf sbf-core)
install(TARGETS SBF-mf RUNTIME DESTINATION .)
