cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)

project(RPF)

set(CMAKE_CXX_STANDARD 14)

find_package(fbksd REQUIRED)

# OpenEXR
# FIXME: what if OpenEXR is in a different path
include_directories(/usr/include/OpenEXR)
set(LIB_OPENEXR IlmImf Half)

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(RPF_FILES src/CImg.h
              src/errcheck.inc
              src/ExrUtilities.h
              src/Globals.h
              src/MutualInformationWrapper.h
              src/PreComputeGaussian.h
              src/RPF.h
              src/SampleSet.h
              src/SSE_Math.h
              src/SSE_Wrapper.h
              src/SampleWriter/CImg.h
              src/SampleWriter/Globals.h
              src/SampleWriter/SampleWriter.h
              src/timer.h
              src/ExrUtilities.cpp
              src/Globals.cpp
              src/MutualInformationWrapper.cpp
              src/PreComputeGaussian.cpp
              src/RPF.cpp
              src/SampleSet.cpp
              src/SSE_Math.cpp
              src/SSE_Wrapper.cpp
              src/SampleWriter/SampleWriter.cpp
              src/timer.cpp
              src/main.cpp
)

add_executable(RPF ${RPF_FILES})
target_link_libraries(RPF ${LIB_OPENEXR} X11 fbksd::client)
install(TARGETS RPF RUNTIME DESTINATION .)
install(FILES info.json DESTINATION .)