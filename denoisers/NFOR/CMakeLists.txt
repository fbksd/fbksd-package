cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)

project(NFOR)

set(CMAKE_CXX_STANDARD 14)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "7.0.0")
        message(FATAL_ERROR "gcc version greather than 6 not supported by cuda.")
    endif()
endif()

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

find_package(fbksd REQUIRED)

set(SRC
    src/utils.h
    src/main.cpp
    src/featurefilter.h
    src/featurefilter.cu
    src/film.h
    src/film.cpp
    src/imageio.cpp
    src/imageio.h
    src/multifilm.cpp
    src/multifilm.h
    src/nlm.cpp
    src/nlm.h
    src/NFOR.cpp
    src/NFOR.h
)

cuda_add_executable(NFOR ${SRC})
target_link_libraries(NFOR ${OPENEXR_LIBRARIES} fbksd::client)
install(TARGETS NFOR RUNTIME DESTINATION .)
install(FILES info.json DESTINATION .)
