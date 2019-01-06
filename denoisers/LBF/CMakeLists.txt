cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)

project(LBF)

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
find_package(CUDA REQUIRED)
# fixes some compile errors (probably Cuda bugs on Ubuntu 16.04)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES")

include_directories(src)
set(SRC src/LBF/CudaFeatureFuncs.cu
        src/LBF/CudaWrappers.cuh
        src/LBF/FeatureExtractor.cpp
        src/LBF/FeatureExtractor.h
        src/LBF/Globals.h
        src/LBF/LBF.cpp
        src/LBF/Matrix.h
        src/LBF/NeuralNetwork.cpp
        src/LBF/NeuralNetwork.h
        src/LBF/Utilities.cpp
        src/LBF/Utilities.h
        src/SampleWriter/Globals.h
        src/SampleWriter/SampleWriter.cpp
        src/SampleWriter/SampleWriter.h
)

set(LBF_SRC src/main.cpp)
set(LBF_MF_SRC src/LBF-mf.cpp)

cuda_add_library(lbf-core STATIC ${SRC})
cuda_add_cublas_to_target(lbf-core)
target_link_libraries(lbf-core fbksd::client)

cuda_add_executable(LBF ${LBF_SRC})
target_link_libraries(LBF lbf-core)
install(TARGETS LBF RUNTIME DESTINATION .)
install(FILES data/FeatureNorm.dat data/Weights.dat DESTINATION .)
install(FILES info.json DESTINATION .)

cuda_add_executable(LBF-mf ${LBF_MF_SRC})
target_link_libraries(LBF-mf lbf-core)
install(TARGETS LBF-mf RUNTIME DESTINATION .)

add_custom_command(TARGET LBF POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${PROJECT_SOURCE_DIR}/data/FeatureNorm.dat ${PROJECT_SOURCE_DIR}/data/Weights.dat
    $<TARGET_FILE_DIR:LBF>
)
