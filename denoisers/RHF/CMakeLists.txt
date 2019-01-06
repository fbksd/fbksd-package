cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)

project(RHF)

set(CMAKE_CXX_STANDARD 14)

find_package(fbksd REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# OpenEXR
include_directories(/usr/include/OpenEXR)
set(OPENEXR_LIBRARIES IlmImf Half)

set(SRC
    io_exr.cpp
    io_exr.h
    pbrt.h
    libauxiliar.cpp
    libauxiliar.h
    libdenoising.cpp
    libdenoising.h
    mitchell.cpp
    mitchell.h
    imageHisto.cpp
    imageHisto.h
    rhf.cpp
)

add_executable(RHF ${SRC})
target_link_libraries(RHF ${OPENEXR_LIBRARIES} fbksd::client)
install(TARGETS RHF RUNTIME DESTINATION .)
install(FILES info.json DESTINATION .)
