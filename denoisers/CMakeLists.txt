cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)

include(ExternalProject)
function(add_filter dir_name)
    message(" - " ${dir_name})
    ExternalProject_Add(${dir_name}
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${dir_name}
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
            -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/denoisers/${dir_name}
    )
    install(DIRECTORY ${CMAKE_BINARY_DIR}/denoisers/${dir_name} DESTINATION denoisers USE_SOURCE_PERMISSIONS)
endfunction(add_filter)

message("Denoisers:")

add_filter(Box)
add_filter(Gaussian)
add_filter(GEM)
add_filter(Mitchell)
add_filter(RHF)
add_filter(RPF)
add_filter(SBF)

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    add_filter(LWR)
    add_filter(LBF)
    add_filter(NFOR)
    add_filter(NLM)
    add_filter(RDFC)
endif()
