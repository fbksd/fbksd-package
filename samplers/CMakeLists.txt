cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)

include(ExternalProject)
function(add_technique dir_name)
    ExternalProject_Add(${dir_name}
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${dir_name}
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
            -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/samplers/${dir_name}
    )
    install(DIRECTORY ${CMAKE_BINARY_DIR}/samplers/${dir_name} DESTINATION samplers USE_SOURCE_PERMISSIONS)
endfunction(add_technique)


message("Samplers:")
file(GLOB children RELATIVE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/*)
foreach(child ${children})
    if(IS_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/${child} AND NOT ${child} MATCHES "^\\..*$")
        message(" - " ${child})
        add_technique(${child})
    endif()
endforeach()

