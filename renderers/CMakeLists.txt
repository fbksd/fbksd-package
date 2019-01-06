cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)

include(ExternalProject)
function(add_renderer dir_name)
    ExternalProject_Add(${dir_name}
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${dir_name}
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
            -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/renderers/${ARGV1}
    )
    install(DIRECTORY ${CMAKE_BINARY_DIR}/renderers/${ARGV1} DESTINATION renderers USE_SOURCE_PERMISSIONS)
endfunction(add_renderer)


message("Renderers:")
file(GLOB children RELATIVE ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/*)
foreach(child ${children})
    if(IS_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/${child} AND NOT ${child} MATCHES "^\\..*$" AND NOT ${child} MATCHES "^_.*$")
        message(" - " ${child})
        add_renderer(${child} ${child})
    endif()
endforeach()

