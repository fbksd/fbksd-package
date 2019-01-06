cmake_minimum_required(VERSION 3.9.0 FATAL_ERROR)
project(fbksd-package)

find_package(fbksd REQUIRED)

add_subdirectory(renderers)
add_subdirectory(denoisers)
add_subdirectory(samplers)
