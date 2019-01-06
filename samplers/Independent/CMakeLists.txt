cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)

project(Independent)

set(CMAKE_CXX_STANDARD 14)

find_package(fbksd REQUIRED)

add_executable(Independent main.cpp)
target_link_libraries(Independent fbksd::client)
install(TARGETS Independent RUNTIME DESTINATION .)
install(FILES info.json DESTINATION .)
