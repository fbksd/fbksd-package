cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)

project(Box)

set(CMAKE_CXX_STANDARD 14)

find_package(fbksd REQUIRED)

add_executable(Box main.cpp)
target_link_libraries(Box fbksd::client)
install(TARGETS Box RUNTIME DESTINATION .)
install(FILES info.json DESTINATION .)

