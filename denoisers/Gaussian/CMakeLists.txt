cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)

project(Gaussian)

set(CMAKE_CXX_STANDARD 14)

find_package(fbksd REQUIRED)

add_executable(Gaussian src/main.cpp)
target_link_libraries(Gaussian fbksd::client)
install(TARGETS Gaussian RUNTIME DESTINATION .)
install(FILES info.json DESTINATION .)
