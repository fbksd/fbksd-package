cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)

project(AdptBox)

set(CMAKE_CXX_STANDARD 14)

find_package(fbksd REQUIRED)

add_executable(AdptBox main.cpp)
target_link_libraries(AdptBox fbksd::client)
install(TARGETS AdptBox RUNTIME DESTINATION .)
install(FILES info.json DESTINATION .)

