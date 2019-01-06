cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)

project(Mitchell)

set(CMAKE_CXX_STANDARD 14)

find_package(fbksd REQUIRED)

add_executable(Mitchell src/main.cpp)
target_link_libraries(Mitchell fbksd::client)
install(TARGETS Mitchell RUNTIME DESTINATION .)
install(FILES info.json DESTINATION .)
