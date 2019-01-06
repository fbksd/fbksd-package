cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)

project(Stratified)

set(CMAKE_CXX_STANDARD 14)

find_package(fbksd REQUIRED)

add_executable(Stratified
    utils.h
    utils.cpp
    main.cpp
    random.h
    random.cpp
    sampler.h
    sampler.cpp
    stratified.h
)
target_link_libraries(Stratified fbksd::client)
install(TARGETS Stratified RUNTIME DESTINATION .)
install(FILES info.json DESTINATION .)
