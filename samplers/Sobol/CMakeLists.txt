cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)

project(Sobol)

set(CMAKE_CXX_STANDARD 14)

find_package(fbksd REQUIRED)

add_executable(Sobol
    utils.h
    utils.cpp
    qmc.h
    qmc.cpp
    main.cpp
    random.h
    random.cpp
    sampler.h
    sampler.cpp
    sobolseq.h
    sobolseq.cpp
    sobol.h
)
target_link_libraries(Sobol fbksd::client)
install(TARGETS Sobol RUNTIME DESTINATION .)
install(FILES info.json DESTINATION .)
