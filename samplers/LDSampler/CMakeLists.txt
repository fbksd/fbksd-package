cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)

project(LDSampler)

set(CMAKE_CXX_STANDARD 14)

find_package(fbksd REQUIRED)

add_executable(LDSampler
    utils.h
    utils.cpp
    qmc.h
    qmc.cpp
    main.cpp
    random.h
    random.cpp
    sampler.h
    sampler.cpp
    ldsampler.h
)
target_link_libraries(LDSampler fbksd::client)
install(TARGETS LDSampler RUNTIME DESTINATION .)
install(FILES info.json DESTINATION .)