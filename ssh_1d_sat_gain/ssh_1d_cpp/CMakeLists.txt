cmake_minimum_required(VERSION 3.5)

project(ssh_1d VERSION 0.1 LANGUAGES CXX)

set(GPROF_EXPORT OFF)

set(CMAKE_CXX_STANDARD 17)

set(PYBIND11_FINDPYTHON ON)
add_subdirectory(pybind11)

pybind11_add_module(ssh_1d main.cpp)
add_executable(ssh_1d_runner main.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
set(total_link_flags " ")
set(mingw_flags_for_python "-static-libgcc -static-libstdc++ -Wl,-Bstatic,--whole-archive -lwinpthread -Wl,--no-whole-archive") #https://cython.readthedocs.io/en/latest/src/tutorial/appendix.html


set_target_properties(ssh_1d PROPERTIES LIBRARY_OUTPUT_DIRECTORY "bin-module")
set_target_properties(ssh_1d_runner PROPERTIES RUNTIME_OUTPUT_DIRECTORY "bin-runner")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/eigen-3.4.0/")


if(WIN32 AND MINGW)
    set(total_link_flags "${total_link_flags} ${mingw_flags_for_python}")
endif()

if (GPROF_EXPORT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg -no-pie")
endif()

set_target_properties(ssh_1d PROPERTIES LINK_FLAGS "${total_link_flags}")
set_target_properties(ssh_1d_runner PROPERTIES LINK_FLAGS "${total_link_flags} -pg")

target_compile_features(ssh_1d PRIVATE cxx_std_17)
target_compile_features(ssh_1d_runner PRIVATE cxx_std_17)

target_link_libraries(ssh_1d_runner PUBLIC pybind11::pybind11 pybind11::module)
