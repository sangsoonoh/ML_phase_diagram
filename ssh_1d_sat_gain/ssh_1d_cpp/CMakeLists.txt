cmake_minimum_required(VERSION 3.5)


project(ssh_1d VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(PYBIND11_FINDPYTHON ON)
add_subdirectory(pybind11)

pybind11_add_module(ssh_1d main.cpp)
add_executable(ssh_1d_runner main.cpp)

set_target_properties(ssh_1d PROPERTIES LIBRARY_OUTPUT_DIRECTORY "bin-module")
set_target_properties(ssh_1d_runner PROPERTIES RUNTIME_OUTPUT_DIRECTORY "bin-runner")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/eigen-3.4.0/")


if(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    #https://cython.readthedocs.io/en/latest/src/tutorial/appendix.html
    set_target_properties(ssh_1d PROPERTIES LINK_FLAGS "-static-libgcc -static-libstdc++ -Wl,-Bstatic,--whole-archive -lwinpthread -Wl,--no-whole-archive")
    set_target_properties(ssh_1d_runner PROPERTIES LINK_FLAGS "-static-libgcc -static-libstdc++ -Wl,-Bstatic,--whole-archive -lwinpthread -Wl,--no-whole-archive")
endif()

# target_link_libraries(ssh_1d PUBLIC Eigen3::Eigen)
target_link_libraries(ssh_1d_runner PUBLIC 
    pybind11::pybind11
    pybind11::module)

target_compile_features(ssh_1d PRIVATE cxx_std_17)
target_compile_features(ssh_1d_runner PRIVATE cxx_std_17)