cmake_minimum_required(VERSION 3.13.4)
project(HelloWorldMLIR LANGUAGES CXX C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)


set(LT_LLVM_INSTALL_DIR "/home/mlevental/dev_projects/llvm-project/install" CACHE PATH "LLVM installation directory")

# Add the location of LLVMConfig.cmake to CMake search paths (so that
# find_package can locate it)
set(LLVM_DIR "${LT_LLVM_INSTALL_DIR}/lib/cmake/llvm/")
set(MLIR_DIR "${LT_LLVM_INSTALL_DIR}/lib/cmake/mlir/")

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(AddLLVM)
include(AddMLIR)

add_llvm_pass_plugin(HelloWorldMLIR HelloWorldMLIR.cpp)
