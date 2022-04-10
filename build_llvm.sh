#!/bin/sh

cd third_party/llvm-project

mkdir build && cd build

cmake -GNinja -DLLVM_ENABLE_PROJECTS="mlir;clang;compiler-rt;libclc" \
  -DLLVM_TARGETS_TO_BUILD="host;AMDGPU;NVPTX" \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DLLVM_ENABLE_CCACHE=ON ../llvm/

ninja
