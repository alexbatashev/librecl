#!/bin/sh

cd third_party/llvm-project

mkdir -p build && cd build

cmake -GNinja -DLLVM_ENABLE_PROJECTS="mlir;clang;compiler-rt;libclc;lld" \
  -DLLVM_TARGETS_TO_BUILD="host;AMDGPU;NVPTX" \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DLIBCLC_TARGETS_TO_BUILD="nvptx64--;nvptx64--nvidiacl;amdgcn--;amdgcn--amdhsa" \
  -DLLVM_CCACHE_BUILD=ON ../llvm/

ninja
