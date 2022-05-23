#!/bin/bash

if [ $CI -eq true ]; then
  echo "enabling ccache for GitHub Actions"
  export CACHE_LOC=$GITHUB_WORKSPACE/cache
fi

cd third_party/llvm-project

mkdir -p build && cd build

cmake -GNinja -DLLVM_ENABLE_PROJECTS="mlir;clang;lld" \
  -DLLVM_TARGETS_TO_BUILD="host;AMDGPU;NVPTX" \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DLLVM_CCACHE_MAX_SIZE="8G" \
  $CACHE_LOC \
  -DLLVM_CCACHE_BUILD=ON ../llvm/

ninja
