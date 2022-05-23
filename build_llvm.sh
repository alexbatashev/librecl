#!/bin/bash

if "$CI"; then
  echo "enabling ccache for GitHub Actions"
  export CACHE_LOC="-DLLVM_CCACHE_DIR=\"$GITHUB_WORKSPACE/cache\" -DLLVM_CCACHE_BUILD=ON -DLLVM_CCACHE_MAXSIZE=\"5G\""
fi

cd third_party/llvm-project

mkdir -p build && cd build

# TODO uplift LLVM to include this revision
wget https://reviews.llvm.org/file/data/tcxogj4kn6pdholkebnb/PHID-FILE-uodbv2aluopojlwdnwkr/D126161.diff
cd ../mlir
git apply ../build/D126161.diff
cd ../build

cmake -GNinja -DLLVM_ENABLE_PROJECTS="mlir;clang;lld" \
  -DLLVM_TARGETS_TO_BUILD="host;AMDGPU;NVPTX" \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  $CACHE_LOC \
  ../llvm/

ninja
