#!/bin/sh

mkdir -p build
cd build
cmake -GNinja \
  -DMLIR_DIR=$PWD/../third_party/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=$PWD/../third_party/llvm-project/build/lib/cmake/llvm \
  -DClang_DIR=$PWD/../third_party/llvm-project/build/lib/cmake/clang \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DCMAKE_BUILD_TYPE=Debug \
  ..
