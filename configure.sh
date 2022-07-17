#!/bin/bash

Help() {
  echo "Available options:"
  echo " h - show this guide"
  echo " d - enable debug builds"
  echo " l - use LLD linker"
  echo " m - use mold linker"
  echo
  exit 0
}

if "$CI"; then
  BUILD_TYPE=Release
  LINKER=mold
else
  BUILD_TYPE=Release
  LINKER=default
  while getopts ":hdlm" options; do
    case "${options}" in
      h)
        Help
        ;;
      d)
        BUILD_TYPE=Debug
        ;;
      l)
        LINKER=lld
        ;;
      m)
        LINKER=mold
        ;;
      \?)
        Help
    esac
  done
fi

mkdir -p build
cd build
cmake -GNinja \
  -DMLIR_DIR=$PWD/../third_party/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=$PWD/../third_party/llvm-project/build/lib/cmake/llvm \
  -DClang_DIR=$PWD/../third_party/llvm-project/build/lib/cmake/clang \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
  -DLLVM_ENABLE_TERMINFo=OFF \
  -DLIBRECL_LINKER="$LINKER" \
  ..
