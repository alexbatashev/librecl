#!/bin/bash

Help() {
  echo "Available options:"
  echo " h - show this guide"
  echo " d - enable debug builds"
  echo " l - use LLD linker"
  echo
}

if "$CI"; then
  echo "enabling ccache for GitHub Actions"
  CACHE_LOC="-DLLVM_CCACHE_DIR=\"$GITHUB_WORKSPACE/cache\" -DLLVM_CCACHE_BUILD=ON -DLLVM_CCACHE_MAXSIZE=\"5G\""
  BUILD_TYPE=Release
  ASSERTIONS=OFF
  USE_LLD=OFF
else
  BUILD_TYPE=Release
  ASSERTIONS=ON
  USE_LLD=OFF
  while getopts ":hdl" options; do
    case "${options}" in
      h)
        Help()
        ;;
      d)
        BUILD_TYPE=Debug
        ;;
      l)
        USE_LLD=ON
        ;;
      \?)
        Help()
    esac
  done
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
  -DLLVM_ENABLE_ASSERTIONS=$ASSERTIONS \
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DLLVM_ENABLE_LLD=$USE_LLD \
  $CACHE_LOC \
  ../llvm/

ninja
