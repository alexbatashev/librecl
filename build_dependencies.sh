#!/bin/bash

BASE=$(dirname -- "$(readlink -f "${BASH_SOURCE}")")

cd $BASE/third_party/llvm-project/

if [[ ! -d `git status --porcelain` ]]; then

  git apply ../llvm_patches/0001-mlir-spirv-Handle-nested-global-variable-references-.patch

fi

