//===- rust_bindings.cpp ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "rust_bindings.hpp"
#include "Compiler.hpp"
#include "kernel_info.hpp"
#include "visibility.hpp"

#include <atomic>
#include <cstring>
#include <string_view>
#include <vector>

lcl::Compiler *gCompiler = nullptr;
std::atomic<int> gRefCount{0};

extern "C" {
LCL_COMP_EXPORT lcl::Compiler *lcl_get_compiler() {
  // TODO this really better be thread-safe
  if (gCompiler == nullptr) {
    gCompiler = new lcl::Compiler();
  }

  gRefCount++;
  return gCompiler;
}

LCL_COMP_EXPORT void lcl_release_compiler(lcl::Compiler *compiler) {
  // TODO same here, not thread-safe
  if (gRefCount.fetch_sub(1) == 1) {
    delete gCompiler;
  }
}

LCL_COMP_EXPORT lcl::CompileResult *
lcl_compile(lcl::Compiler *compiler, size_t sourceLen, const char *source,
            Options options) {
  auto result = compiler->compile(std::span{source, sourceLen}, options);

  return new lcl::CompileResult(std::move(result));
}

LCL_COMP_EXPORT lcl::CompileResult *
lcl_link(lcl::Compiler *compiler, size_t numModules,
         lcl::CompileResult **results, Options options) {
  auto result =
      compiler->compile(std::span{results, numModules}, options);

  return new lcl::CompileResult(std::move(result));
}

LCL_COMP_EXPORT void lcl_release_result(lcl::CompileResult *result) {
  delete result;
}

LCL_COMP_EXPORT size_t lcl_get_num_kernels(lcl::CompileResult *result) {
  if (!result->hasBinary())
    return 0;

  auto &binary = result->getBinary();
  return binary.kernels.size();
}

LCL_COMP_EXPORT const char *lcl_get_kernel_name(lcl::CompileResult *result,
                                                size_t index) {
  auto &binary = result->getBinary();

  return binary.kernels[index].kernelName.c_str();
}

LCL_COMP_EXPORT size_t lcl_get_num_kernel_args(lcl::CompileResult *result,
                                               size_t index) {
  auto &binary = result->getBinary();

  return binary.kernels[index].arguments.size();
}
LCL_COMP_EXPORT void lcl_get_kernel_args(lcl::CompileResult *result,
                                         size_t index, void *dst) {
  auto &binary = result->getBinary();

  std::memcpy(dst, binary.kernels[index].arguments.data(),
              binary.kernels[index].arguments.size() *
                  sizeof(lcl::ArgumentInfo));
}

LCL_COMP_EXPORT size_t lcl_get_program_size(lcl::CompileResult *result) {
  auto &binary = result->getBinary();

  return binary.binary.size();
}
LCL_COMP_EXPORT void lcl_copy_program(lcl::CompileResult *result, void *dst) {
  auto &binary = result->getBinary();

  std::memcpy(dst, binary.binary.data(), binary.binary.size());
}

LCL_COMP_EXPORT int lcl_is_error(lcl::CompileResult *result) {
  return result->isError();
}

LCL_COMP_EXPORT const char *lcl_get_error_message(lcl::CompileResult *result) {
  return result->getError().c_str();
}
}
