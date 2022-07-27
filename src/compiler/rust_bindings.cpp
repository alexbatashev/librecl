//===- rust_bindings.cpp ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangFrontend.hpp"
#include "VulkanBackend.hpp"
#include "backend.hpp"
#include "frontend.hpp"
#include "include/visibility.hpp"
#include "kernel_info.hpp"
#include "visibility.hpp"

#include <cstring>
#include <string_view>
#include <vector>

extern "C" {
LCL_COMP_EXPORT lcl::ClangFrontend *create_clang_frontend() {
  return new lcl::ClangFrontend();
}
LCL_COMP_EXPORT void release_clang_frontend(lcl::ClangFrontend *fe) {
  delete fe;
}
LCL_COMP_EXPORT lcl::VulkanBackend *create_vulkan_backend() {
  return new lcl::VulkanBackend();
}
LCL_COMP_EXPORT void release_vulkan_backend(lcl::VulkanBackend *be) {
  delete be;
}

LCL_COMP_EXPORT lcl::FrontendResult *process_source(lcl::Frontend *fe,
                                                    const char *source,
                                                    const char **options,
                                                    size_t num_options) {
  std::vector<std::string_view> safe_options;
  for (size_t i = 0; i < num_options; i++) {
    safe_options.push_back(std::string_view{options[i]});
  }
  return new lcl::FrontendResult(
      std::move(fe->process(std::string_view{source}, safe_options)));
}

LCL_COMP_EXPORT void release_result(lcl::FrontendResult *fe_res) {
  delete fe_res;
}

LCL_COMP_EXPORT int result_is_ok(const lcl::FrontendResult *res) {
  return res->success();
}

LCL_COMP_EXPORT const char *result_get_error(const lcl::FrontendResult *res) {
  return res->error().c_str();
}
LCL_COMP_EXPORT lcl::BinaryProgram *backend_compile(lcl::Backend *backend,
                                                    lcl::FrontendResult *res) {
  return new lcl::BinaryProgram(std::move(backend->compile(*res)));
}
LCL_COMP_EXPORT size_t
binary_program_get_num_kernels(lcl::BinaryProgram *prog) {
  return prog->kernels.size();
}
LCL_COMP_EXPORT size_t
binary_program_get_num_kernel_args(lcl::BinaryProgram *prog, size_t idx) {
  return prog->kernels[idx].arguments.size();
}
LCL_COMP_EXPORT void binary_program_get_kernel_args(lcl::BinaryProgram *prog,
                                                    size_t index, void *dst) {
  std::memcpy(dst, prog->kernels[index].arguments.data(),
              prog->kernels[index].arguments.size() *
                  sizeof(lcl::ArgumentInfo));
}
LCL_COMP_EXPORT size_t binary_program_get_size(lcl::BinaryProgram *prog) {
  return prog->binary.size();
}
LCL_COMP_EXPORT void binary_program_copy_data(lcl::BinaryProgram *prog,
                                              void *dst) {
  std::memcpy(dst, prog->binary.data(), prog->binary.size());
}
LCL_COMP_EXPORT const char *
binary_program_get_kernel_name(lcl::BinaryProgram *prog, size_t index) {
  return prog->kernels[index].kernelName.c_str();
}
}
