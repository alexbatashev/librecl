//===- rust_bindings.hpp ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "visibility.hpp"
#include "Options.h"

#include <stddef.h>

namespace lcl {
class Compiler;
class CompileResult;
} // namespace lcl

extern "C" {
LCL_COMP_EXPORT lcl::Compiler *lcl_get_compiler();
LCL_COMP_EXPORT void lcl_release_compiler(lcl::Compiler *);

LCL_COMP_EXPORT lcl::CompileResult *
lcl_compile(lcl::Compiler *compiler, size_t sourceLen, const char *source,
            Options options);
LCL_COMP_EXPORT lcl::CompileResult *
lcl_link(lcl::Compiler *compiler, size_t numModules, lcl::CompileResult **,
         Options options);

LCL_COMP_EXPORT void lcl_release_result(lcl::CompileResult *result);

LCL_COMP_EXPORT size_t lcl_get_num_kernels(lcl::CompileResult *result);
LCL_COMP_EXPORT const char *lcl_get_kernel_name(lcl::CompileResult *,
                                                size_t index);
LCL_COMP_EXPORT size_t lcl_get_num_kernel_args(lcl::CompileResult *result,
                                               size_t index);
LCL_COMP_EXPORT void lcl_get_kernel_args(lcl::CompileResult *, size_t index,
                                         void *dst);

LCL_COMP_EXPORT size_t lcl_get_program_size(lcl::CompileResult *);
LCL_COMP_EXPORT void lcl_copy_program(lcl::CompileResult *, void *dst);

LCL_COMP_EXPORT int lcl_is_error(lcl::CompileResult *);
LCL_COMP_EXPORT const char *lcl_get_error_message(lcl::CompileResult *);
}
