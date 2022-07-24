//===- rust_bindings.cpp ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "frontend.hpp"
#include "visibility.hpp"
#include "ClangFrontend.hpp"

#include <string_view>
#include <vector>

extern "C" {
LCL_COMP_EXPORT lcl::ClangFrontend * create_clang_frontend() {
  return new lcl::ClangFrontend();
}
LCL_COMP_EXPORT void release_clang_frontend(lcl::ClangFrontend *fe) {
  delete fe;
}

LCL_COMP_EXPORT lcl::FrontendResult *process_source(lcl::Frontend *fe, const char *source, const char **options, size_t num_options) {
  std::vector<std::string_view> safe_options;
  for (size_t i = 0; i < num_options; i++) {
    safe_options.push_back(std::string_view{options[i]});
  }
  return new lcl::FrontendResult(std::move(fe->process(std::string_view{source}, safe_options)));
}

LCL_COMP_EXPORT void release_result(lcl::FrontendResult *fe_res) {
  delete fe_res;
}
}
