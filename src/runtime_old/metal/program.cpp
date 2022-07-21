//===- program.cpp ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/cl.h>

#include "frontend.hpp"
#include "program.hpp"

#include <optional>
#include <variant>

template <class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

_cl_program::_cl_program(cl_context ctx, std::string_view program)
    : mContext(ctx) {
  mProgramSource = SourceProgram{std::string{program}};
}

void _cl_program::build(std::span<const cl_device_id> devices,
                        std::span<std::string_view> options,
                        std::optional<callback_t> callback) {
  const auto build = [this, devices]() -> cl_int {
    lcl::FrontendResult res = std::visit(
        overloaded{[this](SourceProgram prog) {
                     return mContext->getClangFE().process(prog.source, {});
                   },
                   [](auto prog) { return lcl::FrontendResult(""); }},
        mProgramSource);

    if (!res.success()) {
      mContext->notifyError(res.error());
      return CL_BUILD_PROGRAM_FAILURE;
    }

    std::vector<unsigned char> msl = mContext->getMetalBE().compile(res);

    NS::String *source =
        NS::String::string(reinterpret_cast<const char *>(msl.data()),
                           NS::StringEncoding::ASCIIStringEncoding);
    for (auto dev : devices) {
      NS::Error *err;
      mCompiledProgram[dev] =
          dev->getNativeDevice()->newLibrary(source, nullptr, &err);
      if (err) {
        err->release();
        return CL_BUILD_PROGRAM_FAILURE;
      }
    }
    source->release();

    return CL_SUCCESS;
  };

  if (!callback.has_value()) {
    build();
  }
}
