#pragma once

#include "context.hpp"

#include <span>
#include <string_view>
#include <variant>

struct SourceProgram {
  std::string source;
};

struct OclSPIRVProgram {};

using program_source_t =
    std::variant<std::monostate, SourceProgram, OclSPIRVProgram>;

struct _cl_program {
  _cl_program(cl_context ctx, std::string_view program);

private:
  cl_context mContext;
  program_source_t mProgramSource;
};
