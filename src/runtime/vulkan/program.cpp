#include "program.hpp"

_cl_program::_cl_program(cl_context ctx, std::string_view program)
    : mContext(ctx) {
  mProgramSource = SourceProgram{std::string{program}};
}
