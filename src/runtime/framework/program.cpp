#include "program.hpp"

extern "C" {

cl_program clCreateProgramWithSource(cl_context context, cl_uint count,
                                     const char **strings,
                                     const size_t *lengths,
                                     cl_int *errcode_ret) {
  if (!context) {
    *errcode_ret = CL_INVALID_CONTEXT;
    return nullptr;
  }

  if (count == 0) {
    context->notifyError("count is zero");
    *errcode_ret = CL_INVALID_VALUE;
    return nullptr;
  }

  if (strings == nullptr) {
    context->notifyError("strings is NULL");
    *errcode_ret = CL_INVALID_VALUE;
    return nullptr;
  }

  std::string program;
  for (size_t i = 0; i < count; i++) {
    if (strings[i] == nullptr) {
      context->notifyError("strings[" + std::to_string(i) + "] is NULL");
      *errcode_ret = CL_INVALID_VALUE;
      return nullptr;
    }

    if (lengths && lengths[i] > 0) {
      program += std::string{strings[i], lengths[i]};
    } else {
      program += std::string{strings[i]};
    }
  }

  return new _cl_program(context, program);
}
}
