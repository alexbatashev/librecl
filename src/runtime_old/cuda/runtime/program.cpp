#include "program.hpp"

#include "compiler.hpp"

extern "C" {
cl_program clCreateProgramWithSource(cl_context context, cl_uint count,
                                     const char **strings,
                                     const size_t *lengths,
                                     cl_int *errcode_ret) {
  // TODO error handling

  std::string fullSource;

  for (unsigned i = 0; i < count; i++) {
    if (lengths) {
      fullSource += std::string(strings[i], lengths[i]);
    } else {
      fullSource += std::string(strings[i]);
    }
  }

  cl_program program = new _cl_program(context, fullSource);
  *errcode_ret = CL_SUCCESS;
  return program;
}

cl_int clBuildProgram(cl_program program, cl_uint num_devices,
                      const cl_device_id *device_list, const char *options,
                      void(CL_CALLBACK *pfn_notify)(cl_program program,
                                                    void *user_data),
                      void *user_data) {

  std::string strOptions = "-triple nvptx64-cuda-cuda -cl-ext=all ";
  if (options) {
    strOptions += std::string(options) + " ";
  }

  std::vector<std::string_view> splitOptions;

  size_t lastPos = 0;
  while (lastPos < strOptions.size()) {
    size_t pos = strOptions.find(" ", lastPos);
    std::string_view opt{strOptions.begin() + lastPos,
                         strOptions.begin() + pos};
    if (!opt.empty()) {
      splitOptions.push_back(opt);
    }
    lastPos = pos + 1;
  }

  // TODO iterate over all devices
  std::visit(
      [&](const auto &source) {
        if constexpr (std::is_same_v<std::decay_t<decltype(source)>,
                                     std::string>) {
          program->mContext->mDevices[0]->mCompiler.addModuleFromSource(
              source, splitOptions);
        }
      },
      program->mSource);

  return CL_SUCCESS;
}
}
