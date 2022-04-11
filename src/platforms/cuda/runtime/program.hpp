#pragma once

#include "context.hpp"
#include "device.hpp"

#include <mutex>
#include <unordered_map>
#include <variant>
#include <vector>

#include <cuda.h>

struct BinaryProgram {
  std::vector<unsigned char> mData;
};

struct SPIRVProgram {
  std::vector<unsigned char> mData;
};

struct _cl_program {
  _cl_program(cl_context ctx, std::string source)
      : mContext(ctx), mSource(source) {}
  cl_context mContext;
  using ProgramSource = std::variant<std::string, BinaryProgram, SPIRVProgram>;
  ProgramSource mSource;

  std::unordered_map<cl_device_id, BinaryProgram &> mPrograms;
  std::unordered_map<cl_device_id, CUmodule> mModules;
  std::vector<BinaryProgram> mRawPrograms;
};
