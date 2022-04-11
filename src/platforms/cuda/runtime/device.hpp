#pragma once

#include "compiler.hpp"

#include <CL/cl.h>
#include <cuda.h>

struct _cl_device_id {
  _cl_device_id(CUdevice device, cl_device_type type)
      : mDevice(device), mType(type) {}
  CUdevice mDevice;
  cl_device_type mType;

  lcl::Compiler mCompiler{lcl::BuildTarget::NVPTX};
};
