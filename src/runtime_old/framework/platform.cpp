//===- platform.cpp ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "platform.hpp"
#include "info.hpp"
#include "log.hpp"
#include "ocl_api.hpp"

#include <CL/cl.h>
#include <algorithm>

static _cl_platform_id *gPlatforms = nullptr;
static unsigned gNumPlatforms = 0;

extern "C" {
cl_int LCL_API clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms,
                                cl_uint *num_platforms) {
  if (num_platforms == nullptr && platforms == nullptr) {
    log(LogLevel::Error, "Both platforms and num_platforms are NULL");
    return CL_INVALID_VALUE;
  }

  if (!gPlatforms) {
    _cl_platform_id::initialize(&gPlatforms, gNumPlatforms);
  }

  if (num_platforms) {
    *num_platforms = gNumPlatforms;
  }

  if (platforms) {
    if (num_entries == 0) {
      log(LogLevel::Error, "num_entries is 0");
      return CL_INVALID_VALUE;
    }

    for (unsigned i = 0; i < std::min(num_entries, gNumPlatforms); i++) {
      platforms[i] = &gPlatforms[i];
    }
  }

  return CL_SUCCESS;
}

cl_int LCL_API clGetPlatformInfo(cl_platform_id platform,
                                 cl_platform_info paramName,
                                 size_t paramValueSize, void *paramValue,
                                 size_t *paramValueSizeRet) {
  switch (paramName) {
  case CL_PLATFORM_PROFILE: {
    if (platform->isFullProfile()) {
      return setParamValueStr("FULL_PROFILE", paramValueSize, paramValue,
                              paramValueSizeRet);
    } else {
      return setParamValueStr("EMBEDDED_PROFILE", paramValueSize, paramValue,
                              paramValueSizeRet);
    }
  }
  case CL_PLATFORM_VERSION:
    return setParamValueStr("OpenCL 3.0", paramValueSize, paramValue,
                            paramValueSizeRet);
  case CL_PLATFORM_NUMERIC_VERSION:
    return setParamValue(CL_MAKE_VERSION(3, 0, 0), paramValueSize, paramValue,
                         paramValueSizeRet);
  case CL_PLATFORM_NAME:
    return setParamValueStr(platform->getName(), paramValueSize, paramValue,
                            paramValueSizeRet);
  case CL_PLATFORM_VENDOR:
    return setParamValueStr(platform->getVendorName(), paramValueSize,
                            paramValue, paramValueSizeRet);
  }

  log(LogLevel::Error,
      fmt::format("unsupported cl_platform_info value {}", paramName));
  return CL_INVALID_VALUE;
}
}
