#pragma once

#include "Foundation/NSArray.hpp"
#include "Metal/MTLDevice.hpp"
#include <CL/cl.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <vector>

struct _cl_platform_id {
  static void initialize(_cl_platform_id **platforms, unsigned &numPlatforms) {
    numPlatforms = 1;
    *platforms = new _cl_platform_id[1];
  }

  _cl_platform_id() {
    NS::Array *allDevices = MTL::CopyAllDevices();
    mDevices.reserve(allDevices->count());
    for (unsigned i = 0; i < allDevices->count(); i++) {
      mDevices.push_back(static_cast<MTL::Device*>(allDevices->object(i)));
    }
  }

  std::vector<MTL::Device*> mDevices;
};
