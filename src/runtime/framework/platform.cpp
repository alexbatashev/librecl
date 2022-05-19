#include "platform.hpp"

#include <CL/cl.h>
#include <algorithm>

static _cl_platform_id *gPlatforms = nullptr;
static unsigned gNumPlatforms = 0;

extern "C" {
cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms,
                        cl_uint *num_platforms) {
  if (num_platforms == nullptr && platforms == nullptr) {
    // log: Both platforms and num_platforms are NULL
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
      // log: num_entries is 0
      return CL_INVALID_VALUE;
    }

    for (unsigned i = 0; i < std::min(num_entries, gNumPlatforms); i++) {
      platforms[i] = &gPlatforms[i];
    }
  }

  return CL_SUCCESS;
}
}
