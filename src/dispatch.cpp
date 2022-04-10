#include <CL/cl.h>

#include <array>
#include <mutex>

// TODO replace with spinlock
std::mutex GPlatformCacheMutex;

struct DispatchTable {
#define _OCL_API(name) \
  decltype(name) *name;

#include "apis.def"
#undef _OCL_API
};

struct Platform {
  void *mLibrary;
  DispatchTable mTable;
};

bool PlatformsInitialized = false;
std::array<Platform, 1> GPlatforms;

extern "C" {
cl_int clGetPlatformIDs (cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms) {
  if (!PlatformsInitialized) {
  }

  return CL_SUCCESS;
}
}
