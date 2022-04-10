#include <CL/cl.h>
#include <cuda.h>

#include <string_view>
#include <vector>

struct _cl_platform_id {
  std::vector<CUdevice> mDevices;
};

bool gInitialized = false;
_cl_platform_id *gPlatform;

cl_int translateError(CUresult res) {
  switch (res) {
  case CUDA_SUCCESS:
    return CL_SUCCESS;

  default:
    return CL_OUT_OF_RESOURCES;
  }
}

#define CHECK(call)                                                            \
  do {                                                                         \
    CUresult res = call;                                                       \
    if (res != CUDA_SUCCESS) {                                                 \
      cl_int err = translateError(res);                                        \
      return err;                                                              \
    }                                                                          \
  } while (0)

extern "C" {
cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms,
                        cl_uint *num_platforms) {
  if (num_platforms == nullptr && platforms == nullptr) {
    // log: Both platforms and num_platforms are NULL
    return CL_INVALID_VALUE;
  }

  if (num_platforms) {
    *num_platforms = 1;
  }

  if (platforms) {
    if (num_entries == 0) {
      // log: num_entries is 0
      return CL_INVALID_VALUE;
    }

    if (!gInitialized) {
      CHECK(cuInit(0));

      int numDevices = 0;
      CHECK(cuDeviceGetCount(&numDevices));

      gPlatform = new _cl_platform_id{};

      gPlatform->mDevices.reserve(numDevices);

      for (int i = 0; i < numDevices; i++) {
        CUdevice device;
        CUresult res = cuDeviceGet(&device, i);
        if (res == CUDA_SUCCESS) {
          gPlatform->mDevices.push_back(device);
        }
        // else log: error
      }
    }

    platforms[0] = gPlatform;
  }

  return CL_SUCCESS;
}

cl_int clGetPlatformInfo(cl_platform_id platform, cl_platform_info param_name,
                         size_t param_value_size, void *param_value,
                         size_t *param_value_size_ret) {

  if (platform != gPlatform) {
    // log: meaningful error
    return CL_INVALID_PLATFORM;
  }

  if (param_value == nullptr && param_value_size_ret == nullptr) {
    // log: param_value and param_value_size_ret are NULL
    return CL_INVALID_VALUE;
  }

  const auto setSVInfo = [](std::string_view SV, void *paramValue,
                            size_t *size) {
    if (size) {
      *size = SV.size();
    }
    if (paramValue) {
      *static_cast<const char **>(paramValue) = SV.data();
    }
  };

  switch (param_name) {
  case CL_PLATFORM_PROFILE: {
    std::string_view profile = "FULL_PROFILE";
    setSVInfo(profile, param_value, param_value_size_ret);
    break;
  }
  case CL_PLATFORM_VERSION: {
    std::string_view profile = "OpenCL 3.0";
    setSVInfo(profile, param_value, param_value_size_ret);
    break;
  }
  case CL_PLATFORM_NAME: {
    std::string_view name = "LibreCL over NVIDIA CUDA";
    setSVInfo(name, param_value, param_value_size_ret);
    break;
  }
  case CL_PLATFORM_VENDOR: {
    std::string_view vendor = "NVIDIA";
    setSVInfo(vendor, param_value, param_value_size_ret);
    break;
  }
  case CL_PLATFORM_EXTENSIONS: {
    std::string_view ext = "";
    setSVInfo(ext, param_value, param_value_size_ret);
    break;
  }
  default:
    // log: invalid or unsupported param_name
    return CL_INVALID_VALUE;
  }

  return CL_SUCCESS;
}
}
