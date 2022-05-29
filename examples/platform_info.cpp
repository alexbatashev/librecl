#include <CL/cl.h>

#include <iostream>
#include <vector>

void check(cl_int err) {
  if (err == CL_SUCCESS)
    return;

  if (err == CL_INVALID_VALUE) {
    std::cerr << "CL_INVALID_VALUE\n";
    std::terminate();
  } else if (err == CL_INVALID_DEVICE) {
    std::cerr << "CL_INVALID_DEVICE\n";
    std::terminate();
  }
  std::cerr << "Unknown error\n";
  std::terminate();
}

void printSingleDevice(cl_device_id device) {
  std::string deviceName;

  size_t size = 0;

  check(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &size));
  deviceName.resize(size);
  check(clGetDeviceInfo(device, CL_DEVICE_NAME, size, deviceName.data(),
                        nullptr));

  std::cout << "    Name: " << deviceName << "\n";
}

void printDevicesInfo(cl_platform_id platform) {
  unsigned numDevices;
  std::vector<cl_device_id> devices;

  check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices));
  devices.resize(numDevices);
  check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices.data(),
                       nullptr));

  std::cout << "  Devices: " << numDevices << "\n\n";

  for (auto dev : devices)
    printSingleDevice(dev);
}

int main() {
  std::vector<cl_platform_id> platforms;

  unsigned numPlatforms = 0;

  check(clGetPlatformIDs(0, nullptr, &numPlatforms));

  if (numPlatforms == 0) {
    std::cout << "No platforms found\n";
    return 0;
  }

  platforms.resize(numPlatforms);
  check(clGetPlatformIDs(numPlatforms, platforms.data(), nullptr));

  std::cout << "Found " << numPlatforms << " platforms:\n";

  for (auto plt : platforms) {
    std::string platformName;
    std::string vendorName;
    std::string profile;

    size_t size;

    check(clGetPlatformInfo(plt, CL_PLATFORM_NAME, 0, nullptr, &size));
    platformName.resize(size);
    check(clGetPlatformInfo(plt, CL_PLATFORM_NAME, size, platformName.data(),
                            nullptr));

    check(clGetPlatformInfo(plt, CL_PLATFORM_VENDOR, 0, nullptr, &size));
    vendorName.resize(size);
    check(clGetPlatformInfo(plt, CL_PLATFORM_VENDOR, size, vendorName.data(),
                            nullptr));

    check(clGetPlatformInfo(plt, CL_PLATFORM_PROFILE, 0, nullptr, &size));
    profile.resize(size);
    check(clGetPlatformInfo(plt, CL_PLATFORM_PROFILE, size, profile.data(),
                            nullptr));

    std::cout << "  Name: " << platformName << "\n";
    std::cout << "  Vendor: " << vendorName << "\n";
    std::cout << "  Profile: " << profile << "\n";

    printDevicesInfo(plt);
  }

  return 0;
}
