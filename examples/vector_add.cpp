#include <CL/cl.h>

#include <array>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

constexpr size_t N = 10'000;

constexpr auto programSrc = R"(
__kernel void vectorAdd(__global int *a, __global int *b, __global int *c, size_t n) {
  int id = get_global_id(0);

  if (id < n) {
    c[id] = a[id] + b[id];
  }
}
)";

std::vector<int> getReference(const std::vector<int> &a,
                              const std::vector<int> &b) {
  std::vector<int> res;
  res.resize(a.size());

  for (size_t i = 0; i < a.size(); i++) {
    res[i] = a[i] + b[i];
  }

  return res;
}

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

void printPlatformDeviceInfo(cl_platform_id platform, cl_device_id device) {
  std::string platformName, deviceName;

  size_t size;

  check(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &size));
  platformName.resize(size);
  check(clGetPlatformInfo(platform, CL_PLATFORM_NAME, size, platformName.data(),
                          nullptr));

  check(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &size));
  deviceName.resize(size);
  check(clGetDeviceInfo(device, CL_DEVICE_NAME, size, deviceName.data(),
                        nullptr));

  std::cout << "Platform : " << platformName << "\n";
  std::cout << "Device : " << deviceName << "\n";
}

int main() {
  std::vector<int> a, b, res;
  a.resize(N);
  b.resize(N);
  res.resize(N);

  std::iota(a.begin(), a.end(), 0);
  std::iota(b.begin(), b.end(), 1000);

  auto reference = getReference(a, b);

  std::vector<cl_platform_id> platforms;

  unsigned numPlatforms = 0;

  check(clGetPlatformIDs(0, nullptr, &numPlatforms));

  if (numPlatforms == 0) {
    std::cout << "No platforms found\n";
    return 0;
  }

  platforms.resize(numPlatforms);
  check(clGetPlatformIDs(numPlatforms, platforms.data(), nullptr));

  cl_platform_id platform = platforms[0];

  unsigned numDevices;
  std::vector<cl_device_id> devices;

  check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices));
  devices.resize(numDevices);
  check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices.data(),
                       nullptr));

  cl_device_id device = devices[0];

  printPlatformDeviceInfo(platform, device);

  cl_int err;

  // TODO use callbacks
  cl_context context =
      clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
  check(err);

  cl_command_queue queue =
      clCreateCommandQueueWithProperties(context, device, nullptr, &err);
  check(err);

  cl_program program = clCreateProgramWithSource(
      context, 1, (const char **)&programSrc, nullptr, &err);
  check(err);

  /*
  check(clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr));

  cl_kernel kernel = clCreateKernel(program, "vectorAdd", &err);
  check(err);

  cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(int),
  nullptr, nullptr); cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, N *
  sizeof(int), nullptr, nullptr); cl_mem bufC = clCreateBuffer(context,
  CL_MEM_WRITE_ONLY, N * sizeof(int), nullptr, nullptr);

  check(clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, N * sizeof(int), a.data(),
  0, nullptr, nullptr)); check(clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, N *
  sizeof(int), b.data(), 0, nullptr, nullptr));

  check(clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA));
  check(clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB));
  check(clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC));
  check(clSetKernelArg(kernel, 0, sizeof(size_t), &N));

  std::array<size_t, 1> globalSize = { N };
  cl_event evt;
  check(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, globalSize.data(),
  nullptr, 0, nullptr, &evt));

  clFinish(queue);

  clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, N * sizeof(int), res.data(), 0,
  nullptr, nullptr);
  */

  // TODO release resources

  return 0;
}
