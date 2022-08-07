#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "fixture.hpp"

#include <CL/cl.h>

#include <numeric>
#include <string>
#include <vector>
#include <array>

using Catch::Approx;

constexpr unsigned int N = 100;

inline constexpr auto vectorAddSimple = R"(
__kernel void vectorAdd(__global float *a, __global float *b, __global float *c) {
  int id = get_global_id(0);

  c[id] = a[id] + b[id];
}
)";

std::vector<float> getVectorAddReference(const std::vector<float> &a,
                                         const std::vector<float> &b) {
  std::vector<float> res;
  res.resize(a.size());

  for (size_t i = 0; i < a.size(); i++) {
    res[i] = a[i] + b[i];
  }

  return res;
}

TEST_CASE_METHOD(DeviceTest, "Simple kernel execution", "[basic-kernel]") {
  with_all([](cl_device_id device, cl_context context, cl_command_queue queue) {
    std::vector<float> a, b, res;
    a.resize(N);
    b.resize(N);
    res.resize(N);

    std::iota(a.begin(), a.end(), 0);
    std::iota(b.begin(), b.end(), 1000);
    auto reference = getVectorAddReference(a, b);

    cl_int err = CL_SUCCESS;
    
    cl_program program = clCreateProgramWithSource(
        context, 1, (const char **)&vectorAddSimple, nullptr, &err);
    REQUIRE(err == CL_SUCCESS);

    REQUIRE(clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr) == CL_SUCCESS);

    cl_kernel kernel = clCreateKernel(program, "vectorAdd", &err);
    REQUIRE(err == CL_SUCCESS);

    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(float),
                                 nullptr, &err);
    REQUIRE(err == CL_SUCCESS);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(float),
                                 nullptr, &err);
    REQUIRE(err == CL_SUCCESS);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(float),
                                 nullptr, &err);
    REQUIRE(err == CL_SUCCESS);

    REQUIRE(clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, N * sizeof(float),
                               a.data(), 0, nullptr, nullptr) == CL_SUCCESS);
    REQUIRE(clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, N * sizeof(float),
                               b.data(), 0, nullptr, nullptr) == CL_SUCCESS);

    REQUIRE(clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA) == CL_SUCCESS);
    REQUIRE(clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB) == CL_SUCCESS);
    REQUIRE(clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC) == CL_SUCCESS);

    std::array<size_t, 1> globalSize = {N};
    cl_event evt;
    REQUIRE(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, globalSize.data(),
                                 nullptr, 0, nullptr, &evt) == CL_SUCCESS);

    clFinish(queue);

    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, N * sizeof(int), res.data(), 0,
                        nullptr, nullptr);

    for (size_t i = 0; i < N; i++) {
      CHECK(res[i] == Approx(reference[i]));
    }
  });
}
