#include <catch2/catch_test_macros.hpp>

#include <CL/cl.h>

#include <string>
#include <vector>
#include <iostream>

TEST_CASE("Some platforms are found", "[basic-platform]") {
  std::vector<cl_platform_id> platforms;
  cl_uint numPlatforms = 0;

  REQUIRE(clGetPlatformIDs(0, nullptr, &numPlatforms) == CL_SUCCESS);
  REQUIRE_FALSE(numPlatforms == 0);

  platforms.resize(numPlatforms);

  REQUIRE(clGetPlatformIDs(numPlatforms, platforms.data(), nullptr) ==
          CL_SUCCESS);
}

TEST_CASE("Platforms have correct names", "[basic-platform]") {
  std::vector<cl_platform_id> platforms;
  cl_uint numPlatforms = 0;

  REQUIRE(clGetPlatformIDs(0, nullptr, &numPlatforms) == CL_SUCCESS);
  REQUIRE_FALSE(numPlatforms == 0);

  platforms.resize(numPlatforms);

  REQUIRE(clGetPlatformIDs(numPlatforms, platforms.data(), nullptr) ==
          CL_SUCCESS);

  for (auto p : platforms) {
    size_t len = 0;
    REQUIRE(clGetPlatformInfo(p, CL_PLATFORM_NAME, 0, nullptr, &len) ==
            CL_SUCCESS);
    REQUIRE(len > 0);

    std::string platformName;
    platformName.resize(len);

    REQUIRE(clGetPlatformInfo(p, CL_PLATFORM_NAME, len, platformName.data(),
                              nullptr) == CL_SUCCESS);

    CHECK(platformName.starts_with("LibreCL"));
  }
}
