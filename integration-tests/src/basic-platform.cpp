#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <CL/cl.h>

TEST_CASE("Some platforms are found", "[basic-platform]") {
  std::vector<cl_platform_id> platforms;
  cl_uint numPlatforms = 0;

  REQUIRE(clGetPlatformIDs(0, nullptr, &numPlatforms) == CL_SUCCESS);
  REQUIRE_FALSE(numPlatforms == 0);

  platforms.resize(numPlatforms);

  REQUIRE(clGetPlatformIDs(numPlatforms, platforms.data(), nullptr) == CL_SUCCESS);
}
