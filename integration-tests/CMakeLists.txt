cmake_minimum_required (VERSION 3.17)

project(librecl-integration-tests VERSION 0.1 LANGUAGES C CXX)

cmake_policy(SET CMP0042 NEW)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.1.0
)

FetchContent_MakeAvailable(Catch2)

enable_testing()

add_executable(integration-tests
  src/basic-platform.cpp
  src/basic-kernel.cpp
)

add_test(NAME integration COMMAND integration-tests)

target_link_directories(integration-tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../target/debug/
)
target_include_directories(integration-tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/OpenCL-Headers
  )
target_link_libraries(integration-tests PRIVATE
  Catch2::Catch2WithMain
  lcl_icd_runtime
  )

