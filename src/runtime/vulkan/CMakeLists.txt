find_package(Vulkan REQUIRED)

add_librecl_runtime(lcl_icd_vulkan SHARED
  platform.cpp
  device.cpp
  context.cpp
  queue.cpp
  program.cpp
  vma.cpp
  # TODO make a single list
  ${CMAKE_CURRENT_SOURCE_DIR}/../framework/platform.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../framework/device.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../framework/context.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../framework/queue.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../framework/program.cpp
)


target_include_directories(lcl_icd_vulkan SYSTEM PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/vma
)

target_link_libraries(lcl_icd_vulkan PRIVATE
  Vulkan::Vulkan
  TBB::tbb
  range-v3
  lcl_compiler
  )

target_compile_definitions(lcl_icd_vulkan PRIVATE
  -DVULKAN_HPP_NO_EXCEPTIONS
)


# TODO automate examples generation
file(WRITE ${PROJECT_BINARY_DIR}/empty.cpp)
add_executable(platform_info_vk
  ${PROJECT_BINARY_DIR}/empty.cpp
  $<TARGET_OBJECTS:platform_info_obj>
)

set_target_properties(platform_info_vk
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/examples)
target_include_directories(platform_info_vk PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/OpenCL)
target_link_libraries(platform_info_vk PRIVATE lcl_icd_vulkan)



add_executable(vector_add_vk
  ${PROJECT_BINARY_DIR}/empty.cpp
  $<TARGET_OBJECTS:vector_add_obj>
)

set_target_properties(vector_add_vk
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/examples)
target_include_directories(vector_add_vk PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/OpenCL)
target_link_libraries(vector_add_vk PRIVATE lcl_icd_vulkan)
