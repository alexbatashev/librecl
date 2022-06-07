find_package(Vulkan REQUIRED)

add_librecl_runtime(lcl_icd_vulkan SHARED
  platform.cpp
  device.cpp
  context.cpp
  queue.cpp
  program.cpp
  kernel.cpp
  memory.cpp
  command.cpp
  vma.cpp
  ${FRAMEWORK_SOURCES}
)


target_include_directories(lcl_icd_vulkan SYSTEM PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/vma
  ${CMAKE_CURRENT_SOURCE_DIR}/../
)

target_link_libraries(lcl_icd_vulkan PRIVATE
  Vulkan::Vulkan
  TBB::tbb
  range-v3
  lcl_compiler
  )

setup_runtime_examples(vk lcl_icd_vulkan)

