if (NOT APPLE)
  add_subdirectory(cuda)
endif()

if (APPLE)
  add_subdirectory(metal)
endif()
