if (NOT APPLE)
add_subdirectory(cuda)
endif()
add_subdirectory(cpu)
add_subdirectory(metal)

add_subdirectory(devicelib)
