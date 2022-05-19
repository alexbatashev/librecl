set(SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/index.metal
)
add_custom_target(devicelib_air_macosx ALL
  xcrun -sdk macosx metal -emit-llvm -c
    ${SRC}
    -o ${PROJECT_BINARY_DIR}/devicelib_air_macosx.air
    BYPRODUCTS
    ${PROJECT_BINARY_DIR}/devicelib_air_macosx.air
    SOURCES
    ${SRC}
  )

FileEmbedAdd(
  ${PROJECT_BINARY_DIR}/devicelib_air_macosx.air
)
