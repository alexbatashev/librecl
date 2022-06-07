add_subdirectory(framework)

foreach(rt IN LISTS LIBRECL_BACKENDS)
  add_subdirectory(${rt})
endforeach()
