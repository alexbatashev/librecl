list(APPEND CMAKE_MODULE_PATH
  "${PROJECT_SOURCE_DIR}/third_party/llvm-project/llvm/cmake/modules")

include(AddLLVM)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(LCL_TEST_DEPENDS
        FileCheck count not
        lcl-opt
        lcloc
        )

add_lit_testsuite(check-lcl "Running the standalone regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${LCL_TEST_DEPENDS}
  )
set_target_properties(check-lcl PROPERTIES FOLDER "Tests")
add_lit_testsuites(LCL ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${LCL_TEST_DEPENDS})

