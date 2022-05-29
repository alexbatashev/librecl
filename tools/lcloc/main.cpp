#include "ClangFrontend.hpp"
#include "MetalBackend.hpp"
#include "compiler.hpp"

const char *kernelSource =
    "\n"
    // "void foo() {}                                                    \n"
    "__kernel void vecAdd(  __global float *a,                       \n"
    "                       __global float *b,                       \n"
    "                       __global float *c,                       \n"
    "                       const unsigned int n)                    \n"
    "{                                                               \n"
    "    //Get our global thread ID                                  \n"
    "    int id = get_global_id(0);                                  \n"
    "                                                                \n"
    "    //Make sure we do not go out of bounds                      \n"
    "    if (id < n)                                                 \n"
    "        c[id] = a[id] + b[id];                                  \n"
    // "    foo();                                                      \n"
    "}                                                               \n"
    "\n";

int main() {
  // lcl::Compiler compiler(lcl::BuildTarget::NVPTX);
  // compiler.addModuleFromSource(kernelSource, {});
  lcl::ClangFrontend FE;
  auto IR = FE.process(kernelSource, {});

  if (!IR.success())
    return -1;

  lcl::MetalBackend BE;
  // IR->print(llvm::errs(), nullptr);
  BE.compile(IR);

  return 0;
}
