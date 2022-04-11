#include "compiler.hpp"

const char *kernelSource =
    "\n"
    "#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    \n"
    "__kernel void vecAdd(  __global double *a,                       \n"
    "                       __global double *b,                       \n"
    "                       __global double *c,                       \n"
    "                       const unsigned int n)                    \n"
    "{                                                               \n"
    "    //Get our global thread ID                                  \n"
    "    int id = get_global_id(0);                                  \n"
    "                                                                \n"
    "    //Make sure we do not go out of bounds                      \n"
    "    if (id < n)                                                 \n"
    "        c[id] = a[id] + b[id];                                  \n"
    "}                                                               \n"
    "\n";

int main() {
  lcl::Compiler compiler(lcl::BuildTarget::NVPTX);
  compiler.addModuleFromSource(kernelSource, {});

  return 0;
}
