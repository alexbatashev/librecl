# librecl

An experimental implementation of OpenCL on top of other computing or graphics
standards: Khronos Vulkan, Apple Metal, NVidia CUDA.

## Build from source

- Get the latest version of Rust toolchain from here: https://rustup.rs/ and
  a C++ toolchain with at least C++20 support for your target (GCC 12 on Linux
  would do).
- It is highly recommended to install Ninja: https://ninja-build.org/
- For Linux it is also recommended to install mold: https://github.com/rui314/mold

Once you've installed required tools and download the source code, you can
build the entire project with cargo:
```bash
cargo build --workspace --release
```

To run runtime tests, do:
```bash
cargo test --workspace --release
```

