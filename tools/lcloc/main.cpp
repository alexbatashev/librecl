//===- main.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangFrontend.hpp"
#include "MetalBackend.hpp"
#include "VulkanBackend.hpp"

#include "llvm/Support/CommandLine.h"

#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <span>

enum Target { metal_macos, metal_ios, vulkan_spv };

enum OutputStage { DefaultStage, llvm_ir, llvm_ir_text, mlir, spirv, msl };

enum OptLevel { Debug, O1, O2, O3 };

using out_func_t = std::function<void(std::span<const char>)>;

std::unique_ptr<lcl::Frontend> prepareClangFrontend() {
  return std::make_unique<lcl::ClangFrontend>();
}

std::unique_ptr<lcl::Backend>
prepareMetalBackend(Target target, OutputStage stage, out_func_t printer) {
  std::unique_ptr<lcl::MetalBackend> BE = std::make_unique<lcl::MetalBackend>();

  if (stage == llvm_ir) {
    BE->setLLVMIRPrinter([=](std::span<char> res) { printer(res); });
  } else if (stage == llvm_ir_text) {
    BE->setLLVMTextPrinter([=](std::string_view res) { printer(res); });
  } else if (stage == mlir) {
    BE->setMLIRPrinter([=](std::string_view res) { printer(res); });
  } else if (stage == spirv) {
    BE->setSPVPrinter([=](std::span<unsigned char> res) {
      std::span<char> view{reinterpret_cast<char *>(res.data()),
                           reinterpret_cast<char *>(res.data() + res.size())};
      printer(view);
    });
  } else if (stage == msl) {
    BE->setMSLPrinter([=](std::string_view res) { printer(res); });
  }

  return BE;
}

std::unique_ptr<lcl::Backend>
prepareVulkanBackend(Target target, OutputStage stage, out_func_t printer) {
  std::unique_ptr<lcl::VulkanBackend> BE =
      std::make_unique<lcl::VulkanBackend>();

  if (stage == llvm_ir) {
    BE->setLLVMIRPrinter([=](std::span<char> res) { printer(res); });
  } else if (stage == llvm_ir_text) {
    BE->setLLVMTextPrinter([=](std::string_view res) { printer(res); });
  } else if (stage == mlir) {
    BE->setMLIRPrinter([=](std::string_view res) { printer(res); });
  } else if (stage == spirv) {
    BE->setSPVPrinter([=](std::span<unsigned char> res) {
      std::span<char> view{reinterpret_cast<char *>(res.data()),
                           reinterpret_cast<char *>(res.data() + res.size())};
      printer(view);
    });
  }

  return BE;
}

int main(int argc, char **argv) {
  using namespace llvm;

  cl::opt<std::string> outputFilename("o", cl::desc("Specify output filename"),
                                      cl::value_desc("filename"));
  cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input file>"),
                                     cl::Required);
  cl::opt<Target> target(
      "target", cl::desc("Set output target:"), cl::Required,
      cl::values(
          clEnumValN(metal_ios, "metal-ios", "iOS-style Apple Metal"),
          clEnumValN(metal_macos, "metal-macos", "macOS-style Apple Metal"),
          clEnumValN(vulkan_spv, "vulkan-spv", "Vulkan-style SPIR-V")));
  cl::opt<OutputStage> stage(
      "stage", cl::desc("Set output stage:"),
      cl::values(
          clEnumValN(DefaultStage, "default",
                     "Default output stage for target"),
          clEnumValN(llvm_ir, "llvm-ir", "LLVM IR bitcode"),
          clEnumValN(llvm_ir_text, "llvm-ir-text", "LLVM IR textual form"),
          clEnumVal(mlir, "MLIR representation"),
          clEnumVal(spirv, "SPIR-V binary"),
          clEnumVal(msl, "Metal Shading Language")));
  cl::opt<OptLevel> optLevel(
      cl::desc("Choose optimization level:"),
      cl::values(clEnumValN(Debug, "g", "No optimizations, enable debugging"),
                 clEnumVal(O1, "Enable trivial optimizations"),
                 clEnumVal(O2, "Enable default optimizations"),
                 clEnumVal(O3, "Enable expensive optimizations")));
  cl::ParseCommandLineOptions(argc, argv);

  std::ifstream ifs{inputFilename.c_str()};

  std::string inputData{std::istreambuf_iterator<char>(ifs),
                        std::istreambuf_iterator<char>()};

  // TODO prepare SPIR-V frontend depending on file extension
  auto FE = prepareClangFrontend();

  auto IR = FE->process(inputData, {});

  if (!IR.success()) {
    std::cerr << "Frontend error:\n";
    std::cerr << IR.error();
    return -1;
  }

  std::unique_ptr<lcl::Backend> BE;

  out_func_t printer;

  if (outputFilename == "" || outputFilename == "-") {
    printer = [](std::span<const char> out) {
      std::string_view str{out.begin(), out.end()};
      std::cout << str;
    };
  } else {
    std::string filename = outputFilename;
    printer = [=](std::span<const char> out) {
      std::string_view str{out.begin(), out.end()};
      std::ofstream of{filename};
      of << str;
      of.close();
    };
  }

  OutputStage unwrappedStage = stage;

  if (stage == DefaultStage) {
    if (target == metal_macos || target == metal_ios) {
      unwrappedStage = msl;
    } else if (target == vulkan_spv) {
      unwrappedStage = spirv;
    }
  }

  if (target == metal_macos || target == metal_ios) {
    BE = prepareMetalBackend(target, unwrappedStage, printer);
  } else {
    BE = prepareVulkanBackend(target, unwrappedStage, printer);
  }

  BE->compile(IR);

  return 0;
}
