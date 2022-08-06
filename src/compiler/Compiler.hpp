#pragma once

#include <variant>
#include <string>
#include <span>
#include <memory>
#include <vector>

namespace lcl {

struct CompileOnly {
  std::string getOption();
};

struct NoOpt {
  std::string getOption();
};

struct Target {
  enum class Kind {
    VulkanSPIRV,
    MSL,
    PTX,
    AMDGPU
  };

  Kind targetKind;
};

using Option = std::variant<CompileOnly, NoOpt, std::string_view>;

class CompileResult {};

class CompilerJob {};

class Compiler {
public:
  CompileResult compile(std::span<const char> input, std::span<Option> options);
  CompileResult compile(std::span<CompileResult> inputs, std::span<Option> options);

private:
  // Frontend jobs
  std::unique_ptr<CompilerJob> createStandardClangJob();
  std::unique_ptr<CompilerJob> createSPIRVTranslatorJob();

  // Optimization jobs
  std::unique_ptr<CompilerJob> createOptimizeLLVMIRJob();
  std::unique_ptr<CompilerJob> createOptimizeMLIRJob();

  // Conversion jobs
  std::unique_ptr<CompilerJob> createConvertToMLIRJob();
  std::unique_ptr<CompilerJob> createConvertMLIRToVulkanSPIRVJob();
  std::unique_ptr<CompilerJob> createConvertMLIRToMSLJob();
  std::unique_ptr<CompilerJob> createConvertMLIRToPTXJob();
  std::unique_ptr<CompilerJob> createConvertMLIRToAMDGPUJob();
};

}
