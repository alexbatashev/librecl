#pragma once

#include <memory>
#include <string>

namespace lcl {
void initializeTargets();

enum class BuildTarget {
  NVPTX,
  CPU,
  Metal
};

class CompileResult {
public:
  bool success() const { return mSuccess; }

  const std::string_view moduleName() const {
    return mKey;
  }

private:
  std::string mErrorLog;
  std::string mKey;
  bool mSuccess;
};

class CompilerImpl;

class Compiler {
public:
  Compiler(BuildTarget target);

  void addModuleFromSource(const std::string_view source, const std::string_view options);

  ~Compiler();
private:
  std::unique_ptr<CompilerImpl> mCompiler;
};
}
