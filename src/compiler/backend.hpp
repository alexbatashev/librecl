#pragma once

#include "llvm/IR/Module.h"

#include <vector>

namespace lcl {
class Backend {
public:
  virtual std::vector<unsigned char>
  compile(std::unique_ptr<llvm::Module> module) = 0;

  virtual ~Backend() = default;
};
} // namespace lcl
