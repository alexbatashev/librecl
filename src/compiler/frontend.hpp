#pragma once

#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <string_view>
#include <span>

namespace lcl {
class Frontend {
public:
  virtual std::unique_ptr<llvm::Module>
  process(std::string_view input, std::span<std::string_view> options) = 0;
  virtual ~Frontend() = default;
};
} // namespace lcl
