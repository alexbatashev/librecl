#pragma once

#include "frontend.hpp"

#include "llvm/IR/Module.h"

namespace lcl::detail {
struct Module {
  std::unique_ptr<llvm::Module> module;
};
} // namespace lcl::detail
