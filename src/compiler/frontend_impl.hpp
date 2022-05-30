#pragma once

#include "frontend.hpp"

#include "llvm/IR/Module.h"

#include <memory>

namespace lcl::detail {
struct Module {
  Module(std::unique_ptr<llvm::Module> module) : module(std::move(module)) {}

  std::unique_ptr<llvm::Module> module;
};
} // namespace lcl::detail
