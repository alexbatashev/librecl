#pragma once

#include "backend.hpp"
#include "llvm/IR/Module.h"
#include <memory>

namespace lcl {
namespace detail {
class MetalBackendImpl;
}
class MetalBackend : public Backend {
public:
  MetalBackend();
  
  std::vector<unsigned char>
  compile(std::unique_ptr<llvm::Module> module) final;

  ~MetalBackend() = default;

private:
  std::shared_ptr<detail::MetalBackendImpl> mImpl;
};
}
