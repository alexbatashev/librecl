#include "frontend.hpp"
#include "frontend_impl.hpp"

#include "llvm/IR/Module.h"

namespace lcl {
FrontendResult::FrontendResult(std::shared_ptr<detail::Module> module)
    : mModule(std::move(module)) {}

std::unique_ptr<llvm::Module> FrontendResult::takeModule() {
  return std::move(mModule->module);
}

bool FrontendResult::empty() const { return mModule->module == nullptr; }

FrontendResult::~FrontendResult() = default;
FrontendResult::FrontendResult(FrontendResult &&) = default;
} // namespace lcl
