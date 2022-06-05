//===- SPIRVFrontend.hpp ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SPIRVFrontend.hpp"
#include "frontend_impl.hpp"

#include "LLVMSPIRVLib.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include <sstream>

namespace lcl {
namespace detail {
class SPIRVFrontendImpl {
public:
  SPIRVFrontendImpl() {}

  FrontendResult process(const std::string_view source,
                         const llvm::ArrayRef<llvm::StringRef> options) {
    // TODO this must be copy-free one day, as SPIR-V files tend to be extremely
    // large, up to a few gigabytes.
    std::string spv{source};

    std::istringstream stream{spv};

    llvm::Module *module;

    std::string error;
    if (!llvm::readSpirv(mContext, stream, module, error)) {
      return FrontendResult{error};
    }

    std::unique_ptr<llvm::Module> finalModule{module};
    module = nullptr;

    return FrontendResult{
        std::make_shared<detail::Module>(std::move(finalModule))};
  }

private:
  llvm::LLVMContext mContext;
};

} // namespace detail
SPIRVFrontend::SPIRVFrontend()
    : mImpl(std::make_shared<detail::SPIRVFrontendImpl>()) {}

FrontendResult SPIRVFrontend::process(std::string_view input,
                                      std::span<std::string_view> options) {
  llvm::SmallVector<llvm::StringRef, 15> opts;
  for (std::string_view opt : options) {
    opts.push_back(llvm::StringRef{opt});
  }
  return mImpl->process(input, opts);
}
} // namespace lcl
