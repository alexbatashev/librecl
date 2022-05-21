#pragma once

#include "frontend.hpp"

#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <string_view>
#include <span>

namespace lcl {
namespace detail {
class ClangFrontendImpl;
}
class ClangFrontend : public Frontend {
public:
  ClangFrontend();

  std::unique_ptr<llvm::Module>
  process(std::string_view input, std::span<std::string_view> options) final;

  ~ClangFrontend() = default;

private:
  std::shared_ptr<detail::ClangFrontendImpl> mImpl;
};
} // namespace lcl
