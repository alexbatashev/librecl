#pragma once

#include "frontend.hpp"
#include "visibility.hpp"

#include <memory>
#include <string_view>
#include <span>

namespace lcl {
namespace detail {
class ClangFrontendImpl;
}
class LCL_COMP_EXPORT ClangFrontend : public Frontend {
public:
  ClangFrontend();

  FrontendResult process(std::string_view input,
                         std::span<std::string_view> options) final;

  ~ClangFrontend() = default;

private:
  std::shared_ptr<detail::ClangFrontendImpl> mImpl;
};
} // namespace lcl
