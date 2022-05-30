#pragma once

#include "backend.hpp"
#include "visibility.hpp"

#include <memory>

namespace lcl {
namespace detail {
class MetalBackendImpl;
}
class LCL_COMP_EXPORT MetalBackend : public Backend {
public:
  MetalBackend();

  std::vector<unsigned char> compile(FrontendResult &module) final;

  ~MetalBackend() = default;

private:
  std::shared_ptr<detail::MetalBackendImpl> mImpl;
};
}
