#pragma once

#include "backend.hpp"

#include <memory>

namespace lcl {
namespace detail {
class MetalBackendImpl;
}
class MetalBackend : public Backend {
public:
  MetalBackend();

  std::vector<unsigned char> compile(FrontendResult &module) final;

  ~MetalBackend() = default;

private:
  std::shared_ptr<detail::MetalBackendImpl> mImpl;
};
}
