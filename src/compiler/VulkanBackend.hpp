#pragma once

#include "backend.hpp"
#include "visibility.hpp"

#include <memory>

namespace lcl {
namespace detail {
class VulkanSPVBackendImpl;
}
class LCL_COMP_EXPORT VulkanBackend : public Backend {
public:
  VulkanBackend();

  std::vector<unsigned char> compile(FrontendResult &module) final;

  ~VulkanBackend() = default;

private:
  std::shared_ptr<detail::VulkanSPVBackendImpl> mImpl;
};
} // namespace lcl
