#include "VulkanBackend.hpp"
#include "VulkanSPVBackendImpl.hpp"
#include "passes/mlir/passes.hpp"

#include <memory>
#include <vector>

namespace lcl {
VulkanBackend::VulkanBackend()
    : mImpl(std::make_shared<detail::VulkanSPVBackendImpl>()) {}

std::vector<unsigned char> VulkanBackend::compile(FrontendResult &module) {
  return mImpl->compile(module.takeModule());
}
} // namespace lcl
