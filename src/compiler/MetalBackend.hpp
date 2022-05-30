#pragma once

#include "backend.hpp"
#include "visibility.hpp"

#include <functional>
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

  void setLLVMIRPrinter(std::function<void(std::span<char>)> printer);
  void setMLIRPrinter(std::function<void(std::string_view)> printer);
  void setLLVMTextPrinter(std::function<void(std::string_view)> printer);
  void setSPVPrinter(std::function<void(std::span<unsigned char>)> printer);
  void setMSLPrinter(std::function<void(std::string_view)> printer);

private:
  std::shared_ptr<detail::MetalBackendImpl> mImpl;
};
}
