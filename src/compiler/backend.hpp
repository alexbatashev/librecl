#pragma once

#include "frontend.hpp"
#include "visibility.hpp"

#include <vector>

namespace lcl {
class LCL_COMP_EXPORT Backend {
public:
  virtual std::vector<unsigned char> compile(FrontendResult &module) = 0;

  virtual ~Backend() = default;
};
} // namespace lcl
