#pragma once

#include "frontend.hpp"

#include <vector>

namespace lcl {
class Backend {
public:
  virtual std::vector<unsigned char> compile(FrontendResult &module) = 0;

  virtual ~Backend() = default;
};
} // namespace lcl
