//===- log.cpp --------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "log.hpp"

#include <cstdlib>
#include <fmt/printf.h>
#include <iostream>
#include <string_view>

void log(LogLevel level, std::string_view message) {
  int iLevel = -1;
  if (auto *strLevel = std::getenv("LIBRECL_DEBUG")) {
    iLevel = std::stoi(strLevel);
  }
  if (iLevel < static_cast<int>(level)) {
    return;
  }

  std::string_view kind;

  switch (level) {
  case LogLevel::Debug:
    kind = "DEBUG";
    break;
  case LogLevel::Warning:
    kind = "WARNING";
    break;
  case LogLevel::Error:
    kind = "ERROR";
    break;
  case LogLevel::Performance:
    kind = "PERF";
    break;
  case LogLevel::Information:
    kind = "INFO";
    break;
  }

  fmt::print("[{}]: {}\n", kind, message);
}
