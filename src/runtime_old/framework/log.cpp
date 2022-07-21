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
  std::string_view color;

  constexpr auto red = "\033[31m";
  constexpr auto yellow = "\033[33m";
  constexpr auto white = "\033[37m";
  constexpr auto cyan = "\033[36m";
  constexpr auto green = "\033[32m";
  constexpr auto reset = "\033[0m";
  constexpr auto colorPrefix = "\033[1m";

  switch (level) {
  case LogLevel::Debug:
    kind = "[DEBUG]:";
    color = white;
    break;
  case LogLevel::Warning:
    kind = "[WARNING]:";
    color = yellow;
    break;
  case LogLevel::Error:
    kind = "[ERROR]:";
    color = red;
    break;
  case LogLevel::Performance:
    kind = "[PERF]:";
    color = cyan;
    break;
  case LogLevel::Information:
    kind = "[INFO]:";
    color = green;
    break;
  }

  fmt::print("{}{}{:10}{} {}\n", colorPrefix, color, kind, reset, message);
}
