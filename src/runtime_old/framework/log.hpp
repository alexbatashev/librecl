//===- log.hpp --------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <string_view>

enum class LogLevel {
  Error = 0,
  Warning = 1,
  Debug = 2,
  Performance = 3,
  Information = 4
};

void log(LogLevel level, std::string_view message);

inline void log(std::string_view message) { log(LogLevel::Error, message); }
