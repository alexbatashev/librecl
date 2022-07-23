//===- error.hpp ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <exception>
#include <fmt/core.h>
#include <string>

#include "utils.hpp"

template <typename ErrorType> struct LCLError : public std::exception {
  const char *what() const noexcept final { return mMessage.c_str(); }

protected:
  LCLError(const std::string &message) {
    constexpr auto fmtString =
        R"(
  Error kind: {}
  Error message: {}

  {}
)";
    mMessage = fmt::format(fmtString, ErrorType::getErrorKind(), message,
                           getStackTrace(3));
  }

private:
  std::string mMessage;
};

struct UnsupportedFeature : public LCLError<UnsupportedFeature> {
  UnsupportedFeature(const std::string &errorMessage,
                     const std::string &supportedFeatures)
      : LCLError(errorMessage), mSupportedFeatures(supportedFeatures) {}

  static const char *getErrorKind() noexcept { return "UnsupportedFeature"; }

  const char *getSupportedFeatures() const noexcept {
    return mSupportedFeatures.c_str();
  }

private:
  std::string mSupportedFeatures;
};

struct InternalBackendError : public LCLError<InternalBackendError> {
  InternalBackendError(const std::string &msg) : LCLError(msg) {}

  static const char *getErrorKind() noexcept { return "InternalBackendError"; }
};
