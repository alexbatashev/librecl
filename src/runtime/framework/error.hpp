//===- error.hpp ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <exception>
#include <string>

struct UnsupportedFeature : public std::exception {
  UnsupportedFeature(const std::string &errorMessage,
                     const std::string &supportedFeatures)
      : mErrorMessage(errorMessage), mSupportedFeatures(supportedFeatures) {}

  const char *what() const noexcept final { return mErrorMessage.c_str(); }

  const char *getSupportedFeatures() const noexcept {
    return mSupportedFeatures.c_str();
  }

private:
  std::string mErrorMessage;
  std::string mSupportedFeatures;
};
