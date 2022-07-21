//===- debug_modes.hpp ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "framework/utils.hpp"

#include <optional>
#include <string_view>

namespace lcl {
enum class DebugModeFlags {
  None,
  LogReport,
  CaptureObjectLocations,
  CaptureObjectNames,
  _Last
};

using DebugMode = lcl::bitset<DebugModeFlags>;

namespace detail {
struct debuggable_object_impl {
  debuggable_object_impl(DebugMode mode, const std::string &definingFunciton);

  std::string renderLogMessage(std::string_view message);

  DebugMode getDebugMode() const noexcept { return mDebugMode; }

  std::optional<std::string> getDebugName() const;

protected:
  void setNameImpl(const std::string &name) { mDebugName = name; }

  struct LocationTag {
    std::optional<std::string> filename;
    std::optional<int> line;
    std::optional<std::string> function;
  };
  std::optional<LocationTag> mDebugLocation;
  std::optional<std::string> mDebugName;
  DebugMode mDebugMode;
};
} // namespace detail

template <typename Object>
struct debuggable_object : public detail::debuggable_object_impl {
  using detail::debuggable_object_impl::debuggable_object_impl;

  void setName(const std::string &name) {
    setNameImpl(name);
    if (mDebugMode.has(DebugModeFlags::CaptureObjectNames)) {
      Object::setBackendObjectNames(name);
    }
  }
};
} // namespace lcl
