//===- debug_modes.hpp ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "debug_modes.hpp"

#include <backward.hpp>
#include <fmt/core.h>
#include <optional>

namespace lcl {
namespace detail {
debuggable_object_impl::debuggable_object_impl(
    lcl::DebugMode mode, const std::string &definingFunction)
    : mDebugMode(mode) {
  if (mode.has(DebugModeFlags::CaptureObjectLocations)) {
    backward::StackTrace st;
    st.load_here(32);

    backward::TraceResolver resolver;
    resolver.load_stacktrace(st);

    for (size_t idx = st.size(); idx > 0; idx--) {
      backward::ResolvedTrace trace = resolver.resolve(st[idx]);
      if (trace.object_function.starts_with(definingFunction)) {
        backward::ResolvedTrace parent = resolver.resolve(st[idx + 1]);
        LocationTag tag;
        bool needsUpdate = false;
        if (!parent.source.filename.empty()) {
          needsUpdate = true;
          tag.filename = parent.source.filename;
        }
        if (parent.source.line != 0) {
          needsUpdate = true;
          tag.line = parent.source.line;
        }
        if (!parent.source.function.empty()) {
          needsUpdate = true;
          tag.function = parent.source.function;
        }
        if (needsUpdate) {
          mDebugLocation = tag;
        }
        break;
      }
    }
  }
}

std::optional<std::string> debuggable_object_impl::getDebugName() const {
  if (mDebugLocation) {
    return fmt::format("{} created at {} +{} ({})", mDebugName.value_or(""),
                       mDebugLocation->filename.value_or("unknown file"),
                       mDebugLocation->line.value_or(0),
                       mDebugLocation->function.value_or("unknown function"));
  }
  if (mDebugName) {
    return mDebugName;
  }

  return std::nullopt;
}

std::string debuggable_object_impl::renderLogMessage(std::string_view message) {
  auto debugName = getDebugName();

  if (debugName)
    return fmt::format("For object {}:\n{}", debugName.value(), message);

  return std::string{message};
}
} // namespace detail
} // namespace lcl
