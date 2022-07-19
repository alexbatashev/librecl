//===- kernel.cpp -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <backward.hpp>
#include <bitset>
#include <cstdint>
#include <filesystem>
#include <fmt/core.h>
#include <sstream>
#include <string>

template <class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

inline std::string getStackTrace(size_t offset = 0) {
  std::stringstream msgStream;
  backward::StackTrace st;
  st.load_here(32);

  backward::TraceResolver resolver;
  resolver.load_stacktrace(st);

  offset += 2;

  msgStream << "Stacktrace (most recent call first):\n\n";
  for (size_t idx = offset; idx < st.size(); idx++) {
    backward::ResolvedTrace trace = resolver.resolve(st[idx]);
    std::filesystem::path path{trace.object_filename};
    msgStream << fmt::format("  #{:<5} {:25} : {}[{}]\n", idx - offset,
                             path.filename().c_str(), trace.object_function,
                             trace.addr);
    // TODO consider printing snippets
  }

  return msgStream.str();
}

namespace lcl {
template <typename T> struct bitset {
  bool has(T bit) { return mBits.test(static_cast<size_t>(bit)); }

  void set(T bit) { mBits.set(static_cast<size_t>(bit)); }

  void reset(T bit) { mBits.reset(static_cast<size_t>(bit)); }

private:
  std::bitset<static_cast<size_t>(T::_Last)> mBits;
};
} // namespace lcl
