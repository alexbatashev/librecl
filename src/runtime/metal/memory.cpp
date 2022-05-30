//===- memory.cpp -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "memory.hpp"
#include "context.hpp"

_cl_mem::_cl_mem(cl_context ctx, cl_mem_flags, size_t size) : mContext(ctx) {
  for (auto &dev : mContext->getDevices()) {
    dev->getNativeDevice()->newBuffer(size, MTL::ResourceStorageModePrivate);
  }
}
