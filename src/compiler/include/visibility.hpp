//===- visibility.hpp -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#ifndef LCL_COMP_EXPORT
#ifdef _WIN32
#ifdef LCL_CPMP_BUILD_DLL
#define LCL_COMP_EXPORT __declspec(dllexport)
#else
#define LCL_COMP_EXPORT __declspec(dllimport)
#endif // LCL_CPMP_BUILD_DLL
#else  // _WIN32
#define LCL_COMP_EXPORT __attribute__ ((visibility ("default")))
#endif // _WIIN32
#endif // LCL_COMP_EXPORT
