#pragma once

#ifndef LCL_API
#ifdef _WIN32
#ifdef LCL_BUILD_DLL
#define LCL_API __declspec(dllexport)
#else
#define LCL_API __declspec(dllimport)
#endif // LCL_BUILD_DLL
#else  // _WIN32
#define LCL_API __attribute__((visibility("default")))
#endif // _WIIN32
#endif // LCL_API
