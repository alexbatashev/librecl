#pragma once

#include <iostream>
#include <string>

inline cl_int setParamValueStr(const std::string &str, size_t paramValueSize,
                               void *paramValue, size_t *paramValueSizeRet) {
  if (paramValueSize < (str.size() + 1) && paramValue) {
    // log: allocated memory is less than actual value
    return CL_INVALID_VALUE;
  }

  if (paramValueSizeRet) {
    *paramValueSizeRet = str.size() + 1;
  }

  if (paramValue) {
    std::memcpy(paramValue, str.data(), str.size() + 1);
  }

  return CL_SUCCESS;
}

template <typename T>
inline cl_int setParamValue(T val, size_t paramValueSize, void *paramValue,
                            size_t *paramValueSizeRet) {
  if (paramValueSize < sizeof(T) && paramValue) {
    // log: allocated memory is less than actual value
    return CL_INVALID_VALUE;
  }

  if (paramValueSizeRet) {
    *paramValueSizeRet = sizeof(T);
  }

  if (paramValue) {
    *static_cast<T *>(paramValue) = val;
  }

  return CL_SUCCESS;
}
