#include <metal_stdlib>

extern "C" {
size_t _Z13get_global_idj(unsigned i, uint3 posInGrid, uint3 posInGroup,
                          uint3 tgPosInGrid, uint3 threadsPerGroup) {
  if (i == 0) {
    return posInGrid.x;
  } else if (i == 1) {
    return posInGrid.y;
  }

  return posInGrid.z;
}
}
