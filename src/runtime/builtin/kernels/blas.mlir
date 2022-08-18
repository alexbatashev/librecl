module {
func.func @matmul(%a: memref<?x?xf32, offset: ?, strides: [?, 1]>, %b: memref<?x?xf32, offset: ?, strides: [?, 1]>, %output: memref<?x?xf32, offset: ?, strides: [?, 1]>) {
  linalg.matmul ins(%a, %b: memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>) outs(%output: memref<?x?xf32, offset: ?, strides: [?, 1]>)
  return
}
}
