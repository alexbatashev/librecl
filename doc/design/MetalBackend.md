# Metal backend

## Compiler

### Address space mapping

| OpenCL AS | SPIR AS | AIR AS |
|-----------|---------|--------|
| Private   | 0       | 0      |
| Global    | 1       | 1      |
| Constant  | 2       | 2      |
| Local     | 3       | 3      |
| Generic   | 4       | N/A    |
