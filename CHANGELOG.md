# Changelog

## 0.1.0 (2022-10-08)


### Features

* **compiler, rt:** outline builtin functions to a separate library ([#49](https://github.com/alexbatashev/librecl/issues/49)) ([6fa571a](https://github.com/alexbatashev/librecl/commit/6fa571a3fb8e02fbb497204b56d139a443c07f4f))
* **compiler, vk:** add Vulkan backend execution capabilities ([#10](https://github.com/alexbatashev/librecl/issues/10)) ([04a12a7](https://github.com/alexbatashev/librecl/commit/04a12a77bf7da23c6d7fe60c4b6a478a15098f3a))
* **compiler:** add SPIR-V frontend ([#9](https://github.com/alexbatashev/librecl/issues/9)) ([dcd6f64](https://github.com/alexbatashev/librecl/commit/dcd6f645509636cef2ecd81320e4b4e385216f7f))
* **compiler:** add support for SPIR-V translator as a frontend ([#30](https://github.com/alexbatashev/librecl/issues/30)) ([fa938c0](https://github.com/alexbatashev/librecl/commit/fa938c0868b0a8739580ab76b90b30b78cb6923e))
* **compiler:** emit Metal Shading Language source code from MLIR ([#13](https://github.com/alexbatashev/librecl/issues/13)) ([a2a61dd](https://github.com/alexbatashev/librecl/commit/a2a61ddb656677b35dfa4ee8eae3359c886d23e2))
* **compiler:** enable support for multiple files in lcloc ([#51](https://github.com/alexbatashev/librecl/issues/51)) ([7fab7b2](https://github.com/alexbatashev/librecl/commit/7fab7b2f47cc9e62efa4f44e902613abb3a7e7ec))
* **compiler:** make online compiler optional ([#43](https://github.com/alexbatashev/librecl/issues/43)) ([00cf1f2](https://github.com/alexbatashev/librecl/commit/00cf1f2e5160e81f5bee318565c7b80750012961))
* **compiler:** rewrite lcloc in rust ([#42](https://github.com/alexbatashev/librecl/issues/42)) ([503adaf](https://github.com/alexbatashev/librecl/commit/503adafe6ac437800f16e5d2f1980f7f7868a1a0))
* **compiler:** support structures in OpenCL compiler ([#24](https://github.com/alexbatashev/librecl/issues/24)) ([abf3415](https://github.com/alexbatashev/librecl/commit/abf34154be5399c37be024ee7c3c76b070883002))
* **framework, vk:** add framework for improved debug output ([#14](https://github.com/alexbatashev/librecl/issues/14)) ([86033dc](https://github.com/alexbatashev/librecl/commit/86033dc7d05c4fb742f79ec3dc835178d6991c3f))
* **metal:** add support for platform info queries ([#31](https://github.com/alexbatashev/librecl/issues/31)) ([cba1d67](https://github.com/alexbatashev/librecl/commit/cba1d67027d87f8425ea6f8e3884cd1c9409bfc3))
* **rt, cpu:** kick off CPU backend ([#28](https://github.com/alexbatashev/librecl/issues/28)) ([6d92f8d](https://github.com/alexbatashev/librecl/commit/6d92f8d2fe70c5c0ff0ac0bbc2e9f9ec4b8dbc93))
* **rt, vk:** add support for more device info queries ([#32](https://github.com/alexbatashev/librecl/issues/32)) ([a405936](https://github.com/alexbatashev/librecl/commit/a4059360091cdf7d85feabaae3a419fb639e7760))
* **rt, vk:** support for all available platform infos ([#26](https://github.com/alexbatashev/librecl/issues/26)) ([7145286](https://github.com/alexbatashev/librecl/commit/7145286ddee5c9770b678e4afd52219588acae2f))
* **rt, vk:** Vulkan execution ([#17](https://github.com/alexbatashev/librecl/issues/17)) ([7b79235](https://github.com/alexbatashev/librecl/commit/7b7923539885a4ba06eac32e923627f10e645876))
* **rt:** add basic support for cl_khr_il_program extension ([#35](https://github.com/alexbatashev/librecl/issues/35)) ([bea2428](https://github.com/alexbatashev/librecl/commit/bea24284f4807d0e5a88cea201cb9670a46a7016))
* **rt:** basic trace output ([#47](https://github.com/alexbatashev/librecl/issues/47)) ([7814def](https://github.com/alexbatashev/librecl/commit/7814def8f0c022b1e6c090482c42173a5a49d25b))
* **rt:** enable device retain and release ([#40](https://github.com/alexbatashev/librecl/issues/40)) ([ae9a96f](https://github.com/alexbatashev/librecl/commit/ae9a96fcd4acf19389e024c5ab0c2a050f0178a3))
* **rt:** implement resource release and retain ([#23](https://github.com/alexbatashev/librecl/issues/23)) ([90bad8b](https://github.com/alexbatashev/librecl/commit/90bad8bb0ff522aa6d291fdded74eb060d14d0ea))
* **rt:** improve support for device info queries ([#44](https://github.com/alexbatashev/librecl/issues/44)) ([8fd2ec7](https://github.com/alexbatashev/librecl/commit/8fd2ec7a6b5b1ae10e950dd5a8a1a6c2ff9dc2e8))
* **rt:** improve support for device info queries ([#45](https://github.com/alexbatashev/librecl/issues/45)) ([09af6cf](https://github.com/alexbatashev/librecl/commit/09af6cf750422ab1da6aaca322f111d4cabbd511))
* **rt:** initial support for OpenCL ICD Loader ([#22](https://github.com/alexbatashev/librecl/issues/22)) ([cc55396](https://github.com/alexbatashev/librecl/commit/cc55396e619415f7f31fdbcef8eee02caf2b1a12))
* **rt:** support context info queries ([#46](https://github.com/alexbatashev/librecl/issues/46)) ([e0254fa](https://github.com/alexbatashev/librecl/commit/e0254fa1c0a035f1365dd97eb1988955b1c18b69))


### Bug Fixes

* cl_khr_icd extension support ([#39](https://github.com/alexbatashev/librecl/issues/39)) ([cee0c91](https://github.com/alexbatashev/librecl/commit/cee0c918ff3d57e42b6a1ba2ea896f65fd16dbea))
* **compiler:** -c option was ignored ([#48](https://github.com/alexbatashev/librecl/issues/48)) ([dc1a695](https://github.com/alexbatashev/librecl/commit/dc1a695cdabec82b94bf53864305d3a2413c8181))
* minor fix for build system ([ad7d698](https://github.com/alexbatashev/librecl/commit/ad7d69836cc252f19cd6777a7a0011fd1dc9936f))
* **rt:** prohibit warnings in runtime code ([#27](https://github.com/alexbatashev/librecl/issues/27)) ([3acee45](https://github.com/alexbatashev/librecl/commit/3acee45a3cc3c0bf1e480153c35b59f8cf0edf68))
* **vk:** invalid memory access ([#15](https://github.com/alexbatashev/librecl/issues/15)) ([7af6f70](https://github.com/alexbatashev/librecl/commit/7af6f7051eb2475374ad102178300db92cf92942))
