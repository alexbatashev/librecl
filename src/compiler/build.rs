use cmake::Config;
use std::env;
use std::process::Command;

fn main() {
    let dir = String::from(env::current_dir().unwrap().as_os_str().to_str().unwrap());

    Command::new("bash")
        .args(&[std::format!("{}{}", dir, "/../../build_dependencies.sh")])
        .status()
        .unwrap();

    let dst = Config::new("../../")
        .define(
            "MLIR_DIR",
            std::format!(
                "{}{}",
                dir,
                "/../../third_party/llvm-project/build/lib/cmake/mlir"
            ),
        )
        .define(
            "LLVM_DIR",
            std::format!(
                "{}{}",
                dir,
                "/../../third_party/llvm-project/build/lib/cmake/llvm"
            ),
        )
        .define(
            "Clang_DIR",
            std::format!(
                "{}{}",
                dir,
                "/../../third_party/llvm-project/build/lib/cmake/clang"
            ),
        )
        .define("LLVM_ENABLE_ASSERTIONS", "OFF")
        .build();

    println!("cargo:rustc-link-search=native={}/lib64/", dst.display());
    println!("cargo:rerun-if-changed=../../third_party/llvm-project");
    println!("cargo:rerun-if-changed=../compiler");
    println!("cargo:rustc-link-lib=lcl_compiler");
}
