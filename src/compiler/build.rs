use cmake::Config;
use std::env;
use std::process::Command;
use which::which;

fn main() {
    let dir = String::from(env::current_dir().unwrap().as_os_str().to_str().unwrap());

    Command::new("bash")
        .args(&[std::format!("{}{}", dir, "/../../build_dependencies.sh")])
        .status()
        .unwrap();

    let mut cfg = Box::new(Config::new("../../"));
    cfg.define(
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
    .define("LLVM_ENABLE_ASSERTIONS", "OFF");
    let mold = which("mold");
    #[cfg(not(target_os = "macos"))]
    let use_mold = mold.is_ok();
    #[cfg(target_os = "macos")]
    let use_mold = false;
    if use_mold {
        cfg.define("LIBRECL_LINKER", "mold");
        println!("cargo:rustc-link-arg=-fuse-ld={}", mold.unwrap().display());
    } else if which("lld").is_ok() {
        cfg.define("LIBRECL_LINKER", "lld");
    }

    if which("ninja").is_ok() {
        cfg.generator("Ninja");
    }

    let dst = cfg.build();

    println!("cargo:rustc-link-search=native={}/lib64/", dst.display());
    println!("cargo:rerun-if-changed=../../third_party/llvm-project");
    println!("cargo:rerun-if-changed=../compiler");
    println!("cargo:rustc-link-lib=lcl_compiler");
}
