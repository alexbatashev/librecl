use cmake::Config;
use std::env;
use std::path::Path;
use std::process::Command;
use which::which;

// Borrowed from Rust repo
// https://github.com/rust-lang/rust/blob/master/compiler/rustc_llvm/build.rs
fn rerun_if_changed_anything_in_dir(dir: &Path) {
    let mut stack = dir
        .read_dir()
        .unwrap()
        .map(|e| e.unwrap())
        .filter(|e| &*e.file_name() != ".git")
        .collect::<Vec<_>>();
    while let Some(entry) = stack.pop() {
        let path = entry.path();
        if entry.file_type().unwrap().is_dir() {
            stack.extend(path.read_dir().unwrap().map(|e| e.unwrap()));
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

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
        println!("cargo:rustc-link-arg=-fuse-ld=mold");
    } else if which("lld").is_ok() {
        cfg.define("LIBRECL_LINKER", "lld");
    }

    if which("ninja").is_ok() {
        cfg.generator("Ninja");
    }

    let dst = cfg.build();

    println!("cargo:rustc-link-search=native={}/lib64/", dst.display());
    println!("cargo:rustc-link-search=native={}/lib/", dst.display());
    rerun_if_changed_anything_in_dir(Path::new("../../third_party/llvm-project"));
    rerun_if_changed_anything_in_dir(Path::new("../compiler"));
    println!("cargo:rustc-link-lib=dylib=lcl_compiler");
}
