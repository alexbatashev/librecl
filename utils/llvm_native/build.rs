use cmake::Config;
use std::env;
use std::fs;
use std::path::Path;
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

fn build_llvm(
    dir: &str,
    wrapper: &str,
    out_dir: &str,
    use_mold: bool,
    use_lld: bool,
    use_ninja: bool,
    use_clang: bool,
) -> String {
    let llvm_out = format!("{}/../../../llvm_build", out_dir);
    fs::create_dir_all(llvm_out.as_str()).expect("failed to create target dir");

    let mut llvm_cfg = Box::new(Config::new("../../third_party/llvm-project/llvm/"));

    llvm_cfg
        .define("LLVM_EXTERNAL_PROJECTS", "llvm-spirv")
        .define(
            "LLVM_EXTERNAL_LLVM_SPIRV_SOURCE_DIR",
            std::format!("{}/../../third_party/SPIRV-LLVM-Translator", dir),
        )
        .define("LLVM_ENABLE_PROJECTS", "mlir;clang;lld")
        .define("LLVM_TARGETS_TO_BUILD", "host;AMDGPU;NVPTX")
        .define("CMAKE_EXPORT_COMPILE_COMMANDS", "ON")
        .define("LLVM_ENABLE_WARNINGS", "OFF")
        .define("LLVM_OPTIMIZED_TABLEGEN", "ON")
        .profile("Release")
        .out_dir(llvm_out.as_str());

    if wrapper.contains("ccache") {
        llvm_cfg.define("LLVM_CCACHE_BUILD", "ON");
    } else if wrapper.contains("sccache") {
        llvm_cfg
            .define("CMAKE_CXX_COMPILER_LAUNCHER", wrapper)
            .define("CMAKE_C_COMPILER_LAUNCHER", wrapper);
    }

    if use_mold {
        llvm_cfg.define("LLVM_USE_LINKER", "mold");
    } else if use_lld {
        llvm_cfg.define("LLVM_USE_LINKER", "lld");
    }

    if use_clang {
        llvm_cfg.define("CMAKE_C_COMPILER", "clang")
            .define("CMAKE_CXX_COMPILER", "clang++");
        let profile = env::var("PROFILE").unwrap_or("debug".to_string());
        if profile == "release" {
            llvm_cfg.define("LLVM_ENABLE_LTO", "Thin");
        }
    }

    let is_ci = env::var("CI").unwrap_or("false".to_string());

    if is_ci != "true" {
        llvm_cfg.define("LLVM_APPEND_VC_REV", "OFF");
        llvm_cfg.define("LLVM_PARALLEL_LINK_JOBS", "2");
    }

    if use_ninja {
        llvm_cfg.generator("Ninja");
    }

    llvm_cfg.build();

    return llvm_out;
}

fn build_online_compiler() {
    let dir = String::from(env::current_dir().unwrap().as_os_str().to_str().unwrap());

    let wrapper = env::var("RUSTC_WRAPPER").unwrap_or("none".to_string());
    let out_dir = env::var("OUT_DIR").unwrap_or("none".to_string());

    #[cfg(not(target_os = "macos"))]
    let use_mold = which("mold").is_ok();
    #[cfg(target_os = "macos")]
    let use_mold = false;

    let use_lld = which("lld").is_ok();

    let use_ninja = which("ninja").is_ok();

    let use_clang = which("clang").is_ok();

    let llvm_out = build_llvm(&dir, &wrapper, &out_dir, use_mold, use_lld, use_ninja, use_clang);

    let compiler_out = format!("{}/../../../lcl_compiler", out_dir);
    fs::create_dir_all(compiler_out.as_str()).expect("failed to create target dir");

    let mut cfg = Box::new(Config::new("../../"));
    if wrapper.contains("sccache") {
        cfg.define("CMAKE_CXX_COMPILER_LAUNCHER", wrapper.as_str())
            .define("CMAKE_C_COMPILER_LAUNCHER", wrapper.as_str());
    }
    cfg.define(
        "MLIR_DIR",
        std::format!("{}/lib/cmake/mlir", llvm_out.as_str()),
    )
    .define(
        "LLVM_DIR",
        std::format!("{}/lib/cmake/llvm", llvm_out.as_str()),
    )
    .define(
        "Clang_DIR",
        std::format!("{}/lib/cmake/clang", llvm_out.as_str()),
    )
    .define("LLVM_ENABLE_ASSERTIONS", "OFF")
    .profile("Debug")
    .out_dir(compiler_out.as_str());

    if use_mold {
        cfg.define("LIBRECL_LINKER", "mold");
        println!("cargo:rustc-link-arg=-fuse-ld=mold");
    } else if use_lld {
        cfg.define("LIBRECL_LINKER", "lld");
        println!("cargo:rustc-link-arg=-fuse-ld=lld");
    }

    if use_clang {
        cfg.define("CMAKE_C_COMPILER", "clang")
            .define("CMAKE_CXX_COMPILER", "clang++");
    }

    if use_ninja {
        cfg.generator("Ninja");
    }

    cfg.build();

    if env::var("CI").unwrap_or("false".to_owned()) == "false" {
        rerun_if_changed_anything_in_dir(Path::new("../../third_party/llvm-project"));
        rerun_if_changed_anything_in_dir(Path::new("../../src/compiler"));
        println!("cargo:rerun-if-env-changed=RUSTC_WRAPPER");
        println!("cargo:rerun-if-changed=build.rs");
    }
}

fn main() {
    build_online_compiler();
}
