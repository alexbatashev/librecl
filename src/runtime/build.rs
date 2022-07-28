extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    let khr_icd_bindings = bindgen::Builder::default()
        .header("../../third_party/OpenCL-Headers/CL/cl_icd.h")
        .clang_arg("-I../../third_party/OpenCL-Headers")
        .ignore_functions()
        .allowlist_type("_cl_icd_dispatch")
        .generate()
        .expect("Failed to generate wrappers for cl_icd.h");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    khr_icd_bindings
        .write_to_file(out_path.join("cl_icd.rs"))
        .expect("Couldn't write bindings!");
    if build_target::target_os().unwrap() == build_target::Os::Linux {
        println!("cargo:rustc-cfg=feature=\"vulkan\"");
    } else if build_target::target_os().unwrap() == build_target::Os::MacOs {
        println!("cargo:rustc-cfg=feature=\"metal\"");
    }
}
