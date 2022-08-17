extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    let khr_icd_bindings = bindgen::Builder::default()
        .header("../../third_party/OpenCL-Headers/CL/cl_icd.h")
        .clang_arg("-I../../third_party/OpenCL-Headers")
        .ignore_functions()
        .allowlist_type("_cl_icd_dispatch")
        .allowlist_type("cl_api_.*")
        .allowlist_type("cl_command_queue_properties")
        .allowlist_type("cl_command_queue_info")
        .allowlist_type("cl_image.*")
        .allowlist_type("cl_mem_.*")
        .blocklist_type("cl_mem_flags")
        .allowlist_type("cl_pipe.*")
        .allowlist_type("cl_svm.*")
        .allowlist_type("cl_sampler.*")
        .allowlist_type("cl_program_.*")
        .allowlist_type("cl_kernel_.*")
        .allowlist_type("cl_event.*")
        .allowlist_type("cl_profiling.*")
        .allowlist_type("cl_map.*")
        .allowlist_type("cl_GL.*")
        .allowlist_type("cl_gl.*")
        .allowlist_type("CLegl.*")
        .allowlist_type("cl_buffer_.*")
        .allowlist_type("cl_device_.*")
        .blocklist_type("cl_device_type")
        .blocklist_type("cl_device_info")
        .allowlist_type("cl_egl.*")
        .allowlist_type("cl_addressing.*")
        .allowlist_type("cl_filter.*")
        .allowlist_type("_cl_event.*")
        .allowlist_type("cl_bitfield.*")
        .allowlist_type("_cl_sampler.*")
        .allowlist_type("_cl_image.*")
        .allowlist_type("cl_properties.*")
        .allowlist_type("cl_channel.*")
        .allowlist_type("_cl_buffer_.*")
        .allowlist_type("_cl_mem_.*")
        .allowlist_type("_cl_device_.*")
        .blocklist_type("_cl_device_id")
        .allowlist_type("_GL.*")
        .allowlist_type("__GL.*")
        .allowlist_recursively(false)
        .generate()
        .expect("Failed to generate wrappers for cl_icd.h");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    khr_icd_bindings
        .write_to_file(out_path.join("cl_icd.rs"))
        .expect("Couldn't write bindings!");

    #[cfg(not(any(feature="vulkan",feature="metal")))]
    panic!("One of the features must be enabled: vulkan, metal");

    let out_dir = env::var("OUT_DIR").unwrap_or("none".to_string());
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,{}/../../../lcl_compiler/lib",
        out_dir
    );
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,{}/../../../lcl_compiler/lib64",
        out_dir
    );
    println!("cargo:rustc-link-lib=dylib=lcl_compiler");
}
