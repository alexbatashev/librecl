use std::env;
use std::path::PathBuf;

fn main() {
    let compiler_bindings = bindgen::Builder::default()
        .header("opencl/rust_bindings.hpp")
        .ignore_functions()
        .generate()
        .expect("Failed to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    compiler_bindings
        .write_to_file(out_path.join("compiler.rs"))
        .expect("Couldn't write bindings!");
}
