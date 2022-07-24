fn main() {
    if build_target::target_os().unwrap() == build_target::Os::Linux {
        println!("cargo:rustc-cfg=feature=\"vulkan\"");
    } else if build_target::target_os().unwrap() == build_target::Os::MacOs {
        println!("cargo:rustc-cfg=feature=\"metal\"");
    }
}
