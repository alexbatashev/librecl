// TODO this code is too fragile, yet existing arg parsing libraries are not
// flexible enough to cover all cases. Should I make a new library?

#[derive(PartialEq, Eq)]
pub enum Target {
    VulkanSPIRV,
    MetalMacOS,
    MetalIOS,
    NVPTX,
    AMDGPU,
    OpenCLSPIRV,
}

#[derive(PartialEq, Eq)]
pub enum OptLevel {
    OptNone,
    O1,
    O2,
    O3,
}

#[derive(PartialEq, Eq)]
pub enum Stage {
    Binary,
    MLIR,
    LLVMIR,
}

pub struct CompilerArgs {
    pub targets: Vec<Target>,
    pub stage: Stage,
    pub out: String,
    pub inputs: Vec<String>,
    pub compile_only: bool,
    pub opt_level: OptLevel,
    pub mad_enable: bool,
    pub kernel_arg_info: bool,
    pub print_before_all_mlir: bool,
    pub print_after_all_mlir: bool,
    pub print_before_all_llvm: bool,
    pub print_after_all_llvm: bool,
    pub other_options: Vec<String>,
}

impl CompilerArgs {
    pub fn new() -> CompilerArgs {
        CompilerArgs {
            targets: vec![],
            stage: Stage::Binary,
            out: "a.out".to_owned(),
            inputs: vec![],
            compile_only: false,
            opt_level: OptLevel::O2,
            mad_enable: false,
            kernel_arg_info: false,
            print_before_all_mlir: false,
            print_after_all_mlir: false,
            print_before_all_llvm: false,
            print_after_all_llvm: false,
            other_options: vec![],
        }
    }
}

// TODO not all OpenCL options are supported. See
// https://registry.khronos.org/OpenCL/specs/2.2/html/OpenCL_API.html#compiler-options
pub fn parse_options(options: &[String]) -> Result<CompilerArgs, String> {
    let mut parsed_opts = CompilerArgs::new();

    let mut i = 0;
    while i < options.len() {
        if options[i].starts_with("-D") || options[i].starts_with("-I") {
            parsed_opts.other_options.push(options[i].to_owned());
            if options[i].len() == 2 {
                i += 1;
                parsed_opts.other_options.push(options[i].to_owned());
            }
            i += 1;
            continue;
        }

        if options[i] == "-o" {
            parsed_opts.out = options[i + 1].to_owned();
            i += 2;
            continue;
        }

        if options[i] == "-cl-single-precision-constant"
            || options[i] == "-cl-denorms-are-zero"
            || options[i].starts_with("-cl-fp32-correctly-rounded-divide-sqrt")
            || options[i] == "-cl-no-signed-zeros"
            || options[i] == "-cl-unsafe-math-optimizations"
            || options[i] == "-cl-finite-math-only"
            || options[i] == "-cl-fast-relaxed-math"
            || options[i] == "-cl-uniform-work-group-size"
            || options[i] == "-cl-no-subgroup-ifp"
            || options[i] == "-w"
            || options[i] == "-Werror"
            || options[i].starts_with("-cl-std=")
        {
            parsed_opts.other_options.push(options[i].to_owned());
            i += 1;
            continue;
        }

        if options[i].starts_with("--targets=") {
            let targets: String = options[i].chars().skip(10).collect();
            let split_targets = targets.split(",");

            for t in split_targets {
                match t {
                    "vulkan-spirv" => parsed_opts.targets.push(Target::VulkanSPIRV),
                    "metal-macos" => parsed_opts.targets.push(Target::MetalMacOS),
                    "metal-ios" => parsed_opts.targets.push(Target::MetalIOS),
                    "nvptx" => parsed_opts.targets.push(Target::NVPTX),
                    "amdgpu" => parsed_opts.targets.push(Target::AMDGPU),
                    "opencl-spirv" => parsed_opts.targets.push(Target::OpenCLSPIRV),
                    _ => {
                        return Err(format!("Unsupported target: {}", t));
                    }
                }
            }
            i += 1;
            continue;
        }

        if options[i].starts_with("--stage=") {
            let stage: String = options[i].chars().skip(8).collect();
            match stage.as_str() {
                "binary" => parsed_opts.stage = Stage::Binary,
                "mlir" => parsed_opts.stage = Stage::MLIR,
                "llvm" => parsed_opts.stage = Stage::LLVMIR,
                _ => {
                    return Err(format!("Unsupported stage: {}", stage));
                }
            }
            i += 1;
            continue;
        }

        if !options[i].starts_with("-") {
            parsed_opts.inputs.push(options[i].to_owned());
            i += 1;
            continue;
        }

        match options[i].as_str() {
            "-cl-kernel-arg-info" => parsed_opts.kernel_arg_info = true,
            "-cl-mad-enable" => parsed_opts.mad_enable = true,
            "-cl-opt-disable" => parsed_opts.opt_level = OptLevel::OptNone,
            "-print-before-all-mlir" => parsed_opts.print_before_all_mlir = true,
            "-print-after-all-mlir" => parsed_opts.print_after_all_mlir = true,
            "-print-before-all-llvm" => parsed_opts.print_before_all_llvm = true,
            "-print-after-all-llvm" => parsed_opts.print_after_all_llvm = true,
            "-O0" => parsed_opts.opt_level = OptLevel::OptNone,
            "-O1" => parsed_opts.opt_level = OptLevel::O1,
            "-O2" => parsed_opts.opt_level = OptLevel::O2,
            "-O3" => parsed_opts.opt_level = OptLevel::O3,
            "-c" => parsed_opts.compile_only = true,
            _ => return Err(format!("Unsupported compiler option: {}", options[i])),
        }

        i += 1;
    }

    Ok(parsed_opts)
}

pub fn get_help() -> String {
    "LibreCL compiler

Options guide:
    -c compile only
    --stage output stage. Valid values are:
            * binary
            * mlir
            * llvm
    --targets [target] create binary for specific target. Supported targets
            * vulkan-spirv
            * metal-macos
            * metal-ios
            * nvptx
            * amdgpu
            * opencl-spirv
    -cl-opt-disable disable all compiler optimizations
    -On choose compiler optimization level. n is one of 0, 1, 2, 3
    -print-before-all-mlir
    -print-after-all-mlir
    -print-before-all-llvm
    -print-after-all-llvm
    "
    .to_owned()
}

#[cfg(test)]
mod test {
    use crate::{parse_options, OptLevel, Target};

    #[test]
    fn positive() {
        let input: Vec<String> = "--targets=nvptx,vulkan-spirv,metal-ios -D test=1 -Dtest2 -D test3 -w -cl-opt-disable -print-after-all-mlir".split_whitespace().map(|s| s.to_owned()).collect();
        let opts = parse_options(&input).expect("failed to parse options");

        assert!(opts.opt_level == OptLevel::O2);
        assert!(opts.print_after_all_mlir == true);
        assert!(opts.targets.contains(&Target::NVPTX));
        assert!(opts.targets.contains(&Target::VulkanSPIRV));
        assert!(opts.targets.contains(&Target::MetalIOS));
        assert!(opts.other_options.contains(&"-D".to_owned()));
        assert!(opts.other_options.contains(&"-Dtest2".to_owned()));
        assert!(opts.other_options.contains(&"test=1".to_owned()));
        assert!(opts.other_options.contains(&"test3".to_owned()));
        assert!(opts.other_options.contains(&"-w".to_owned()));
    }

    #[test]
    fn negative_target() {
        let input: Vec<String> = "--targets=unknown"
            .split_whitespace()
            .map(|s| s.to_owned())
            .collect();
        let opts = parse_options(&input);
        match opts {
            Ok(_) => assert!(false),
            Err(err) => assert!(err.starts_with("Unsupported target: unknown")),
        }
    }

    #[test]
    fn negative_flag() {
        let input: Vec<String> = "-foo".split_whitespace().map(|s| s.to_owned()).collect();
        let opts = parse_options(&input);
        match opts {
            Ok(_) => assert!(false),
            Err(err) => assert!(err.starts_with("Unsupported compiler option: -foo")),
        }
    }
}
