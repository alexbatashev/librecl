use librecl_compiler::Compiler;
use ocl_args::{get_help, parse_options};
use std::env;
use std::fs::{self, File};
use std::io::Write;

const COMMON_BUILTINS: &str = include_str!("../../../src/runtime/builtin/common.mlir");

fn main() {
    let args: Vec<String> = env::args().into_iter().skip(1).collect();

    if args.len() == 0 || args[0] == "-h" || args[0] == "--help" {
        println!("{}", get_help());
        return;
    }

    let maybe_opts = parse_options(&args);
    match maybe_opts {
        Ok(ref options) => {
            // Only one input is supported right now.
            println!("Compiling {}", options.inputs.first().unwrap());
            let input = fs::read_to_string(options.inputs.first().unwrap())
                .expect("Failed to read from a file");
            let compiler = Compiler::new();
            let compile_result = compiler.compile_source(&input, options);

            if !compile_result.is_ok() {
                let error = compile_result.get_error();
                println!("Failed to compile source file:\n{}", error);
            }

            let builtins = compiler.compile_mlir(COMMON_BUILTINS, options);

            if !builtins.is_ok() {
                let error = builtins.get_error();
                println!("Failed to compile builtins:\n{}", error);
            }

            let result = compiler.link(&[compile_result, builtins], options);

            if result.is_ok() {
                let mut out = File::create(&options.out).expect("Failed to create a file");
                let binary = result.get_binary();
                out.write(&binary).expect("Failed to write a file");
            } else {
                let error = result.get_error();
                println!("Failed to compile source file:\n{}", error);
            }
        }
        Err(err) => println!("{}", err),
    }
}
