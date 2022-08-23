use librecl_compiler::{CompileResult, Compiler};
use ocl_args::{get_help, parse_options, CompilerArgs};
use std::env;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::sync::Arc;

const COMMON_BUILTINS: &str = include_str!("../../../src/runtime/builtin/common.mlir");

fn compile_source(
    name: &str,
    compiler: &Arc<Compiler>,
    options: &CompilerArgs,
) -> Arc<CompileResult> {
    let input = fs::read_to_string(name).expect("Failed to read from a file");
    compiler.compile_source(&input, options)
}

fn compile_spirv(
    name: &str,
    compiler: &Arc<Compiler>,
    options: &CompilerArgs,
) -> Arc<CompileResult> {
    let mut input: Vec<u8> = vec![];
    let mut file = File::open(name).expect("Failed to open the file");
    file.read_to_end(&mut input)
        .expect("Failed to read from the file");
    compiler.compile_spirv(unsafe { std::mem::transmute(input.as_slice()) }, options)
}

fn main() {
    let args: Vec<String> = env::args().into_iter().skip(1).collect();

    if args.len() == 0 || args[0] == "-h" || args[0] == "--help" {
        println!("{}", get_help());
        return;
    }

    let maybe_opts = parse_options(&args);
    match maybe_opts {
        Ok(ref options) => {
            let mut modules = vec![];

            let compiler = Compiler::new();

            let builtins = compiler.compile_mlir(COMMON_BUILTINS, options);

            if !builtins.is_ok() {
                let error = builtins.get_error();
                println!("Failed to compile builtins:\n{}", error);
            }

            modules.push(builtins);

            for inp in &options.inputs {
                println!("Compiling {}", inp);
                let module = if inp.ends_with("spv") {
                    compile_spirv(inp, &compiler, options)
                } else {
                    compile_source(inp, &compiler, options)
                };

                if module.is_ok() {
                    modules.push(module);
                } else {
                    let error = module.get_error();
                    println!("Failed to compile source file:\n{}", error);
                }
            }

            let result = compiler.link(&modules, options);

            if result.is_ok() {
                let mut out = File::create(&options.out).expect("Failed to create a file");
                let binary = result.get_binary();
                out.write(&binary).expect("Failed to write a file");
            } else {
                let error = result.get_error();
                println!("Failed to compile:\n{}", error);
            }
        }
        Err(err) => println!("{}", err),
    }
}
