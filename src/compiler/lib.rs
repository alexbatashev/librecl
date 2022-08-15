mod ffi;

use ocl_args::{CompilerArgs, OptLevel};
use std::{ffi::CString, sync::Arc};

pub struct CompileResult {
    handle: *mut ffi::lcl_CompileResult,
}

unsafe impl Send for CompileResult {}
unsafe impl Sync for CompileResult {}

struct OptionsWrapper {
    pub options: ffi::Options,
    _cstr_opts: Vec<std::ffi::CString>,
    _c_opts: Vec<*const i8>,
}

impl From<&CompilerArgs> for OptionsWrapper {
    fn from(args: &CompilerArgs) -> Self {
        let opt_level: i32 = match args.opt_level {
            OptLevel::OptNone => 0,
            OptLevel::O1 => 1,
            OptLevel::O2 => 2,
            OptLevel::O3 => 3,
        };

        let mut cstr_opts: Vec<_> = vec![];
        let mut c_opts: Vec<_> = vec![];

        for opt in &args.other_options {
            cstr_opts.push(std::ffi::CString::new(opt.clone()).unwrap());
            c_opts.push(cstr_opts.last().as_ref().unwrap().as_ptr());
        }

        let options = ffi::Options {
            compile_only: args.compile_only,
            target_vulkan_spv: args.targets.contains(&ocl_args::Target::VulkanSPIRV),
            target_opencl_spv: args.targets.contains(&ocl_args::Target::OpenCLSPIRV),
            target_metal_macos: args.targets.contains(&ocl_args::Target::MetalMacOS),
            target_metal_ios: args.targets.contains(&ocl_args::Target::MetalIOS),
            target_nvptx: args.targets.contains(&ocl_args::Target::NVPTX),
            target_amdgpu: args.targets.contains(&ocl_args::Target::AMDGPU),
            print_before_mlir: args.print_before_all_mlir,
            print_after_mlir: args.print_after_all_mlir,
            print_before_llvm: args.print_before_all_llvm,
            print_after_llvm: args.print_after_all_llvm,
            opt_level,
            mad_enable: args.mad_enable,
            kernel_arg_info: args.kernel_arg_info,
            other_options: c_opts.as_mut_ptr(),
            num_other_options: c_opts.len() as u64,
        };

        OptionsWrapper {
            options,
            _cstr_opts: cstr_opts,
            _c_opts: c_opts,
        }
    }
}

impl CompileResult {
    fn from_raw(handle: *mut ffi::lcl_CompileResult) -> Arc<CompileResult> {
        Arc::new(CompileResult { handle })
    }

    pub fn is_ok(&self) -> bool {
        return unsafe { ffi::lcl_is_error(self.handle) == 0 };
    }

    pub fn get_error(&self) -> String {
        let err_str = unsafe { ffi::lcl_get_error_message(self.handle) };
        return unsafe { std::ffi::CStr::from_ptr(err_str) }
            .to_str()
            .unwrap()
            .to_owned();
    }

    pub fn get_kernels(&self) -> Vec<KernelInfo> {
        if !self.is_ok() {
            panic!("Invalid binary");
        }
        let mut kernels: Vec<KernelInfo> = vec![];

        let num_kernels = unsafe { ffi::lcl_get_num_kernels(self.handle) };

        kernels.reserve(num_kernels as usize);

        for kidx in 0..num_kernels {
            let num_args = unsafe { ffi::lcl_get_num_kernel_args(self.handle, kidx) };
            let mut args: Vec<KernelArgInfo> = vec![];
            args.resize(
                num_args as usize,
                KernelArgInfo {
                    arg_type: KernelArgType::POD,
                    index: 0,
                    size: 0,
                },
            );
            unsafe {
                ffi::lcl_get_kernel_args(self.handle, kidx, args.as_mut_ptr() as *mut libc::c_void)
            };
            let name =
                unsafe { std::ffi::CStr::from_ptr(ffi::lcl_get_kernel_name(self.handle, kidx)) }
                    .to_str()
                    .unwrap()
                    .to_owned();

            kernels.push(KernelInfo {
                name,
                arguments: args,
            });
        }

        return kernels;
    }

    pub fn get_binary(&self) -> Vec<u8> {
        let bin_size = unsafe { ffi::lcl_get_program_size(self.handle) };

        let mut data: Vec<u8> = vec![];
        data.resize(bin_size as usize, 0);

        unsafe { ffi::lcl_copy_program(self.handle, data.as_mut_ptr() as *mut libc::c_void) };

        return data;
    }
}

pub struct Compiler {
    handle: *mut ffi::lcl_Compiler,
}

unsafe impl Send for Compiler {}
unsafe impl Sync for Compiler {}

impl Compiler {
    pub fn new() -> Arc<Compiler> {
        let handle = unsafe { ffi::lcl_get_compiler() };
        Arc::new(Compiler { handle })
    }

    // TODO use dispatch table and dynamic library loading
    pub fn is_available(&self) -> bool {
        true
    }

    pub fn compile_source(&self, source: &str, options: &CompilerArgs) -> Arc<CompileResult> {
        let c_source = CString::new(source).unwrap();

        let c_opts = OptionsWrapper::from(options);

        let result = unsafe {
            ffi::lcl_compile(
                self.handle,
                source.len() as ffi::size_t,
                c_source.as_ptr(),
                c_opts.options,
            )
        };

        CompileResult::from_raw(result)
    }

    pub fn link(
        &self,
        modules: &[Arc<CompileResult>],
        options: &CompilerArgs,
    ) -> Arc<CompileResult> {
        let mut c_modules = vec![];
        let c_opts = OptionsWrapper::from(options);

        for m in modules {
            c_modules.push(m.handle);
        }

        let result = unsafe {
            ffi::lcl_link(
                self.handle,
                modules.len() as ffi::size_t,
                c_modules.as_mut_ptr(),
                c_opts.options,
            )
        };

        CompileResult::from_raw(result)
    }
}

impl Drop for Compiler {
    fn drop(&mut self) {
        unsafe { ffi::lcl_release_compiler(self.handle) }
    }
}

// The following defenitions must be the same as in kernel_info.hpp

#[derive(Clone)]
#[repr(u32)]
pub enum KernelArgType {
    GlobalBuffer,
    USMPointer,
    Image,
    POD,
}

#[derive(Clone)]
#[repr(C)]
pub struct KernelArgInfo {
    pub arg_type: KernelArgType,
    pub index: u64,
    pub size: libc::size_t,
}

pub struct KernelInfo {
    pub name: String,
    pub arguments: Vec<KernelArgInfo>,
}
