#![allow(non_camel_case_types)]
mod ffi;

use ocl_args::{CompilerArgs, OptLevel};
use std::{ffi::CString, sync::Arc};

type get_compiler_t = unsafe extern "C" fn() -> *mut ffi::lcl_Compiler;
type release_compiler_t = unsafe extern "C" fn(*mut ffi::lcl_Compiler);
type compile_t = unsafe extern "C" fn(
    *mut ffi::lcl_Compiler,
    ffi::size_t,
    *const i8,
    ffi::Options,
) -> *mut ffi::lcl_CompileResult;
type link_t = unsafe extern "C" fn(
    *mut ffi::lcl_Compiler,
    ffi::size_t,
    *mut *mut ffi::lcl_CompileResult,
    ffi::Options,
) -> *mut ffi::lcl_CompileResult;
type release_result_t = unsafe extern "C" fn(*mut ffi::lcl_CompileResult);
type get_num_kernels_t = unsafe extern "C" fn(*mut ffi::lcl_CompileResult) -> ffi::size_t;
type get_num_kernel_args_t =
    unsafe extern "C" fn(*mut ffi::lcl_CompileResult, ffi::size_t) -> ffi::size_t;
type get_kernel_args_t =
    unsafe extern "C" fn(*mut ffi::lcl_CompileResult, ffi::size_t, *mut libc::c_void);
type get_kernel_name_t =
    unsafe extern "C" fn(*mut ffi::lcl_CompileResult, ffi::size_t) -> *const i8;
type get_program_size_t = unsafe extern "C" fn(*mut ffi::lcl_CompileResult) -> ffi::size_t;
type copy_program_t = unsafe extern "C" fn(*mut ffi::lcl_CompileResult, *mut libc::c_void);

type is_error_t = unsafe extern "C" fn(*mut ffi::lcl_CompileResult) -> libc::c_int;
type get_error_message_t = unsafe extern "C" fn(*mut ffi::lcl_CompileResult) -> *const i8;

struct Context {
    library: Option<Arc<libloading::Library>>,
}

impl Context {
    pub fn default() -> Context {
        Context { library: None }
    }

    pub fn get_compiler(&self) -> Result<*mut ffi::lcl_Compiler, String> {
        let library = self
            .library
            .as_ref()
            .ok_or("Compiler not available".to_string())?;
        let sym = unsafe { library.get::<get_compiler_t>("lcl_get_compiler".as_bytes()) }
            .map_err(|_| "Failed to find symbol".to_owned())?;
        Ok(unsafe { sym() })
    }

    pub fn release_compiler(&self, compiler: *mut ffi::lcl_Compiler) -> Result<(), String> {
        let library = self
            .library
            .as_ref()
            .ok_or("Compiler not available".to_string())?;
        let sym = unsafe { library.get::<release_compiler_t>("lcl_release_compiler".as_bytes()) }
            .map_err(|_| "Failed to find symbol".to_owned())?;
        Ok(unsafe { sym(compiler) })
    }

    pub fn compile(
        &self,
        compiler: *mut ffi::lcl_Compiler,
        src_size: ffi::size_t,
        source: *const i8,
        opts: ffi::Options,
    ) -> Result<*mut ffi::lcl_CompileResult, String> {
        let library = self
            .library
            .as_ref()
            .ok_or("Compiler not available".to_string())?;
        let sym = unsafe { library.get::<compile_t>("lcl_compile".as_bytes()) }
            .map_err(|_| "Failed to find symbol".to_owned())?;
        Ok(unsafe { sym(compiler, src_size, source, opts) })
    }

    pub fn link(
        &self,
        compiler: *mut ffi::lcl_Compiler,
        num_modules: ffi::size_t,
        modules: *mut *mut ffi::lcl_CompileResult,
        opts: ffi::Options,
    ) -> Result<*mut ffi::lcl_CompileResult, String> {
        let library = self
            .library
            .as_ref()
            .ok_or("Compiler not available".to_string())?;
        let sym = unsafe { library.get::<link_t>("lcl_link".as_bytes()) }
            .map_err(|_| "Failed to find symbol".to_owned())?;
        Ok(unsafe { sym(compiler, num_modules, modules, opts) })
    }

    pub fn release_result(&self, res: *mut ffi::lcl_CompileResult) -> Result<(), String> {
        let library = self
            .library
            .as_ref()
            .ok_or("Compiler not available".to_string())?;
        let sym = unsafe { library.get::<release_result_t>("lcl_release_result".as_bytes()) }
            .map_err(|_| "Failed to find symbol".to_owned())?;
        Ok(unsafe { sym(res) })
    }

    pub fn get_num_kernels(&self, res: *mut ffi::lcl_CompileResult) -> Result<ffi::size_t, String> {
        let library = self
            .library
            .as_ref()
            .ok_or("Compiler not available".to_string())?;
        let sym = unsafe { library.get::<get_num_kernels_t>("lcl_get_num_kernels".as_bytes()) }
            .map_err(|_| "Failed to find symbol".to_owned())?;
        Ok(unsafe { sym(res) })
    }

    pub fn get_num_kernel_args(
        &self,
        res: *mut ffi::lcl_CompileResult,
        idx: ffi::size_t,
    ) -> Result<ffi::size_t, String> {
        let library = self
            .library
            .as_ref()
            .ok_or("Compiler not available".to_string())?;
        let sym =
            unsafe { library.get::<get_num_kernel_args_t>("lcl_get_num_kernel_args".as_bytes()) }
                .map_err(|_| "Failed to find symbol".to_owned())?;
        Ok(unsafe { sym(res, idx) })
    }

    pub fn get_kernel_args(
        &self,
        res: *mut ffi::lcl_CompileResult,
        idx: ffi::size_t,
        data: *mut libc::c_void,
    ) -> Result<(), String> {
        let library = self
            .library
            .as_ref()
            .ok_or("Compiler not available".to_string())?;
        let sym = unsafe { library.get::<get_kernel_args_t>("lcl_get_kernel_args".as_bytes()) }
            .map_err(|_| "Failed to find symbol".to_owned())?;
        Ok(unsafe { sym(res, idx, data) })
    }

    pub fn get_kernel_name(
        &self,
        res: *mut ffi::lcl_CompileResult,
        idx: ffi::size_t,
    ) -> Result<*const i8, String> {
        let library = self
            .library
            .as_ref()
            .ok_or("Compiler not available".to_string())?;
        let sym = unsafe { library.get::<get_kernel_name_t>("lcl_get_kernel_name".as_bytes()) }
            .map_err(|_| "Failed to find symbol".to_owned())?;
        Ok(unsafe { sym(res, idx) })
    }

    pub fn get_program_size(
        &self,
        res: *mut ffi::lcl_CompileResult,
    ) -> Result<ffi::size_t, String> {
        let library = self
            .library
            .as_ref()
            .ok_or("Compiler not available".to_string())?;
        let sym = unsafe { library.get::<get_program_size_t>("lcl_get_program_size".as_bytes()) }
            .map_err(|_| "Failed to find symbol".to_owned())?;
        Ok(unsafe { sym(res) })
    }

    pub fn copy_program(
        &self,
        res: *mut ffi::lcl_CompileResult,
        data: *mut libc::c_void,
    ) -> Result<(), String> {
        let library = self
            .library
            .as_ref()
            .ok_or("Compiler not available".to_string())?;
        let sym = unsafe { library.get::<copy_program_t>("lcl_copy_program".as_bytes()) }
            .map_err(|_| "Failed to find symbol".to_owned())?;
        Ok(unsafe { sym(res, data) })
    }

    pub fn is_error(&self, res: *mut ffi::lcl_CompileResult) -> Result<libc::c_int, String> {
        let library = self
            .library
            .as_ref()
            .ok_or("Compiler not available".to_string())?;
        let sym = unsafe { library.get::<is_error_t>("lcl_is_error".as_bytes()) }
            .map_err(|_| "Failed to find symbol".to_owned())?;
        Ok(unsafe { sym(res) })
    }

    pub fn get_error_message(&self, res: *mut ffi::lcl_CompileResult) -> Result<*const i8, String> {
        let library = self
            .library
            .as_ref()
            .ok_or("Compiler not available".to_string())?;
        let sym = unsafe { library.get::<get_error_message_t>("lcl_get_error_message".as_bytes()) }
            .map_err(|_| "Failed to find symbol".to_owned())?;
        Ok(unsafe { sym(res) })
    }

    pub fn from_library(library: Arc<libloading::Library>) -> Context {
        Context {
            library: Some(library.clone()),
        }
    }

    pub fn new() -> Arc<Context> {
        #[cfg(target_os = "linux")]
        let maybe_library = unsafe { libloading::Library::new("liblcl_compiler.so") };
        #[cfg(target_os = "macos")]
        let maybe_library = unsafe { libloading::Library::new("liblcl_compiler.dylib") };

        if maybe_library.is_err() {
            return Arc::new(Context::default());
        }

        let library = Arc::new(maybe_library.unwrap());

        Arc::new(Context::from_library(library))
    }

    pub fn is_available(&self) -> bool {
        self.library.is_some()
    }
}

pub struct CompileResult {
    context: Arc<Context>,
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
    fn from_raw(context: Arc<Context>, handle: *mut ffi::lcl_CompileResult) -> Arc<CompileResult> {
        Arc::new(CompileResult { context, handle })
    }

    pub fn is_ok(&self) -> bool {
        return self.context.is_error(self.handle).unwrap() == 0;
    }

    pub fn get_error(&self) -> String {
        let err_str = self.context.get_error_message(self.handle).unwrap();
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

        let num_kernels = self.context.get_num_kernels(self.handle).unwrap();

        kernels.reserve(num_kernels as usize);

        for kidx in 0..num_kernels {
            let num_args = self.context.get_num_kernel_args(self.handle, kidx).unwrap();
            let mut args: Vec<KernelArgInfo> = vec![];
            args.resize(
                num_args as usize,
                KernelArgInfo {
                    arg_type: KernelArgType::POD,
                    index: 0,
                    size: 0,
                },
            );
            self.context
                .get_kernel_args(self.handle, kidx, args.as_mut_ptr() as *mut libc::c_void)
                .unwrap();
            let name = unsafe {
                std::ffi::CStr::from_ptr(self.context.get_kernel_name(self.handle, kidx).unwrap())
            }
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
        let bin_size = self.context.get_program_size(self.handle).unwrap();

        let mut data: Vec<u8> = vec![];
        data.resize(bin_size as usize, 0);

        self.context
            .copy_program(self.handle, data.as_mut_ptr() as *mut libc::c_void)
            .expect("Failed to copy program");

        return data;
    }
}

impl Drop for CompileResult {
    fn drop(&mut self) {
        if self.context.is_available() {
            self.context
                .release_result(self.handle)
                .expect("Failed to release compile result");
        }
    }
}

pub struct Compiler {
    context: Arc<Context>,
    handle: *mut ffi::lcl_Compiler,
}

unsafe impl Send for Compiler {}
unsafe impl Sync for Compiler {}

impl Compiler {
    pub fn new() -> Arc<Compiler> {
        let context = Context::new();
        let handle = context.get_compiler().unwrap();
        Arc::new(Compiler { context, handle })
    }

    // TODO use dispatch table and dynamic library loading
    pub fn is_available(&self) -> bool {
        self.context.is_available()
    }

    pub fn compile_source(&self, source: &str, options: &CompilerArgs) -> Arc<CompileResult> {
        let c_source = CString::new(source).unwrap();

        let c_opts = OptionsWrapper::from(options);

        let result = self
            .context
            .compile(
                self.handle,
                source.len() as ffi::size_t,
                c_source.as_ptr(),
                c_opts.options,
            )
            .unwrap();

        CompileResult::from_raw(self.context.clone(), result)
    }

    pub fn compile_spirv(&self, source: &[i8], options: &CompilerArgs) -> Arc<CompileResult> {
        let c_opts = OptionsWrapper::from(options);

        let result = self
            .context
            .compile(
                self.handle,
                source.len() as ffi::size_t,
                source.as_ptr(),
                c_opts.options,
            )
            .unwrap();

        CompileResult::from_raw(self.context.clone(), result)
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

        let result = self
            .context
            .link(
                self.handle,
                modules.len() as ffi::size_t,
                c_modules.as_mut_ptr(),
                c_opts.options,
            )
            .unwrap();

        CompileResult::from_raw(self.context.clone(), result)
    }
}

impl Drop for Compiler {
    fn drop(&mut self) {
        if self.context.is_available() {
            self.context
                .release_compiler(self.handle)
                .expect("Failed to release compiler");
        }
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
