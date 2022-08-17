#![allow(non_camel_case_types)]
mod ffi;

use libloading::Symbol;
use std::{borrow::Borrow, ffi::CString, sync::Arc};

type get_compiler_t = unsafe extern "C" fn() -> *mut ffi::lcl_Compiler;
type release_compiler_t = unsafe extern "C" fn(*mut ffi::lcl_Compiler);
type compile_t = unsafe extern "C" fn(
    *mut ffi::lcl_Compiler,
    ffi::size_t,
    *const i8,
    ffi::size_t,
    *mut *const i8,
) -> *mut ffi::lcl_CompileResult;
type link_t = unsafe extern "C" fn(
    *mut ffi::lcl_Compiler,
    ffi::size_t,
    *mut *mut ffi::lcl_CompileResult,
    ffi::size_t,
    *mut *const i8,
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

struct Context<'a> {
    library: Option<Arc<libloading::Library>>,
    pub get_compiler: Option<Symbol<'a, get_compiler_t>>,
    pub release_compiler: Option<Symbol<'a, release_compiler_t>>,
    pub compile: Option<Symbol<'a, compile_t>>,
    pub link: Option<Symbol<'a, link_t>>,
    pub release_result: Option<Symbol<'a, release_result_t>>,
    pub get_num_kernels: Option<Symbol<'a, get_num_kernels_t>>,
    pub get_num_kernel_args: Option<Symbol<'a, get_num_kernel_args_t>>,
    pub get_kernel_args: Option<Symbol<'a, get_kernel_args_t>>,
    pub get_kernel_name: Option<Symbol<'a, get_kernel_name_t>>,
    pub get_program_size: Option<Symbol<'a, get_program_size_t>>,
    pub copy_program: Option<Symbol<'a, copy_program_t>>,
    pub is_error: Option<Symbol<'a, is_error_t>>,
    pub get_error_message: Option<Symbol<'a, get_error_message_t>>,
}

impl<'a> Context<'a> {
    pub fn default() -> Context<'static> {
        Context {
            library: None,
            get_compiler: None,
            release_compiler: None,
            compile: None,
            link: None,
            release_result: None,
            get_num_kernels: None,
            get_num_kernel_args: None,
            get_kernel_args: None,
            get_kernel_name: None,
            get_program_size: None,
            copy_program: None,
            is_error: None,
            get_error_message: None,
        }
    }

    pub fn get_compiler(&self) -> Result<*mut ffi::lcl_Compiler, String> {
        let library = self.library.ok_or("Compiler not available".to_string())?;
        let sym = unsafe { library.get::<get_compiler_t>("lcl_get_compiler") }?;
        Ok(unsafe { sym() })
    }

    pub fn release_compiler(&self, compiler: *mut ffi::lcl_Compiler) -> Result<(), String> {
        let library = self.library.ok_or("Compiler not available".to_string())?;
        let sym = unsafe { library.get::<release_compiler_t>("lcl_release_compiler") }?;
        Ok(unsafe { sym(compiler) })
    }

    pub fn compile(
        &self,
        compiler: *mut ffi::lcl_Compiler,
        src_size: ffi::size_t,
        source: *const i8,
        num_opts: ffi::size_t,
        opts: *mut *const i8,
    ) -> Result<*mut ffi::lcl_CompileResult, String> {
        let library = self.library.ok_or("Compiler not available".to_string())?;
        let sym = unsafe { library.get::<compile_t>("lcl_compile") }?;
        Ok(unsafe { sym(compiler, src_size, source, num_opts, opts) })
    }

    pub fn link(
        &self,
        compiler: *mut ffi::lcl_Compiler,
        num_modules: ffi::size_t,
        modules: *mut *mut ffi::lcl_CompileResult,
        num_opts: ffi::size_t,
        opts: *mut *const i8,
    ) -> Result<*mut ffi::lcl_CompileResult, String> {
        let library = self.library.ok_or("Compiler not available".to_string())?;
        let sym = unsafe { library.get::<link_t>("lcl_link") }?;
        Ok(unsafe { sym(compiler, num_modules, modules, num_opts, opts) })
    }

    pub fn release_result(&self, res: *mut ffi::lcl_CompileResult) -> Result<(), String> {
        let library = self.library.ok_or("Compiler not available".to_string())?;
        let sym = unsafe { library.get::<release_result_t>("lcl_release_result") }?;
        Ok(unsafe { sym(res) })
    }

    pub fn get_num_kernels(&self, res: *mut ffi::lcl_CompileResult) -> Result<ffi::size_t, String> {

    }

    fn get_symbol<T>(&'a self, name: &'static str) -> Option<Symbol<'a, T>> {
        unsafe {
            self.library
                .as_ref()
                .unwrap()
                .get::<T>(name.as_bytes())
                .ok()
        }
    }

    pub fn from_library(library: Arc<libloading::Library>) -> Context<'static> {
        let mut context = Context {
            library: Some(library.clone()),
            get_compiler: None,
            release_compiler: None,
            compile: None,
            link: None,
            release_result: None,
            get_num_kernels: None,
            get_num_kernel_args: None,
            get_kernel_args: None,
            get_kernel_name: None,
            get_program_size: None,
            copy_program: None,
            is_error: None,
            get_error_message: None,
        };

        context.get_compiler = context.borrow().get_symbol("lcl_get_compiler");
        /*
        context.release_compiler =
            context.get_symbol("lcl_release_compiler");
        context.compile = context.get_symbol("lcl_compile");
        context.link = context.get_symbol("lcl_link");
        context.release_result =
            context.get_symbol("lcl_release_result");
        context.get_num_kernels =
            context.get_symbol("num_get_num_kernels");
        context.get_num_kernel_args =
            context.get_symbol("lcl_get_num_kernel_args");
        context.get_kernel_args =
            context.get_symbol("lcl_get_kernel_args");
        context.get_kernel_name = context.get_symbol("get_kernel_name");
        context.get_program_size =
            context.get_symbol("lcl_get_program_size");
        context.copy_program = context.get_symbol("lcl_copy_program");
        context.is_error = context.get_symbol("lcl_is_error");
        context.get_error_message =
            context.get_symbol("lcl_get_error_message");
            */

        context
    }

    pub fn new() -> Arc<Context<'static>> {
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
    context: Arc<Context<'static>>,
    handle: *mut ffi::lcl_CompileResult,
}

unsafe impl Send for CompileResult {}
unsafe impl Sync for CompileResult {}

impl CompileResult {
    fn from_raw(
        context: Arc<Context<'static>>,
        handle: *mut ffi::lcl_CompileResult,
    ) -> Arc<CompileResult> {
        Arc::new(CompileResult { context, handle })
    }

    pub fn is_ok(&self) -> bool {
        return unsafe { self.context.is_error.unwrap()(self.handle) == 0 };
    }

    pub fn get_error(&self) -> String {
        let err_str = unsafe { self.context.get_error_message.unwrap()(self.handle) };
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

        let num_kernels = unsafe { self.context.get_num_kernels.unwrap()(self.handle) };

        kernels.reserve(num_kernels as usize);

        for kidx in 0..num_kernels {
            let num_args = unsafe { self.context.get_num_kernel_args.unwrap()(self.handle, kidx) };
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
                self.context.get_kernel_args.unwrap()(
                    self.handle,
                    kidx,
                    args.as_mut_ptr() as *mut libc::c_void,
                )
            };
            let name = unsafe {
                std::ffi::CStr::from_ptr(self.context.get_kernel_name.unwrap()(self.handle, kidx))
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
        let bin_size = unsafe { self.context.get_program_size.unwrap()(self.handle) };

        let mut data: Vec<u8> = vec![];
        data.resize(bin_size as usize, 0);

        unsafe {
            self.context.copy_program.unwrap()(self.handle, data.as_mut_ptr() as *mut libc::c_void)
        };

        return data;
    }
}

pub struct Compiler {
    context: Arc<Context<'static>>,
    handle: *mut ffi::lcl_Compiler,
}

unsafe impl Send for Compiler {}
unsafe impl Sync for Compiler {}

impl Compiler {
    pub fn new() -> Arc<Compiler> {
        let context = Context::new();
        let handle = unsafe { context.get_compiler.unwrap()() };
        Arc::new(Compiler { context, handle })
    }

    // TODO use dispatch table and dynamic library loading
    pub fn is_available(&self) -> bool {
        self.context.is_available()
    }

    pub fn compile_source(&self, source: &str, options: &[String]) -> Arc<CompileResult> {
        let mut c_opts = vec![];
        let mut char_opts = vec![];

        for o in options {
            c_opts.push(CString::new(o.as_str()).unwrap());
            char_opts.push(c_opts.last().unwrap().as_ptr());
        }

        let c_source = CString::new(source).unwrap();

        let result = unsafe {
            self.context.compile.unwrap()(
                self.handle,
                source.len() as ffi::size_t,
                c_source.as_ptr(),
                char_opts.len() as ffi::size_t,
                char_opts.as_ptr() as *mut *const i8,
            )
        };

        CompileResult::from_raw(self.context.clone(), result)
    }

    pub fn compile_spirv(&self, source: &[i8], options: &[String]) -> Arc<CompileResult> {
        let mut c_opts = vec![];
        let mut char_opts = vec![];

        for o in options {
            c_opts.push(CString::new(o.as_str()).unwrap());
            char_opts.push(c_opts.last().unwrap().as_ptr());
        }

        let result = unsafe {
            self.context.compile.unwrap()(
                self.handle,
                source.len() as ffi::size_t,
                source.as_ptr(),
                char_opts.len() as ffi::size_t,
                char_opts.as_ptr() as *mut *const i8,
            )
        };

        CompileResult::from_raw(self.context.clone(), result)
    }

    pub fn link(&self, modules: &[Arc<CompileResult>], options: &[String]) -> Arc<CompileResult> {
        let mut c_opts = vec![];
        let mut char_opts = vec![];

        for o in options {
            c_opts.push(CString::new(o.as_str()).unwrap());
            char_opts.push(c_opts.last().unwrap().as_ptr());
        }

        let mut c_modules = vec![];

        for m in modules {
            c_modules.push(m.handle);
        }

        let result = unsafe {
            self.context.link.unwrap()(
                self.handle,
                modules.len() as ffi::size_t,
                c_modules.as_mut_ptr(),
                char_opts.len() as ffi::size_t,
                char_opts.as_ptr() as *mut *const i8,
            )
        };

        CompileResult::from_raw(self.context.clone(), result)
    }
}

impl Drop for Compiler {
    fn drop(&mut self) {
        unsafe { self.context.release_compiler.unwrap()(self.handle) }
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
