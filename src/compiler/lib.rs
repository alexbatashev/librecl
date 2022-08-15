mod ffi;

use std::{ffi::CString, sync::Arc};

pub struct CompileResult {
    handle: *mut ffi::lcl_CompileResult,
}

unsafe impl Send for CompileResult {}
unsafe impl Sync for CompileResult {}

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

    pub fn compile_source(&self, source: &str, options: &[String]) -> Arc<CompileResult> {
        let mut c_opts = vec![];
        let mut char_opts = vec![];

        for o in options {
            c_opts.push(CString::new(o.as_str()).unwrap());
            char_opts.push(c_opts.last().unwrap().as_ptr());
        }

        let c_source = CString::new(source).unwrap();

        let result = unsafe {
            ffi::lcl_compile(
                self.handle,
                source.len() as ffi::size_t,
                c_source.as_ptr(),
                char_opts.len() as ffi::size_t,
                char_opts.as_ptr() as *mut *const i8,
            )
        };

        CompileResult::from_raw(result)
    }

    pub fn compile_spirv(&self, source: &[i8], options: &[String]) -> Arc<CompileResult> {
        let mut c_opts = vec![];
        let mut char_opts = vec![];

        for o in options {
            c_opts.push(CString::new(o.as_str()).unwrap());
            char_opts.push(c_opts.last().unwrap().as_ptr());
        }

        let result = unsafe {
            ffi::lcl_compile(
                self.handle,
                source.len() as ffi::size_t,
                source.as_ptr(),
                char_opts.len() as ffi::size_t,
                char_opts.as_ptr() as *mut *const i8,
            )
        };

        CompileResult::from_raw(result)
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
            ffi::lcl_link(
                self.handle,
                modules.len() as ffi::size_t,
                c_modules.as_mut_ptr(),
                char_opts.len() as ffi::size_t,
                char_opts.as_ptr() as *mut *const i8,
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
