#[link(name = "lcl_compiler", kind = "dylib")]
extern "C" {
    fn create_clang_frontend() -> *mut libc::c_void;
    fn release_clang_frontend(fe: *mut libc::c_void);
    fn process_source(
        fe: *mut libc::c_void,
        source: *const libc::c_char,
        options: *const *const libc::c_char,
        num_options: libc::size_t,
    ) -> *mut libc::c_void;
    fn release_result(fe_res: *mut libc::c_void);
    fn result_is_ok(handle: *const libc::c_void) -> libc::c_int;
    fn result_get_error(handle: *const libc::c_void) -> *const libc::c_char;
    fn create_vulkan_backend() -> *mut libc::c_void;
    fn release_vulkan_backend(be: *mut libc::c_void);
    fn backend_compile(be: *mut libc::c_void, inp: *mut libc::c_void) -> *mut libc::c_char;
    fn binary_program_get_num_kernels(handle: *mut libc::c_void) -> libc::size_t;
    fn binary_program_get_num_kernel_args(
        handle: *mut libc::c_void,
        index: libc::size_t,
    ) -> libc::size_t;
    fn binary_program_get_kernel_args(
        handle: *mut libc::c_void,
        index: libc::size_t,
        array: *mut libc::c_void,
    );
    fn binary_program_get_size(handle: *mut libc::c_void) -> libc::size_t;
    fn binary_program_copy_data(handle: *mut libc::c_void, dst: *mut libc::c_void);
    fn binary_program_get_kernel_name(
        handle: *mut libc::c_void,
        index: libc::size_t,
    ) -> *const libc::c_char;
}

pub struct FrontendResult {
    handle: *mut libc::c_void,
}

impl FrontendResult {
    pub fn new(handle: *mut libc::c_void) -> FrontendResult {
        return FrontendResult { handle };
    }

    pub fn is_ok(&self) -> bool {
        return unsafe { result_is_ok(self.handle) } == 1;
    }

    pub fn get_error(&self) -> String {
        let err_str = unsafe { result_get_error(self.handle) };
        return unsafe { std::ffi::CStr::from_ptr(err_str) }
            .to_str()
            .unwrap()
            .to_owned();
    }

    pub fn get_handle(&self) -> *mut libc::c_void {
        return self.handle;
    }
}

impl Drop for FrontendResult {
    fn drop(&mut self) {
        unsafe { release_result(self.handle) };
    }
}

pub struct ClangFrontend {
    handle: *mut libc::c_void,
}

impl ClangFrontend {
    pub fn new() -> ClangFrontend {
        return ClangFrontend {
            handle: unsafe { create_clang_frontend() },
        };
    }

    // TODO handle options
    pub fn process_source(&self, source: &str) -> FrontendResult {
        let source_string = std::ffi::CString::new(source).unwrap();
        let result =
            unsafe { process_source(self.handle, source_string.as_ptr(), std::ptr::null(), 0) };

        return FrontendResult::new(result);
    }
}

impl Drop for ClangFrontend {
    fn drop(&mut self) {
        unsafe { release_clang_frontend(self.handle) };
    }
}

pub struct BinaryProgram {
    handle: *mut libc::c_void,
}

impl BinaryProgram {
    fn new(handle: *mut libc::c_void) -> BinaryProgram {
        return BinaryProgram { handle };
    }

    pub fn get_kernels(&self) -> Vec<KernelInfo> {
        let mut kernels: Vec<KernelInfo> = vec![];

        let num_kernels = unsafe { binary_program_get_num_kernels(self.handle) };

        kernels.reserve(num_kernels);

        for kidx in 0..num_kernels {
            let num_args = unsafe { binary_program_get_num_kernel_args(self.handle, kidx) };
            let mut args: Vec<KernelArgInfo> = vec![];
            args.resize(
                num_args,
                KernelArgInfo {
                    arg_type: KernelArgType::POD,
                    index: 0,
                    size: 0,
                },
            );
            unsafe {
                binary_program_get_kernel_args(
                    self.handle,
                    kidx,
                    args.as_mut_ptr() as *mut libc::c_void,
                )
            };
            let name = unsafe {
                std::ffi::CStr::from_ptr(binary_program_get_kernel_name(self.handle, kidx))
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
        let bin_size = unsafe { binary_program_get_size(self.handle) };

        let mut data: Vec<u8> = vec![];
        data.resize(bin_size, 0);

        unsafe { binary_program_copy_data(self.handle, data.as_mut_ptr() as *mut libc::c_void) };

        return data;
    }
}

pub trait Backend {
    fn compile(&self, res: &FrontendResult) -> BinaryProgram;
}

pub struct VulkanBackend {
    handle: *mut libc::c_void,
}

impl VulkanBackend {
    pub fn new() -> VulkanBackend {
        let handle = unsafe { create_vulkan_backend() };
        return VulkanBackend { handle };
    }
}

impl Backend for VulkanBackend {
    fn compile(&self, res: &FrontendResult) -> BinaryProgram {
        let compile_result =
            unsafe { backend_compile(self.handle, res.get_handle()) as *mut libc::c_void };
        return BinaryProgram::new(compile_result);
    }
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        unsafe { release_vulkan_backend(self.handle) };
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
