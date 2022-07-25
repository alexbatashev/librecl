#[link(name = "lcl_compiler")]
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
}

pub struct FrontendResult {
    handle: *mut libc::c_void,
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
}

impl Drop for ClangFrontend {
    fn drop(&mut self) {
        unsafe { release_clang_frontend(self.handle) };
    }
}
