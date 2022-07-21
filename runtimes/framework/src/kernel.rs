use crate::cl_types::*;

pub trait Kernel {}

#[no_mangle]
pub extern "C" fn clCreateKernel(
    program: cl_program,
    kernel_name: *const libc::c_char,
    errcode_ret: *mut cl_int,
) -> cl_kernel {
    unimplemented!()
}
