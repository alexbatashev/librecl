use crate::common::cl_types::*;
pub trait Program {}

#[no_mangle]
pub extern "C" fn clCreateProgramWithSource(
    context: *mut cl_context,
    count: cl_uint,
    strings: *const *const libc::c_char,
    lingths: *const libc::size_t,
    errcode_ret: *mut cl_int,
) -> cl_program {
    unimplemented!()
}

#[no_mangle]
pub extern "C" fn clBuildProgram(
    program: cl_program,
    num_devices: cl_uint,
    device_list: *const cl_device_id,
    options: *const libc::c_char,
    callback: cl_build_callback,
    user_data: *mut libc::c_void,
) -> cl_int {
    unimplemented!()
}
