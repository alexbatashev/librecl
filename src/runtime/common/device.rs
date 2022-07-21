use crate::common::cl_types::*;

pub trait Device {}

#[no_mangle]
pub extern "C" fn clGetDeviceInfo(
    device: cl_device_id,
    param_name: cl_device_info,
    param_value_size: libc::size_t,
    param_value: *mut libc::c_void,
    param_value_size_ret: libc::size_t,
) -> cl_uint {
    unimplemented!();
}
