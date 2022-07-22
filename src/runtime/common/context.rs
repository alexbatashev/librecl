use crate::common::cl_types::*;
use enum_dispatch::enum_dispatch;

#[cfg(feature = "vulkan")]
use crate::vulkan::Context as VkContext;

#[cfg(feature = "metal")]
use crate::metal::Context as MTLContext;

#[enum_dispatch(ClContext)]
pub trait Context {}

#[enum_dispatch]
#[repr(C)]
pub enum ClContext{
    #[cfg(feature = "vulkan")]
    Vulkan(VkContext),
    #[cfg(feature = "metal")]
    Metal(MTLContext),
}

#[no_mangle]
pub extern "C" fn clCreateContext(
    properties: *const cl_context_properties,
    num_devices: cl_uint,
    devices: *const cl_device_id,
    callback: cl_context_callback,
    user_data: *mut libc::c_void,
    errcode_ret: *mut cl_int,
) -> cl_context {
    unimplemented!();
}

#[no_mangle]
pub extern "C" fn clGetContextInfo(
    context: cl_context,
    param_name: cl_context_info,
    param_value_size: libc::size_t,
    param_value_size_ret: *mut libc::size_t,
) -> cl_int {
    unimplemented!();
}
