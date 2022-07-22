use crate::common::cl_types::*;

use enum_dispatch::enum_dispatch;

#[cfg(feature = "metal")]
use crate::metal::Device as MTLDevice;
#[cfg(feature = "vulkan")]
use crate::vulkan::device::Device as VkDevice;

#[enum_dispatch]
#[repr(C)]
pub enum ClDevice {
    #[cfg(feature = "vulkan")]
    Vulkan(VkDevice),
    #[cfg(feature = "metal")]
    Metal(MTLDevice),
}

#[enum_dispatch(ClDevice)]
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
