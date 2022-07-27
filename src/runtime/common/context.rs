use crate::common::device::Device;
use crate::common::platform::Platform;
use crate::{common::cl_types::*, format_error, lcl_contract};
use enum_dispatch::enum_dispatch;
use tokio::runtime::Runtime;

#[cfg(feature = "vulkan")]
use crate::vulkan::Context as VkContext;

#[cfg(feature = "metal")]
use crate::metal::Context as MTLContext;

#[enum_dispatch(ClContext)]
pub trait Context {
    fn notify_error(&self, message: String);
    fn has_device(&self, device: cl_device_id) -> bool;
    fn create_program_with_source(&self, context: cl_context, source: String) -> cl_program;
    fn get_threading_runtime(&self) -> &Runtime;
    fn get_associated_devices(&self) -> &[cl_device_id];
    fn create_buffer(&mut self, context: cl_context, size: usize, flags: cl_mem_flags) -> cl_mem;
}

#[enum_dispatch]
#[repr(C)]
pub enum ClContext {
    #[cfg(feature = "vulkan")]
    Vulkan(VkContext),
    #[cfg(feature = "metal")]
    Metal(MTLContext),
}

#[no_mangle]
pub extern "C" fn clCreateContext(
    _properties: *const cl_context_properties,
    num_devices: cl_uint,
    devices: *const cl_device_id,
    callback: cl_context_callback,
    user_data: *mut libc::c_void,
    errcode_ret: *mut cl_int,
) -> cl_context {
    // TODO support properties

    lcl_contract!(
        num_devices > 0,
        "context requires at leas one device",
        CL_INVALID_VALUE,
        errcode_ret
    );

    lcl_contract!(
        !devices.is_null(),
        "devices can't be NULL",
        CL_INVALID_VALUE,
        errcode_ret
    );

    let devices_array = unsafe { std::slice::from_raw_parts(devices, num_devices as usize) };

    lcl_contract!(
        devices_array.iter().all(|&d| !d.is_null()),
        "some of devices are NULL",
        CL_INVALID_DEVICE,
        errcode_ret
    );
    lcl_contract!(
        devices_array
            .iter()
            .all(|&d| unsafe { d.as_ref().unwrap() }.is_available()),
        "some devices are unavailable",
        CL_DEVICE_NOT_AVAILABLE,
        errcode_ret
    );

    // TODO figure out if there's a simpler way to call functions on CL objects
    let context = unsafe {
        devices_array
            .first()
            .unwrap()
            .as_ref()
            .unwrap()
            .get_platform()
            .as_ref()
            .unwrap()
    }
    .create_context(devices_array, callback, user_data);

    unsafe { *errcode_ret = CL_SUCCESS };

    return context;
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
