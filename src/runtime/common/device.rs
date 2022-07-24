use crate::{common::cl_types::*, format_error, lcl_contract, set_info_str, Platform};
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
pub trait Device {
    fn get_device_type(&self) -> cl_device_type;
    fn get_device_name(&self) -> String;
    fn is_available(&self) -> bool;
    fn get_platform(&self) -> cl_platform_id;
    fn create_queue(&self, context: cl_context, device: cl_device_id) -> cl_command_queue;
}

#[no_mangle]
pub extern "C" fn clGetDeviceIDs(
    platform: cl_platform_id,
    device_type: cl_device_type,
    num_entries: cl_uint,
    devices_raw: *mut cl_device_id,
    num_devices: *mut cl_uint,
) -> cl_int {
    let platform_safe = unsafe { platform.as_ref() };

    lcl_contract!(
        platform_safe.is_some(),
        "platform can not be NULL",
        CL_INVALID_PLATFORM
    );

    lcl_contract!(
        !num_devices.is_null() || !devices_raw.is_null(),
        "ether devices or num_devices must be non-NULL",
        CL_INVALID_VALUE
    );

    // TODO figure out why mut is required here and why I can't simply borrow
    // iterators by constant reference.
    let mut devices = platform_safe
        .unwrap()
        .get_devices()
        .into_iter()
        .filter(|&d| device_type.contains(d.get_device_type()))
        .into_iter();

    if !num_devices.is_null() {
        // TODO find safe way to do this
        unsafe {
            *num_devices = devices.by_ref().count() as u32;
        }
    }

    if !devices_raw.is_null() {
        let devices_array = unsafe {
            std::slice::from_raw_parts_mut(devices_raw as *mut *mut ClDevice, num_entries as usize)
        };
        for (i, d) in devices.take(num_entries as usize).enumerate() {
            let device_ptr = std::rc::Rc::as_ptr(&d) as *mut ClDevice;
            devices_array[i] = device_ptr;
        }
    }

    return CL_SUCCESS;
}

#[no_mangle]
pub extern "C" fn clGetDeviceInfo(
    device: cl_device_id,
    param_name_num: cl_device_info,
    param_value_size: libc::size_t,
    param_value: *mut libc::c_void,
    param_value_size_ret: *mut libc::size_t,
) -> cl_int {
    let device_safe = unsafe { device.as_ref() };

    lcl_contract!(
        device_safe.is_some(),
        "device can't be NULL",
        CL_INVALID_DEVICE
    );

    let param_name = DeviceInfoNames::try_from(param_name_num);

    lcl_contract!(
        param_name.is_ok(),
        "invalid param_name value",
        CL_INVALID_VALUE
    );

    match param_name.unwrap() {
        DeviceInfoNames::CL_DEVICE_NAME => {
            let device_name = device_safe.unwrap().get_device_name();
            set_info_str!(device_name, param_value, param_value_size_ret);
        }
    }

    return CL_SUCCESS;
}
