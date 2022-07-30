use std::ops::Deref;

use crate::api::cl_types::*;
use crate::interface::{DeviceImpl, DeviceKind, PlatformImpl, PlatformKind};
use crate::{format_error, lcl_contract, set_info_str};

#[no_mangle]
pub unsafe extern "C" fn clGetDeviceIDs(
    platform: cl_platform_id,
    device_type: cl_device_type,
    num_entries: cl_uint,
    devices_raw: *mut cl_device_id,
    num_devices: *mut cl_uint,
) -> cl_int {
    let maybe_platform = PlatformKind::try_from_cl(platform);

    lcl_contract!(
        maybe_platform.is_ok(),
        "platform can not be NULL",
        CL_INVALID_PLATFORM
    );

    lcl_contract!(
        !num_devices.is_null() || !devices_raw.is_null(),
        "ether devices or num_devices must be non-NULL",
        CL_INVALID_VALUE
    );

    let platform_safe = maybe_platform.unwrap();

    // TODO figure out why mut is required here and why I can't simply borrow
    // iterators by constant reference.
    let mut devices = platform_safe
        .deref()
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
            std::slice::from_raw_parts_mut(devices_raw as *mut cl_device_id, num_entries as usize)
        };
        for (i, d) in devices.take(num_entries as usize).enumerate() {
            let device_ptr = d.get_cl_handle();
            devices_array[i] = device_ptr;
        }
    }

    return CL_SUCCESS;
}

#[no_mangle]
pub unsafe extern "C" fn clGetDeviceInfo(
    device: cl_device_id,
    param_name_num: cl_device_info,
    param_value_size: cl_size_t,
    param_value: *mut libc::c_void,
    param_value_size_ret: *mut cl_size_t,
) -> cl_int {
    let device_safe = DeviceKind::try_from_cl(device);

    lcl_contract!(
        device_safe.is_ok(),
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
