use crate::sync::*;
use std::ops::Deref;

use super::cl_types::*;
use crate::{
    format_error,
    interface::{DeviceImpl, DeviceKind, PlatformImpl},
    lcl_contract,
};

#[no_mangle]
pub unsafe extern "C" fn clCreateContext(
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

    let devices_array: Vec<_> =
        unsafe { std::slice::from_raw_parts(devices, num_devices as usize) }
            .iter()
            .map(|&d| DeviceKind::try_from_cl(d))
            .collect();

    lcl_contract!(
        devices_array.iter().all(|d| d.is_ok()),
        "some of devices are NULL",
        CL_INVALID_DEVICE,
        errcode_ret
    );
    lcl_contract!(
        devices_array
            .iter()
            .all(|d| d.as_ref().unwrap().is_available()),
        "some devices are unavailable",
        CL_DEVICE_NOT_AVAILABLE,
        errcode_ret
    );

    let ok_devices: Vec<_> = devices_array
        .iter()
        .map(|d| SharedPtr::downgrade(&d.as_ref().unwrap()))
        .collect();

    // TODO figure out if there's a simpler way to call functions on CL objects
    let owned_platform = devices_array
        .first()
        .unwrap()
        .as_ref()
        .unwrap()
        .deref()
        .get_platform()
        .upgrade();
    let context =
        owned_platform
            .unwrap()
            .deref()
            .create_context(ok_devices.as_slice(), callback, user_data);

    unsafe { *errcode_ret = CL_SUCCESS };

    return _cl_context::wrap(context);
}

#[no_mangle]
pub unsafe extern "C" fn clGetContextInfo(
    context: cl_context,
    param_name: cl_context_info,
    param_value_size: cl_size_t,
    param_value: *mut libc::c_void,
    param_value_size_ret: *mut cl_size_t,
) -> cl_int {
    unimplemented!();
}

#[no_mangle]
pub unsafe extern "C" fn clRetainContext(context: cl_context) -> cl_int {
    lcl_contract!(
        !context.is_null(),
        "context can't be NULL",
        CL_INVALID_CONTEXT
    );

    let context_ref = &mut *context;

    context_ref.retain();

    return CL_SUCCESS;
}

#[no_mangle]
pub unsafe extern "C" fn clReleaseContext(context: cl_context) -> cl_int {
    lcl_contract!(
        !context.is_null(),
        "context can't be NULL",
        CL_INVALID_CONTEXT
    );

    let context_ref = &mut *context;

    if context_ref.release() == 1 {
        // Intentionally ignore value to destroy pointer and its content
        Box::from_raw(context);
    }

    return CL_SUCCESS;
}
