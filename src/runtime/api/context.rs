use super::cl_types::*;
use crate::api::error_handling::ClError;
use crate::{
    interface::{DeviceImpl, DeviceKind, PlatformImpl},
    lcl_contract,
};
use crate::{success, sync::*};
use ocl_type_wrapper::cl_api;
use std::ops::Deref;

#[cl_api]
fn clCreateContext(
    _properties: *const cl_context_properties,
    num_devices: cl_uint,
    devices: *const cl_device_id,
    callback: cl_context_callback,
    user_data: *mut libc::c_void,
) -> Result<cl_context, ClError> {
    // TODO support properties

    lcl_contract!(
        num_devices > 0,
        ClError::InvalidValue,
        "context requires at leas one device"
    );

    lcl_contract!(
        !devices.is_null(),
        ClError::InvalidValue,
        "devices can't be NULL"
    );

    let devices_array: Vec<_> =
        unsafe { std::slice::from_raw_parts(devices, num_devices as usize) }
            .iter()
            .map(|&d| DeviceKind::try_from_cl(d))
            .collect();

    lcl_contract!(
        devices_array.iter().all(|d| d.is_ok()),
        ClError::InvalidDevice,
        "some of devices are NULL"
    );
    lcl_contract!(
        devices_array
            .iter()
            .all(|d| d.as_ref().unwrap().is_available()),
        ClError::DeviceNotAvailable,
        "some devices are unavailable"
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

    return Ok(_cl_context::wrap(context));
}

#[cl_api]
fn clGetContextInfo(
    _context: cl_context,
    _param_name: cl_context_info,
    _param_value_size: cl_size_t,
    _param_value: *mut libc::c_void,
    _param_value_size_ret: *mut cl_size_t,
) -> Result<(), ClError> {
    unimplemented!();
}

#[cl_api]
fn clRetainContext(context: cl_context) -> Result<(), ClError> {
    lcl_contract!(
        !context.is_null(),
        ClError::InvalidContext,
        "context can't be NULL"
    );

    let context_ref = unsafe { &mut *context };

    context_ref.retain();

    return success!();
}

#[cl_api]
fn clReleaseContext(context: cl_context) -> Result<(), ClError> {
    lcl_contract!(
        !context.is_null(),
        ClError::InvalidContext,
        "context can't be NULL"
    );

    let context_ref = unsafe { &mut *context };

    if context_ref.release() == 1 {
        // Intentionally ignore value to destroy pointer and its content
        unsafe { Box::from_raw(context) };
    }

    return success!();
}
