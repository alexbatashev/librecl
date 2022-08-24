use super::cl_types::*;
use super::error_handling::map_invalid_context;
use crate::api::error_handling::ClError;
use crate::interface::{ContextImpl, ContextKind};
use crate::{
    interface::{DeviceImpl, DeviceKind, PlatformImpl},
    lcl_contract,
};
use crate::{set_info_array, set_info_int, success, sync::*};
use lcl_derive::cl_api;
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
    context: cl_context,
    param_name_num: cl_context_info,
    _param_value_size: cl_size_t,
    param_value: *mut libc::c_void,
    param_value_size_ret: *mut cl_size_t,
) -> Result<(), ClError> {
    let context_safe = ContextKind::try_from_cl(context).map_err(map_invalid_context)?;

    let param_name = ContextInfoNames::try_from(param_name_num).map_err(|_| {
        ClError::InvalidValue(format!("unknown param_name: {}", param_name_num).into())
    })?;

    match param_name {
        ContextInfoNames::ReferenceCount => {
            let ctx = unsafe { context.as_ref() }.unwrap();
            let ref_count = ctx.reference_count() as cl_uint;
            set_info_int!(cl_uint, ref_count, param_value, param_value_size_ret)
        }
        ContextInfoNames::NumDevices => {
            let num_devices = context_safe.get_associated_devices().len() as cl_uint;
            set_info_int!(cl_uint, num_devices, param_value, param_value_size_ret)
        }
        ContextInfoNames::Devices => {
            let devices = context_safe.get_associated_devices();
            let mut cl_devices = vec![];
            for d in devices {
                let owned_device = d.upgrade().ok_or(map_invalid_context(
                    "failed to get owning reference to device".to_owned(),
                ))?;
                cl_devices.push(owned_device.get_cl_handle());
            }
            set_info_array!(cl_device_id, cl_devices, param_value, param_value_size_ret)
        }
        ContextInfoNames::Properties => {
            let props: [cl_context_properties; 0] = [];
            set_info_array!(
                cl_context_properties,
                props,
                param_value,
                param_value_size_ret
            )
        }
    }
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
