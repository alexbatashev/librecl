use super::error_handling::ClError;
use crate::api::cl_types::*;
use crate::interface::{DeviceImpl, DeviceKind, PlatformImpl, PlatformKind, DeviceLimitsInterface};
use crate::{lcl_contract, set_info_array, set_info_int, set_info_str, success};
use ocl_type_wrapper::cl_api;
use std::ops::Deref;

#[cl_api]
fn clGetDeviceIDs(
    platform: cl_platform_id,
    device_type: cl_device_type,
    num_entries: cl_uint,
    devices_raw: *mut cl_device_id,
    num_devices: *mut cl_uint,
) -> Result<(), ClError> {
    let platform_safe = PlatformKind::try_from_cl(platform).map_err(|reason| {
        ClError::InvalidPlatform(std::format!("not a valid platform: {}", reason).into())
    })?;

    lcl_contract!(
        !num_devices.is_null() || !devices_raw.is_null(),
        ClError::InvalidPlatform,
        "ether devices or num_devices must be non-NULL"
    );

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
        unsafe { *num_devices = devices.by_ref().count() as u32 };
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

    return success!();
}

#[cl_api]
fn clGetDeviceInfo(
    device: cl_device_id,
    param_name_num: cl_device_info,
    _param_value_size: cl_size_t,
    param_value: *mut libc::c_void,
    param_value_size_ret: *mut cl_size_t,
) -> Result<(), ClError> {
    let device_safe = DeviceKind::try_from_cl(device)
        .map_err(|reason| ClError::InvalidDevice(format!("invalid device: {}", reason).into()))?;

    let param_name = DeviceInfoNames::try_from(param_name_num).map_err(|_| {
        ClError::InvalidValue(format!("unknown param_name: {}", param_name_num).into())
    })?;

    match param_name {
        DeviceInfoNames::Name => {
            let info = device_safe.get_device_name();
            set_info_str!(info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::Vendor => {
            let info = device_safe.get_vendor_name();
            set_info_str!(info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::VendorId => {
            let info = device_safe.get_vendor_id();
            set_info_int!(cl_uint, info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::MaxComputeUnits => {
            let info = device_safe.max_compute_units();
            set_info_int!(cl_uint, info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::MaxWorkItemDimensions => {
            let info = device_safe.max_work_item_dimensions();
            set_info_int!(cl_uint, info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::MaxWorkItemSizes => {
            let info = device_safe.max_work_item_sizes();
            set_info_array!(cl_size_t, info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::MaxWorkGroupSize => {
            let info = device_safe.max_work_group_size();
            set_info_int!(cl_size_t, info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::PreferredVectorWidthChar => {
            let info = device_safe.preferred_vector_width_char();
            set_info_int!(cl_uint, info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::PreferredVectorWidthShort => {
            let info = device_safe.preferred_vector_width_short();
            set_info_int!(cl_uint, info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::PreferredVectorWidthInt => {
            let info = device_safe.preferred_vector_width_int();
            set_info_int!(cl_uint, info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::PreferredVectorWidthLong => {
            let info = device_safe.preferred_vector_width_long();
            set_info_int!(cl_uint, info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::PreferredVectorWidthFloat => {
            let info = device_safe.preferred_vector_width_float();
            set_info_int!(cl_uint, info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::PreferredVectorWidthDouble => {
            let info = device_safe.preferred_vector_width_double();
            set_info_int!(cl_uint, info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::PreferredVectorWidthHalf => {
            let info = device_safe.preferred_vector_width_half();
            set_info_int!(cl_uint, info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::NativeVectorWidthChar => {
            let info = device_safe.native_vector_width_char();
            set_info_int!(cl_uint, info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::NativeVectorWidthShort => {
            let info = device_safe.native_vector_width_short();
            set_info_int!(cl_uint, info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::NativeVectorWidthInt => {
            let info = device_safe.native_vector_width_int();
            set_info_int!(cl_uint, info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::NativeVectorWidthLong => {
            let info = device_safe.native_vector_width_long();
            set_info_int!(cl_uint, info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::NativeVectorWidthFloat => {
            let info = device_safe.native_vector_width_float();
            set_info_int!(cl_uint, info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::NativeVectorWidthDouble => {
            let info = device_safe.native_vector_width_double();
            set_info_int!(cl_uint, info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::NativeVectorWidthHalf => {
            let info = device_safe.native_vector_width_half();
            set_info_int!(cl_uint, info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::Available => {
            let info = if device_safe.is_available() {
                1 as cl_bool
            } else {
                0 as cl_bool
            };
            set_info_int!(cl_bool, info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::CompilerAvailable => {
            let info = if device_safe.is_compiler_available() {
                1 as cl_bool
            } else {
                0 as cl_bool
            };
            set_info_int!(cl_bool, info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::LinkerAvailable => {
            // With current architecture both compiler and linker are
            // the same component.
            let info = if device_safe.is_compiler_available() {
                1 as cl_bool
            } else {
                0 as cl_bool
            };
            set_info_int!(cl_bool, info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::Platform => {
            let platform = device_safe.get_platform().upgrade().unwrap();
            let info = platform.get_cl_handle();
            set_info_int!(cl_platform_id, info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::ClDriverVersion => {
            // TODO get version dynamically
            let info =
                "LibreCL 0.1.0 over ".to_owned() + device_safe.get_native_driver_version().as_str();
            set_info_str!(info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::Profile => {
            let info = device_safe.get_device_profile();
            set_info_str!(info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::Version => {
            let info = String::from("OpenCL 3.0 ") + device_safe.get_device_version_info().as_str();
            set_info_str!(info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::NumericVersion => {
            let info = make_version(3, 0, 0);
            set_info_int!(cl_version, info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::OpenclCVersion => {
            // TODO need support for OpenCL C 2.1
            let info = String::from("OpenCL C 1.2");
            set_info_str!(info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::Extensions => {
            let extensions_vec: Vec<_> = device_safe.get_extension_names().to_vec();
            let info = extensions_vec.join(" ");
            set_info_str!(info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::ExtensionsWithVersion => {
            let names = device_safe.get_extension_names();
            let versions = device_safe.get_extension_versions();

            let info: Vec<_> = names
                .iter()
                .zip(versions.iter())
                .map(|(&n, &v)| cl_name_version {
                    version: v,
                    name: n.as_bytes().try_into().expect("failed to convert to array"),
                })
                .collect();
            set_info_array!(cl_name_version, info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::MaxMemAllocSize => {
            let info = device_safe.max_mem_alloc_size();
            set_info_int!(cl_ulong, info, param_value, param_value_size_ret)
        }
        DeviceInfoNames::PreferredWorkGroupSizeMultiple => {
            let info = device_safe.preferred_work_group_size_multiple();
            set_info_int!(cl_size_t, info, param_value, param_value_size_ret)
        }
        _ => Err(ClError::InvalidValue(
            format!("{} is not supported yet", param_name.as_cl_str()).into(),
        )),
    }
}

#[cl_api]
fn clRetainDevice(_device: cl_device_id) -> Result<(), ClError> {
    // since all devices are root devices for now, simply return
    return success!();
}

#[cl_api]
fn clReleaseDevice(_device: cl_device_id) -> Result<(), ClError> {
    return success!();
}
