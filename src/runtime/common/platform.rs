use crate::format_error;
use crate::set_info_str;

use once_cell::sync::Lazy;

#[cfg(feature = "vulkan")]
use crate::vulkan;

#[cfg(feature = "metal")]
use crate::metal;

use crate::common::cl_types::cl_int;
use crate::common::cl_types::cl_platform_id;
use crate::common::cl_types::cl_uint;
use crate::common::cl_types::PlatformInfoNames;
use crate::common::cl_types::{CL_INVALID_PLATFORM, CL_INVALID_VALUE, CL_SUCCESS};

use enum_dispatch::enum_dispatch;

#[cfg(feature = "vulkan")]
use crate::vulkan::platform::Platform as VkPlatform;

#[cfg(feature = "metal")]
use crate::metal::Platform as MTLPlatform;

use std::sync::Arc;

use crate::lcl_contract;

use super::device::ClDevice;

#[enum_dispatch(ClPlatform)]
pub trait Platform {
    fn get_platform_name(&self) -> &str;
    fn get_devices(&self) -> &Vec<Arc<ClDevice>>;
}

#[enum_dispatch]
#[repr(C)]
pub enum ClPlatform {
    #[cfg(feature = "vulkan")]
    Vulkan(VkPlatform),
    #[cfg(feature = "metal")]
    Metal(MTLPlatform),
}

static mut GLOBAL_PLATFORMS: Lazy<Vec<Arc<ClPlatform>>> = Lazy::new(|| {
    let mut platforms: Vec<Arc<ClPlatform>> = vec![];

    #[cfg(feature = "vulkan")]
    vulkan::platform::Platform::create_platforms(platforms.as_mut());

    #[cfg(feature = "metal")]
    metal::Platform::create_platforms(platforms.as_mut());

    return platforms;
});

#[no_mangle]
pub extern "C" fn clGetPlatformIDs(
    num_entries: cl_uint,
    platforms_raw: *mut *mut ClPlatform,
    num_platforms_raw: *mut cl_uint,
) -> cl_int {
    let num_platforms = unsafe { num_platforms_raw.as_ref() };
    let platforms = unsafe { platforms_raw.as_ref() };

    lcl_contract!(
        num_entries != 0 || !num_platforms.is_none(),
        "either num_platforms is not NULL or num_entries is not 0",
        CL_INVALID_VALUE
    );

    lcl_contract!(
        !platforms.is_none() || !num_platforms.is_none(),
        "num_platforms and platforms can not be NULL at the same time",
        CL_INVALID_VALUE
    );

    if !platforms.is_none() {
        let platforms_array = unsafe {
            std::slice::from_raw_parts_mut(
                platforms_raw as *mut *mut ClPlatform,
                num_entries as usize,
            )
        };
        for i in 0..num_entries {
            use std::ops::Deref;
            platforms_array[i as usize] = unsafe {
                (GLOBAL_PLATFORMS[i as usize].deref()) as *const ClPlatform as *mut ClPlatform
            };
        }
    }

    if !num_platforms.is_none() {
        unsafe {
            *num_platforms_raw = GLOBAL_PLATFORMS.len() as u32;
        };
    }
    return CL_SUCCESS;
}

#[no_mangle]
pub extern "C" fn clGetPlatformInfo(
    platform: cl_platform_id,
    param_name_num: u32,
    param_value_size: libc::size_t,
    param_value: *mut libc::c_void,
    param_size_ret: *mut libc::size_t,
) -> libc::c_int {
    let platform_safe = unsafe { platform.as_ref() };
    lcl_contract!(
        !platform.is_null(),
        "platfrom can't be NULL",
        CL_INVALID_PLATFORM
    );

    let param_name = PlatformInfoNames::try_from(param_name_num);

    lcl_contract!(
        param_name.is_ok(),
        "invalid param_name value",
        CL_INVALID_VALUE
    );

    match param_name.unwrap() {
        PlatformInfoNames::CL_PLATFORM_NAME => {
            let platform_name = platform_safe.unwrap().get_platform_name();
            // TODO check param value size
            set_info_str!(platform_name, param_value, param_size_ret);
        }
        _ => {}
    }
    return CL_SUCCESS;
}
