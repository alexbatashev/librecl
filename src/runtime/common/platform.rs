use crate::format_error;
use crate::set_info_str;
// use crate::CL_INVALID_VALUE;
use backtrace::Backtrace;
use libc::c_int;
use libc::c_uint;
use libc::c_void;
use libc::size_t;
use stdext::function_name;

// use crate::cl_platform_info;
use crate::lcl_contract;
// use crate::CL_INVALID_PLATFORM;
// use crate::CL_SUCCESS;
use crate::common::cl_types::cl_int;
use crate::common::cl_types::cl_platform_id;
use crate::common::cl_types::cl_uint;
use crate::common::cl_types::PlatformInfoNames;
use crate::common::cl_types::{CL_INVALID_PLATFORM, CL_INVALID_VALUE, CL_SUCCESS};

pub trait Platform {
    fn get_platform_name(&self) -> &str;
}

#[repr(C)]
pub struct ClPlatform {
    pimpl: Box<*mut dyn Platform>,
}

static mut GLOBAL_PLATFORMS: Vec<Box<ClPlatform>> = vec![];

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
            platforms_array[i as usize] = unsafe { GLOBAL_PLATFORMS[i as usize].as_mut() };
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

    println!("Requested param name is {}", param_name_num);
    // TODO correct error code
    lcl_contract!(
        param_name.is_ok(),
        "invalid param_name value ",
        CL_INVALID_VALUE
    );

    println!("Requested param is {:?}", param_name);

    match param_name.unwrap() {
        PlatformInfoNames::CL_PLATFORM_NAME => {
            let platform_name =
                unsafe { platform_safe.unwrap().pimpl.as_ref().get_platform_name() };
            // TODO check param value size
            set_info_str!(platform_name, param_value, param_size_ret);
        }
        _ => {}
    }
    return CL_SUCCESS;
}
