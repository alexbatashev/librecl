use backtrace::Backtrace;
use libc::c_int;
use libc::c_uint;
use libc::c_void;
use libc::size_t;
use stdext::function_name;

use crate::cl_platform_id;
use crate::cl_platform_info;
use crate::lcl_contract;
use crate::CL_INVALID_PLATFORM;
use crate::CL_SUCCESS;

pub trait Platform {
    fn get_platform_name(&self) -> &str;
}

#[no_mangle]
pub extern "C" fn clGetPlatformInfo(
    platform: cl_platform_id,
    param_name: cl_platform_info,
    param_value_size: libc::size_t,
    param_value: *mut libc::c_void,
    param_size_ret: *mut libc::size_t,
) -> libc::c_int {
    lcl_contract!(
        platform != std::ptr::null_mut(),
        "platfrom can't be NULL",
        CL_INVALID_PLATFORM
    );

    match param_name {
        cl_platform_info::CL_PLATFORM_NAME => {
            let platform_name = unsafe { platform.as_ref().unwrap().get_platform_name() };
        }
        _ => {}
    }
    return CL_SUCCESS;
}
