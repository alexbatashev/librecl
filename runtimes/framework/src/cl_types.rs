#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use crate::context::Context;
use crate::device::Device;
use crate::kernel::Kernel;
use crate::platform::Platform;
use crate::program::Program;
use crate::queue::Queue;

pub type cl_int = libc::c_int;
pub type cl_uint = libc::c_uint;

pub type cl_device_type = libc::c_uint;
pub type cl_device_info = libc::c_uint;

#[repr(u32)]
pub enum cl_platform_info {
    CL_PLATFORM_PROFILE = 0x0900,
    CL_PLATFORM_VERSION = 0x0901,
    CL_PLATFORM_NAME = 0x0902,
    CL_PLATFORM_VENDOR = 0x0903,
    CL_PLATFORM_EXTENSIONS = 0x0904,
    CL_PLATFORM_HOST_TIMER_RESOLUTION = 0x0905,
    CL_PLATFORM_NUMERIC_VERSION = 0x0906,
    CL_PLATFORM_EXTENSIONS_WITH_VERSION = 0x0907,
}

#[repr(u32)]
pub enum cl_context_info {
    CL_CONTEXT_REFERENCE_COUNT = 0x1080,
    CL_CONTEXT_DEVICES = 0x1081,
    CL_CONTEXT_PROPERTIES = 0x1082,
    CL_CONTEXT_NUM_DEVICES = 0x1083,
}

#[repr(usize)]
pub enum cl_context_properties {
    CL_CONTEXT_PLATFORM = 0x1084,
    CL_CONTEXT_INTEROP_USER_SYNC = 0x1085,
}
pub type cl_queue_properties = libc::c_uint;

pub type cl_platform_id = *mut dyn Platform;
pub type cl_device_id = *mut dyn Device;
pub type cl_context = *mut dyn Context;
pub type cl_command_queue = *mut dyn Queue;
pub type cl_program = *mut dyn Program;
pub type cl_kernel = *mut dyn Kernel;

pub type cl_context_callback = extern "C" fn(
    errinfo: *const libc::c_char,
    private_info: *const libc::c_void,
    cb: libc::size_t,
    user_data: *mut libc::c_void,
);

pub type cl_build_callback = extern "C" fn(program: cl_program, user_data: *mut libc::c_void);

pub const CL_SUCCESS: cl_int = 0;
pub const CL_INVALID_VALUE: cl_int = -30;
pub const CL_INVALID_PLATFORM: cl_int = -32;
