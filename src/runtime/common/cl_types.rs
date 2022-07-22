#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use crate::common::context::Context;
use crate::common::device::ClDevice;
use crate::common::kernel::Kernel;
use crate::common::platform::ClPlatform;
use crate::common::program::Program;
use crate::common::queue::Queue;
use bitflags::bitflags;
use std::convert::TryFrom;

pub type cl_int = libc::c_int;
pub type cl_uint = libc::c_uint;

pub type cl_device_info = libc::c_uint;

pub const CL_DEVICE_TYPE_DEFAULT: libc::c_ulong = 1;
pub const CL_DEVICE_TYPE_CPU: libc::c_ulong = 1 << 1;
pub const CL_DEVICE_TYPE_GPU: libc::c_ulong = 1 << 2;
pub const CL_DEVICE_TYPE_ACCELERATOR: libc::c_ulong = 1 << 3;
pub const CL_DEVICE_TYPE_CUSTOM: libc::c_ulong = 1 << 4;
pub const CL_DEVICE_TYPE_ALL: libc::c_ulong = 0xFFFFFFFF;

bitflags! {
    #[repr(C)]
    pub struct cl_device_type: libc::c_ulong {
        const DefaultDevice = 0;
        const CPU = 1;
        const GPU = 2;
        const ACC = 3;
        const CustomDevice = 4;
    }
}

#[derive(Debug, Clone)]
#[repr(u32)]
pub enum DeviceInfoNames {
    // TODO support all info queries
    CL_DEVICE_NAME = 0x102B,
}

impl TryFrom<cl_uint> for DeviceInfoNames {
    type Error = ();

    fn try_from(v: cl_uint) -> Result<Self, Self::Error> {
        match v {
            x if x == DeviceInfoNames::CL_DEVICE_NAME as cl_uint => {
                Ok(DeviceInfoNames::CL_DEVICE_NAME)
            }
            _ => Err(()),
        }
    }
}

pub type cl_platform_info = cl_uint;

#[derive(Debug, Clone)]
#[repr(u32)]
pub enum PlatformInfoNames {
    CL_PLATFORM_PROFILE = 0x0900,
    CL_PLATFORM_VERSION = 0x0901,
    CL_PLATFORM_NAME = 0x0902,
    CL_PLATFORM_VENDOR = 0x0903,
    CL_PLATFORM_EXTENSIONS = 0x0904,
    CL_PLATFORM_HOST_TIMER_RESOLUTION = 0x0905,
    CL_PLATFORM_NUMERIC_VERSION = 0x0906,
    CL_PLATFORM_EXTENSIONS_WITH_VERSION = 0x0907,
}

pub const CL_PLATFORM_NAME: u32 = 0x0902;

impl TryFrom<cl_uint> for PlatformInfoNames {
    type Error = ();

    fn try_from(v: cl_uint) -> Result<Self, Self::Error> {
        match v {
            x if x == PlatformInfoNames::CL_PLATFORM_PROFILE as cl_uint => {
                Ok(PlatformInfoNames::CL_PLATFORM_NAME)
            }
            x if x == PlatformInfoNames::CL_PLATFORM_VERSION as cl_uint => {
                Ok(PlatformInfoNames::CL_PLATFORM_VERSION)
            }
            x if x == PlatformInfoNames::CL_PLATFORM_NAME as cl_uint => {
                Ok(PlatformInfoNames::CL_PLATFORM_NAME)
            }
            x if x == PlatformInfoNames::CL_PLATFORM_VENDOR as cl_uint => {
                Ok(PlatformInfoNames::CL_PLATFORM_VENDOR)
            }
            x if x == PlatformInfoNames::CL_PLATFORM_EXTENSIONS as cl_uint => {
                Ok(PlatformInfoNames::CL_PLATFORM_EXTENSIONS)
            }
            x if x == PlatformInfoNames::CL_PLATFORM_HOST_TIMER_RESOLUTION as cl_uint => {
                Ok(PlatformInfoNames::CL_PLATFORM_HOST_TIMER_RESOLUTION)
            }
            x if x == PlatformInfoNames::CL_PLATFORM_NUMERIC_VERSION as cl_uint => {
                Ok(PlatformInfoNames::CL_PLATFORM_NUMERIC_VERSION)
            }
            x if x == PlatformInfoNames::CL_PLATFORM_EXTENSIONS_WITH_VERSION as cl_uint => {
                Ok(PlatformInfoNames::CL_PLATFORM_EXTENSIONS_WITH_VERSION)
            }
            _ => Err(()),
        }
    }
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

pub type cl_platform_id = *mut ClPlatform;
pub type cl_device_id = *mut ClDevice;
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
pub const CL_INVALID_DEVICE: cl_int = -33;