#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use crate::interface::*;

use crate::sync;
use bitflags::bitflags;
use ocl_type_wrapper::cl_object;
use std::convert::TryFrom;

use super::cl_icd::_cl_icd_dispatch;

pub type cl_size_t = u64;
pub type cl_int = libc::c_int;
pub type cl_uint = libc::c_uint;
pub type cl_bool = libc::c_uint;
pub type cl_ulong = libc::c_ulong;
pub type cl_version = cl_uint;

pub const CL_NAME_VERSION_MAX_NAME_SIZE: usize = 64;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct _cl_name_version {
    pub version: cl_version,
    pub name: [libc::c_uchar; CL_NAME_VERSION_MAX_NAME_SIZE],
}
pub type cl_name_version = _cl_name_version;

pub use super::cl_icd::cl_event;

pub trait IntoCl<T> {
    type Error;

    fn try_into_safe(&self) -> Result<T, Self::Error>;
}

pub trait FromCl<T>
where
    Self: Send + Sync,
{
    type Error;

    fn try_from_cl(value: T) -> Result<crate::sync::SharedPtr<Self>, Self::Error>;
}

#[cl_object(PlatformKind)]
pub struct _cl_platform_id;
pub type cl_platform_id = *mut _cl_platform_id;

#[cl_object(DeviceKind)]
pub struct _cl_device_id;
pub type cl_device_id = *mut _cl_device_id;

#[cl_object(ContextKind)]
pub struct _cl_context;
pub type cl_context = *mut _cl_context;

#[cl_object(QueueKind)]
pub struct _cl_command_queue;
pub type cl_command_queue = *mut _cl_command_queue;

#[cl_object(ProgramKind)]
pub struct _cl_program;
pub type cl_program = *mut _cl_program;

#[cl_object(KernelKind)]
pub struct _cl_kernel;
pub type cl_kernel = *mut _cl_kernel;

#[cl_object(MemKind)]
pub struct _cl_mem;
pub type cl_mem = *mut _cl_mem;

pub trait ClObjectImpl<T> {
    fn get_cl_handle(&self) -> T;
    fn set_cl_handle(&mut self, handle: T);
}

pub type cl_device_info = libc::c_uint;
pub type cl_context_info = libc::c_uint;

pub const CL_DEVICE_TYPE_DEFAULT: libc::c_ulong = 1;
pub const CL_DEVICE_TYPE_CPU: libc::c_ulong = 1 << 1;
pub const CL_DEVICE_TYPE_GPU: libc::c_ulong = 1 << 2;
pub const CL_DEVICE_TYPE_ACCELERATOR: libc::c_ulong = 1 << 3;
pub const CL_DEVICE_TYPE_CUSTOM: libc::c_ulong = 1 << 4;
pub const CL_DEVICE_TYPE_ALL: libc::c_ulong = 0xFFFFFFFF;

bitflags! {
    #[repr(C)]
    pub struct cl_device_type: libc::c_ulong {
        const DefaultDevice = 0b00000001;
        const CPU = 0b00000010;
        const GPU = 0b00000100;
        const ACC = 0b00001000;
        const CustomDevice = 0b00010000;
    }
}

include!("info/cl_device_info.rs");
include!("info/cl_context_info.rs");

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
    CL_PLATFORM_ICD_SUFFIX_KHR = 0x0920,
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
            x if x == PlatformInfoNames::CL_PLATFORM_ICD_SUFFIX_KHR as cl_uint => {
                Ok(PlatformInfoNames::CL_PLATFORM_ICD_SUFFIX_KHR)
            }
            _ => Err(()),
        }
    }
}

#[derive(Clone, Copy)]
#[repr(usize)]
pub enum cl_context_properties {
    CL_CONTEXT_PLATFORM = 0x1084,
    CL_CONTEXT_INTEROP_USER_SYNC = 0x1085,
}
pub type cl_queue_properties = libc::c_uint;

pub type cl_context_callback = Option<
    unsafe extern "C" fn(
        errinfo: *const libc::c_char,
        private_info: *const libc::c_void,
        cb: cl_size_t,
        user_data: *mut libc::c_void,
    ),
>;

pub type cl_build_callback =
    Option<unsafe extern "C" fn(program: cl_program, user_data: *mut libc::c_void)>;

bitflags! {
    #[repr(C)]
    pub struct cl_mem_flags: libc::c_ulong {
        const ReadWrite = 0;
        const WriteOnly = 1;
        const ReadOnly = 2;
        const UseHostPtr = 3;
        const AllocHostPtr = 4;
        const CopyHostPtr = 5;
        const Reserved1 = 6;
        const HostWriteOnly = 7;
        const HostReadOnly = 8;
        const HostNoAccess = 9;
        const SVMFineGrainBuffer = 10;
        const SVMAtomics = 11;
        const KernelReadAndWrite = 12;
    }
}

/// This is an exception to generic error handling mechanism, because success
/// error code had no meaning.
pub const CL_SUCCESS: cl_int = 0;

const CL_VERSION_MAJOR_BITS: u32 = 10;
const CL_VERSION_MINOR_BITS: u32 = 10;
const CL_VERSION_PATCH_BITS: u32 = 12;

pub fn make_version(major: u32, minor: u32, patch: u32) -> u32 {
    let major_mask = (1 << CL_VERSION_MAJOR_BITS) - 1 as u32;
    let minor_mask = (1 << CL_VERSION_MINOR_BITS) - 1 as u32;
    let patch_mask = (1 << CL_VERSION_PATCH_BITS) - 1 as u32;

    return ((major & major_mask) << (CL_VERSION_MINOR_BITS + CL_VERSION_PATCH_BITS))
        | ((minor & minor_mask) << CL_VERSION_PATCH_BITS)
        | (patch & patch_mask);
}
