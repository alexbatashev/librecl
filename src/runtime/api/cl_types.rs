#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use crate::interface::*;

use bitflags::bitflags;
use core::fmt;
use ocl_type_wrapper::cl_object;
use std::convert::TryFrom;

use super::cl_icd::_cl_icd_dispatch;

impl _cl_icd_dispatch {
    pub fn new() -> *mut _cl_icd_dispatch {
        Box::leak(Box::new(_cl_icd_dispatch {
            clGetPlatformIDs: None,
            clGetPlatformInfo: None,
            clGetDeviceIDs: None,
            clGetDeviceInfo: None,
            clCreateContext: None,
            clCreateContextFromType: None,
            clRetainContext: None,
            clReleaseContext: None,
            clGetContextInfo: None,
            clCreateCommandQueue: None,
            clRetainCommandQueue: None,
            clReleaseCommandQueue: None,
            clGetCommandQueueInfo: None,
            clSetCommandQueueProperty: None,
            clCreateBuffer: None,
            clCreateImage2D: None,
            clCreateImage3D: None,
            clRetainMemObject: None,
            clReleaseMemObject: None,
            clGetSupportedImageFormats: None,
            clGetMemObjectInfo: None,
            clGetImageInfo: None,
            clCreateSampler: None,
            clRetainSampler: None,
            clReleaseSampler: None,
            clGetSamplerInfo: None,
            clCreateProgramWithSource: None,
            clCreateProgramWithBinary: None,
            clRetainProgram: None,
            clReleaseProgram: None,
            clBuildProgram: None,
            clUnloadCompiler: None,
            clGetProgramInfo: None,
            clGetProgramBuildInfo: None,
            clCreateKernel: None,
            clCreateKernelsInProgram: None,
            clRetainKernel: None,
            clReleaseKernel: None,
            clSetKernelArg: None,
            clGetKernelInfo: None,
            clGetKernelWorkGroupInfo: None,
            clWaitForEvents: None,
            clGetEventInfo: None,
            clRetainEvent: None,
            clReleaseEvent: None,
            clGetEventProfilingInfo: None,
            clFlush: None,
            clFinish: None,
            clEnqueueReadBuffer: None,
            clEnqueueWriteBuffer: None,
            clEnqueueCopyBuffer: None,
            clEnqueueReadImage: None,
            clEnqueueWriteImage: None,
            clEnqueueCopyImage: None,
            clEnqueueCopyImageToBuffer: None,
            clEnqueueCopyBufferToImage: None,
            clEnqueueMapBuffer: None,
            clEnqueueMapImage: None,
            clEnqueueUnmapMemObject: None,
            clEnqueueNDRangeKernel: None,
            clEnqueueTask: None,
            clEnqueueNativeKernel: None,
            clEnqueueMarker: None,
            clEnqueueWaitForEvents: None,
            clEnqueueBarrier: None,
            clGetExtensionFunctionAddress: None,
            clCreateFromGLBuffer: None,
            clCreateFromGLTexture2D: None,
            clCreateFromGLTexture3D: None,
            clCreateFromGLRenderbuffer: None,
            clGetGLObjectInfo: None,
            clGetGLTextureInfo: None,
            clEnqueueAcquireGLObjects: None,
            clEnqueueReleaseGLObjects: None,
            clGetGLContextInfoKHR: None,
            clGetDeviceIDsFromD3D10KHR: std::ptr::null_mut(),
            clCreateFromD3D10BufferKHR: std::ptr::null_mut(),
            clCreateFromD3D10Texture2DKHR: std::ptr::null_mut(),
            clCreateFromD3D10Texture3DKHR: std::ptr::null_mut(),
            clEnqueueAcquireD3D10ObjectsKHR: std::ptr::null_mut(),
            clEnqueueReleaseD3D10ObjectsKHR: std::ptr::null_mut(),
            clSetEventCallback: None,
            clCreateSubBuffer: None,
            clSetMemObjectDestructorCallback: None,
            clCreateUserEvent: None,
            clSetUserEventStatus: None,
            clEnqueueReadBufferRect: None,
            clEnqueueWriteBufferRect: None,
            clEnqueueCopyBufferRect: None,
            clCreateSubDevicesEXT: None,
            clRetainDeviceEXT: None,
            clReleaseDeviceEXT: None,
            clCreateEventFromGLsyncKHR: None,
            clCreateSubDevices: None,
            clRetainDevice: None,
            clReleaseDevice: None,
            clCreateImage: None,
            clCreateProgramWithBuiltInKernels: None,
            clCompileProgram: None,
            clLinkProgram: None,
            clUnloadPlatformCompiler: None,
            clGetKernelArgInfo: None,
            clEnqueueFillBuffer: None,
            clEnqueueFillImage: None,
            clEnqueueMigrateMemObjects: None,
            clEnqueueMarkerWithWaitList: None,
            clEnqueueBarrierWithWaitList: None,
            clGetExtensionFunctionAddressForPlatform: None,
            clCreateFromGLTexture: None,
            clGetDeviceIDsFromD3D11KHR: std::ptr::null_mut(),
            clCreateFromD3D11BufferKHR: std::ptr::null_mut(),
            clCreateFromD3D11Texture2DKHR: std::ptr::null_mut(),
            clCreateFromD3D11Texture3DKHR: std::ptr::null_mut(),
            clCreateFromDX9MediaSurfaceKHR: std::ptr::null_mut(),
            clEnqueueAcquireD3D11ObjectsKHR: std::ptr::null_mut(),
            clEnqueueReleaseD3D11ObjectsKHR: std::ptr::null_mut(),
            clGetDeviceIDsFromDX9MediaAdapterKHR: std::ptr::null_mut(),
            clEnqueueAcquireDX9MediaSurfacesKHR: std::ptr::null_mut(),
            clEnqueueReleaseDX9MediaSurfacesKHR: std::ptr::null_mut(),
            clCreateFromEGLImageKHR: None,
            clEnqueueAcquireEGLObjectsKHR: None,
            clEnqueueReleaseEGLObjectsKHR: None,
            clCreateEventFromEGLSyncKHR: None,
            clCreateCommandQueueWithProperties: None,
            clCreatePipe: None,
            clGetPipeInfo: None,
            clSVMAlloc: None,
            clSVMFree: None,
            clEnqueueSVMFree: None,
            clEnqueueSVMMemcpy: None,
            clEnqueueSVMMemFill: None,
            clEnqueueSVMMap: None,
            clEnqueueSVMUnmap: None,
            clCreateSamplerWithProperties: None,
            clSetKernelArgSVMPointer: None,
            clSetKernelExecInfo: None,
            clGetKernelSubGroupInfoKHR: None,
            clCloneKernel: None,
            clCreateProgramWithIL: None,
            clEnqueueSVMMigrateMem: None,
            clGetDeviceAndHostTimer: None,
            clGetHostTimer: None,
            clGetKernelSubGroupInfo: None,
            clSetDefaultDeviceCommandQueue: None,
            clSetProgramReleaseCallback: None,
            clSetProgramSpecializationConstant: None,
            clCreateBufferWithProperties: None,
            clCreateImageWithProperties: None,
            clSetContextDestructorCallback: None,
        }))
    }
}

pub type cl_int = libc::c_int;
pub type cl_uint = libc::c_uint;
pub type cl_bool = libc::c_uint;

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

pub type cl_context_callback = Option<
    extern "C" fn(
        errinfo: *const libc::c_char,
        private_info: *const libc::c_void,
        cb: libc::size_t,
        user_data: *mut libc::c_void,
    ),
>;

pub type cl_build_callback =
    Option<extern "C" fn(program: cl_program, user_data: *mut libc::c_void)>;

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

pub const CL_DEVICE_NOT_AVAILABLE: cl_int = -2;
pub const CL_BUILD_PROGRAM_FAILURE: cl_int = -11;
pub const CL_INVALID_PLATFORM: cl_int = -32;
pub const CL_INVALID_DEVICE: cl_int = -33;
pub const CL_INVALID_CONTEXT: cl_int = -34;
pub const CL_INVALID_COMMAND_QUEUE: cl_int = -36;
pub const CL_INVALID_MEM_OBJECT: cl_int = -38;
pub const CL_INVALID_PROGRAM: cl_int = -44;
pub const CL_INVALID_BUFFER_SIZE: cl_int = -61;

#[derive(Clone)]
pub struct ClErrorCode<'a> {
    pub value: cl_int,
    pub name: &'a str,
    pub description: &'a str,
}

impl<'a> fmt::Display for ClErrorCode<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} - {} ({})", self.value, self.name, self.description)
    }
}

impl<'a> ClErrorCode<'a> {
    pub(crate) const fn new(value: cl_int, name: &'a str, description: &'a str) -> Self {
        return Self {
            value,
            name: name,
            description,
        };
    }
    pub const Success: Self = Self::new(0, "CL_SUCCESS", "Success");
    pub const InvalidValue: Self = Self::new(
        -30,
        "CL_INVALID_VALUE",
        "One of the API arguments is not valid",
    );
}

pub const CL_SUCCESS: cl_int = ClErrorCode::Success.value;
pub const CL_INVALID_VALUE: cl_int = ClErrorCode::InvalidValue.value;
