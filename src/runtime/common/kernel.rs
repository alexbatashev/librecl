use crate::common::context::Context;
use crate::{
    common::{cl_types::*, program::Program},
    format_error, lcl_contract,
};
use enum_dispatch::enum_dispatch;

#[enum_dispatch(ClKernel)]
pub trait Kernel {
    fn set_data_arg(&self, index: usize, bytes: &[u8]);
    fn set_buffer_arg(&self, index: usize);
}

#[cfg(feature = "vulkan")]
use crate::vulkan::Kernel as VkKernel;

#[cfg(feature = "metal")]
use crate::metal::Kernel as MTLKernel;

#[enum_dispatch]
#[repr(C)]
pub enum ClKernel {
    #[cfg(feature = "vulkan")]
    Vulkan(VkKernel),
    #[cfg(feature = "metal")]
    Metal(MTLKernel),
}

#[no_mangle]
pub extern "C" fn clCreateKernel(
    program: cl_program,
    kernel_name: *const libc::c_char,
    errcode_ret: *mut cl_int,
) -> cl_kernel {
    lcl_contract!(
        !program.is_null(),
        "program can't be NULL",
        CL_INVALID_PROGRAM,
        errcode_ret
    );

    let program_safe = unsafe { program.as_ref() }.unwrap();

    let context = unsafe { program_safe.get_context().as_ref() }.unwrap();

    lcl_contract!(
        context,
        !kernel_name.is_null(),
        "kernel_name can't be NULL",
        CL_INVALID_VALUE,
        errcode_ret
    );

    let kernel_name_safe = unsafe { std::ffi::CStr::from_ptr(kernel_name) }
        .to_str()
        .unwrap_or_default();
    lcl_contract!(
        context,
        !kernel_name_safe.is_empty(),
        "kernel_name can't be empty",
        CL_INVALID_VALUE,
        errcode_ret
    );

    let kernel = program_safe.create_kernel(kernel_name_safe);
    unsafe { *errcode_ret = CL_SUCCESS };

    return kernel;
}

#[no_mangle]
pub extern "C" fn clSetKernelArg(
    kernel: cl_kernel,
    arg_index: cl_uint,
    arg_size: libc::size_t,
    arg_value: *const libc::c_void,
) -> cl_int {
    unimplemented!();
}
