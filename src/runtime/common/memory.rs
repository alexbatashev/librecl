use crate::common::context::Context;
use crate::{
    common::{cl_types::*, program::Program},
    format_error, lcl_contract,
};
use enum_dispatch::enum_dispatch;

#[enum_dispatch(ClMem)]
pub trait MemObject {}

#[cfg(feature = "vulkan")]
use crate::vulkan::Buffer as VkBuffer;

#[cfg(feature = "metal")]
use crate::metal::Buffer as MTLBuffer;

#[enum_dispatch]
#[repr(C)]
pub enum ClMem {
    #[cfg(feature = "vulkan")]
    VulkanBuffer(VkBuffer),
    #[cfg(feature = "metal")]
    MetalBuffer(MTLBuffer),
}

#[no_mangle]
pub extern "C" fn clCreateBuffer(
    context: cl_context,
    flags: cl_mem_flags,
    size: libc::size_t,
    host_ptr: *mut libc::c_void,
    errcode_ret: *mut cl_int,
) -> cl_mem {
    lcl_contract!(
        !context.is_null(),
        "context can't be NULL",
        CL_INVALID_CONTEXT,
        errcode_ret
    );

    let context_safe = unsafe { context.as_mut() }.unwrap();

    lcl_contract!(
        context_safe,
        size > 0,
        "size must be greater than 0",
        CL_INVALID_BUFFER_SIZE,
        errcode_ret
    );

    // TODO check flags
    let mem = context_safe.create_buffer(size, flags);
    unsafe {
        *errcode_ret = CL_SUCCESS;
    }
    return mem;
}
