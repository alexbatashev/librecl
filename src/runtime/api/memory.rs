use super::cl_types::*;
use super::error_handling::map_invalid_context;
use crate::api::error_handling::ClError;
use crate::success;
use crate::{
    interface::{ContextImpl, ContextKind},
    lcl_contract,
};
use lcl_derive::cl_api;

#[cl_api]
pub unsafe extern "C" fn clCreateBuffer(
    context: cl_context,
    flags: cl_mem_flags,
    size: cl_size_t,
    _host_ptr: *mut libc::c_void,
) -> Result<cl_mem, ClError> {
    let mut context_safe = ContextKind::try_from_cl(context).map_err(map_invalid_context)?;

    lcl_contract!(
        context_safe,
        size > 0,
        ClError::InvalidBufferSize,
        "size must be greater than 0"
    );

    // TODO check flags
    let mem = context_safe.create_buffer(size as usize, flags);

    return Ok(_cl_mem::wrap(mem));
}

#[cl_api]
fn clRetainMemObject(mem_obj: cl_mem) -> Result<(), ClError> {
    lcl_contract!(
        !mem_obj.is_null(),
        ClError::InvalidMemObject,
        "mem_obj can't be NULL"
    );

    let mem_obj_ref = unsafe { &mut *mem_obj };

    mem_obj_ref.retain();

    return success!();
}

#[cl_api]
fn clReleaseMemObject(mem_obj: cl_mem) -> Result<(), ClError> {
    lcl_contract!(
        !mem_obj.is_null(),
        ClError::InvalidMemObject,
        "mem_obj can't be NULL"
    );

    let mem_obj_ref = unsafe { &mut *mem_obj };

    if mem_obj_ref.release() == 1 {
        // Intentionally ignore value to destroy pointer and its content
        unsafe { Box::from_raw(mem_obj) };
    }

    return success!();
}
