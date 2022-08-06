use super::cl_types::*;
use crate::{
    format_error,
    interface::{ContextImpl, ContextKind},
    lcl_contract,
};

#[no_mangle]
pub unsafe extern "C" fn clCreateBuffer(
    context: cl_context,
    flags: cl_mem_flags,
    size: cl_size_t,
    _host_ptr: *mut libc::c_void,
    errcode_ret: *mut cl_int,
) -> cl_mem {
    lcl_contract!(
        !context.is_null(),
        "context can't be NULL",
        CL_INVALID_CONTEXT,
        errcode_ret
    );

    let mut context_safe = ContextKind::try_from_cl(context).unwrap();

    lcl_contract!(
        context_safe,
        size > 0,
        "size must be greater than 0",
        CL_INVALID_BUFFER_SIZE,
        errcode_ret
    );

    // TODO check flags
    let mem = context_safe.create_buffer(size as usize, flags);
    *errcode_ret = CL_SUCCESS;

    return _cl_mem::wrap(mem);
}

#[no_mangle]
pub unsafe extern "C" fn clRetainMemObject(mem_obj: cl_mem) -> cl_int {
    lcl_contract!(
        !mem_obj.is_null(),
        "mem_obj can't be NULL",
        CL_INVALID_MEM_OBJECT
    );

    let mem_obj_ref = &mut *mem_obj;

    mem_obj_ref.retain();

    return CL_SUCCESS;
}

#[no_mangle]
pub unsafe extern "C" fn clReleaseMemObject(mem_obj: cl_mem) -> cl_int {
    lcl_contract!(
        !mem_obj.is_null(),
        "mem_obj can't be NULL",
        CL_INVALID_MEM_OBJECT
    );

    let mem_obj_ref = &mut *mem_obj;

    if mem_obj_ref.release() == 1 {
        // Intentionally ignore value to destroy pointer and its content
        Box::from_raw(mem_obj);
    }

    return CL_SUCCESS;
}
