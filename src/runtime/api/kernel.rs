use std::ops::{Deref, DerefMut};

use super::cl_types::*;
use crate::{
    format_error,
    interface::{ContextImpl, KernelImpl, KernelKind, MemKind, ProgramImpl, ProgramKind},
    lcl_contract,
    sync::SharedPtr,
};
use librecl_compiler::KernelArgType;

#[no_mangle]
pub unsafe extern "C" fn clCreateKernel(
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

    let program_safe = ProgramKind::try_from_cl(program).unwrap();

    let context = program_safe.get_context().upgrade().unwrap();

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

    return _cl_kernel::wrap(kernel);
}

#[no_mangle]
pub unsafe extern "C" fn clSetKernelArg(
    kernel: cl_kernel,
    arg_index: cl_uint,
    arg_size: cl_size_t,
    arg_value: *const libc::c_void,
) -> cl_int {
    // TODO proper error handling
    lcl_contract!(!kernel.is_null(), "kernel can't be NULL", CL_INVALID_VALUE);
    lcl_contract!(
        !arg_value.is_null(),
        "arg_value can't be NULL",
        CL_INVALID_VALUE
    );

    let mut kernel_safe = KernelKind::try_from_cl(kernel).unwrap();

    let arg_info = kernel_safe.deref().get_arg_info()[arg_index as usize].clone();

    match arg_info.arg_type {
        KernelArgType::GlobalBuffer => {
            let mem = MemKind::try_from_cl(unsafe { *(arg_value as *const cl_mem) }).unwrap();
            kernel_safe
                .deref_mut()
                .set_buffer_arg(arg_index as usize, SharedPtr::downgrade(&mem));
        }
        KernelArgType::POD => kernel_safe
            .deref_mut()
            .set_data_arg(arg_index as usize, unsafe {
                std::slice::from_raw_parts(arg_value as *const u8, arg_size as usize)
            }),
        _ => panic!("Unsupported!"),
    }

    return CL_SUCCESS;
}
