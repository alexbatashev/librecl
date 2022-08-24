use super::cl_types::*;
use crate::{
    api::error_handling::{map_invalid_kernel, map_invalid_mem, map_invalid_program, ClError},
    success,
};
use crate::{
    interface::{ContextImpl, KernelImpl, KernelKind, MemKind, ProgramImpl, ProgramKind},
    lcl_contract,
    sync::SharedPtr,
};
use librecl_compiler::KernelArgType;
use lcl_derive::cl_api;
use std::ops::{Deref, DerefMut};

#[cl_api]
fn clCreateKernel(
    program: cl_program,
    kernel_name: *const libc::c_char,
) -> Result<cl_kernel, ClError> {
    let program_safe = ProgramKind::try_from_cl(program).map_err(map_invalid_program)?;

    let context = program_safe
        .get_context()
        .upgrade()
        .ok_or(())
        .map_err(|_| {
            ClError::InvalidContext(
                "failed to acquire owning reference to queue. Was it released before?".into(),
            )
        })?;

    lcl_contract!(
        context,
        !kernel_name.is_null(),
        ClError::InvalidValue,
        "kernel_name can't be NULL"
    );

    let kernel_name_safe = unsafe { std::ffi::CStr::from_ptr(kernel_name) }
        .to_str()
        .unwrap_or_default();
    lcl_contract!(
        context,
        !kernel_name_safe.is_empty(),
        ClError::InvalidValue,
        "kernel_name can't be empty"
    );

    // TODO return Result<KernelKind, ClError>
    let kernel = program_safe.create_kernel(kernel_name_safe);

    return Ok(_cl_kernel::wrap(kernel));
}

#[cl_api]
fn clSetKernelArg(
    kernel: cl_kernel,
    arg_index: cl_uint,
    arg_size: cl_size_t,
    arg_value: *const libc::c_void,
) -> Result<(), ClError> {
    // TODO proper error handling
    lcl_contract!(
        !arg_value.is_null(),
        ClError::InvalidValue,
        "arg_value can't be NULL"
    );

    let mut kernel_safe = KernelKind::try_from_cl(kernel).map_err(map_invalid_kernel)?;

    let arg_info = kernel_safe.deref().get_arg_info()[arg_index as usize].clone();

    match arg_info.arg_type {
        KernelArgType::GlobalBuffer => {
            let mem = MemKind::try_from_cl(unsafe { *(arg_value as *const cl_mem) })
                .map_err(map_invalid_mem)?;
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

    return success!();
}

#[cl_api]
fn clRetainKernel(kernel: cl_kernel) -> Result<(), ClError> {
    lcl_contract!(
        !kernel.is_null(),
        ClError::InvalidKernel,
        "kernel can't be NULL"
    );

    let kernel_ref = unsafe { &mut *kernel };

    kernel_ref.retain();

    return success!();
}

#[cl_api]
fn clReleaseKernel(kernel: cl_kernel) -> Result<(), ClError> {
    lcl_contract!(
        !kernel.is_null(),
        ClError::InvalidKernel,
        "kernel can't be NULL"
    );

    let kernel_ref = unsafe { &mut *kernel };

    if kernel_ref.release() == 1 {
        // Intentionally ignore value to destroy pointer and its content
        unsafe { Box::from_raw(kernel) };
    }

    return success!();
}
