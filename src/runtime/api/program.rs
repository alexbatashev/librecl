use super::cl_types::*;
use super::error_handling::ClError;
use crate::api::error_handling::{map_invalid_context, map_invalid_program};
use crate::interface::{DeviceKind, ProgramKind};
use crate::success;
use crate::sync::{SharedPtr, WeakPtr};
use crate::{
    interface::{ContextImpl, ContextKind, ProgramImpl},
    lcl_contract,
};
use ocl_type_wrapper::cl_api;

#[cl_api]
fn clCreateProgramWithSource(
    context: cl_context,
    count: cl_uint,
    strings: *mut *const libc::c_char,
    lengths: *const cl_size_t,
) -> Result<cl_program, ClError> {
    let ctx_safe = ContextKind::try_from_cl(context).map_err(map_invalid_context)?;

    lcl_contract!(
        ctx_safe,
        count != 0,
        ClError::InvalidValue,
        "program must have at least one line"
    );
    lcl_contract!(
        ctx_safe,
        !strings.is_null(),
        ClError::InvalidValue,
        "strings can't be NULL"
    );

    let mut program_source: String = String::new();

    let strings_array = unsafe { std::slice::from_raw_parts(strings, count as usize) };

    if lengths.is_null() {
        for s in strings_array {
            lcl_contract!(
                ctx_safe,
                !s.is_null(),
                ClError::InvalidValue,
                "neither of strings can be NULL"
            );
            let c_str = unsafe { std::ffi::CStr::from_ptr(*s) };
            program_source.push_str(c_str.to_str().unwrap());
        }
    } else {
        let lengths_array = unsafe { std::slice::from_raw_parts(lengths, count as usize) };
        for (s, l) in strings_array.into_iter().zip(lengths_array) {
            lcl_contract!(
                ctx_safe,
                !s.is_null(),
                ClError::InvalidValue,
                "neither of strings can be NULL"
            );

            let cur_string = unsafe {
                String::from_raw_parts(*s as *const u8 as *mut u8, *l as usize, *l as usize)
            };
            std::mem::forget(&cur_string);

            program_source.push_str(&cur_string);
        }
    }

    // TODO switch to Result<ProgramKind, ClError>
    let program = ctx_safe.create_program_with_source(program_source);

    return Ok(_cl_program::wrap(program));
}

#[cl_api]
pub unsafe extern "C" fn clCreateProgramWithIL(
    context: cl_context,
    il: *const libc::c_void,
    length: cl_size_t,
) -> Result<cl_program, ClError> {
    let ctx_safe = ContextKind::try_from_cl(context).map_err(map_invalid_context)?;

    lcl_contract!(
        ctx_safe,
        il.is_null(),
        ClError::InvalidValue,
        "il must not be NULL"
    );
    lcl_contract!(
        ctx_safe,
        length == 0,
        ClError::InvalidValue,
        "length must not be 0"
    );

    let spirv = unsafe { std::slice::from_raw_parts(il as *const i8, length as usize) };

    let program = ctx_safe.create_program_with_spirv(spirv)?;

    Ok(_cl_program::wrap(program))
}

#[no_mangle]
pub unsafe extern "C" fn clCreateProgramWithILKHR(
    context: cl_context,
    il: *const libc::c_void,
    length: cl_size_t,
    errcode_ret: *mut cl_int,
) -> cl_program {
    return clCreateProgramWithILLCL(context, il, length, errcode_ret);
}

#[cl_api]
fn clBuildProgram(
    program: cl_program,
    num_devices: cl_uint,
    device_list: *const cl_device_id,
    _options: *const libc::c_char,
    callback: cl_build_callback,
    user_data: *mut libc::c_void,
) -> Result<(), ClError> {
    let program_safe = ProgramKind::try_from_cl(program).map_err(map_invalid_program)?;

    let context = program_safe
        .get_context()
        .upgrade()
        .ok_or(())
        .map_err(|_| {
            ClError::InvalidProgram(
                "failed to acquire owning reference to program. Was it released before?".into(),
            )
        })?;

    lcl_contract!(
        context,
        ((device_list.is_null() && num_devices == 0) || (!device_list.is_null() && num_devices > 0)),
        ClError::InvalidValue,
        "either num_devices is > 0 and device_list is not NULL, or num_devices == 0 and device_list is NULL"
    );

    // TODO support options
    let build_function = |devices: Vec<WeakPtr<DeviceKind>>,
                          mut program: SharedPtr<ProgramKind>|
     -> Result<(), ClError> {
        if !program.compile_program(devices.as_slice()) {
            // TODO proper error
            return Err(ClError::BuildProgramFailure("".into()));
        }
        if !program.link_programs(devices.as_slice()) {
            // TODO proper error
            return Err(ClError::BuildProgramFailure("".into()));
        }

        return success!();
    };

    let devices_array = if num_devices > 0 {
        (unsafe { std::slice::from_raw_parts(device_list, num_devices as usize) }
            .iter()
            .map(|&d| SharedPtr::downgrade(&DeviceKind::try_from_cl(d).unwrap())))
        .collect::<Vec<WeakPtr<DeviceKind>>>()
    } else {
        context
            .get_associated_devices()
            .iter()
            .map(|d| d.clone())
            .collect::<Vec<WeakPtr<DeviceKind>>>()
    };

    if callback.is_some() {
        let _guard = context.get_threading_runtime().enter();
        let _safe_data = unsafe { user_data.as_ref() };
        // TODO figure out how to make the whole thing thread-safe
        /*
        tokio::spawn(async move {
            build_function(devices_array, program_safe);
            let user_data_unwrapped = if safe_data.is_some() {
                safe_data.unwrap() as *const libc::c_void as *mut libc::c_void
            } else {
                std::ptr::null_mut()
            };
            callback.unwrap()(program, user_data_unwrapped);
        });
        */

        return success!();
    } else {
        return build_function(devices_array, program_safe);
    }
}

#[cl_api]
fn clRetainProgram(program: cl_program) -> Result<(), ClError> {
    lcl_contract!(
        !program.is_null(),
        ClError::InvalidProgram,
        "program can't be NULL"
    );

    let program_ref = unsafe { &mut *program };

    program_ref.retain();

    return success!();
}

#[cl_api]
fn clReleaseProgram(program: cl_program) -> Result<(), ClError> {
    lcl_contract!(
        !program.is_null(),
        ClError::InvalidProgram,
        "program can't be NULL"
    );

    let program_ref = unsafe { &mut *program };

    if program_ref.release() == 1 {
        // Intentionally ignore value to destroy pointer and its content
        unsafe { Box::from_raw(program) };
    }

    return success!();
}
