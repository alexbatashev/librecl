use super::cl_types::*;
use crate::interface::{DeviceKind, ProgramKind};
use crate::sync::{SharedPtr, WeakPtr};
use crate::{
    format_error,
    interface::{ContextImpl, ContextKind, ProgramImpl},
    lcl_contract,
};

#[no_mangle]
pub unsafe extern "C" fn clCreateProgramWithSource(
    context: cl_context,
    count: cl_uint,
    strings: *mut *const libc::c_char,
    lengths: *const cl_size_t,
    errcode_ret: *mut cl_int,
) -> cl_program {
    lcl_contract!(
        !context.is_null(),
        "context must not be NULL",
        CL_INVALID_CONTEXT,
        errcode_ret
    );

    let ctx_safe = ContextKind::try_from_cl(context).unwrap();

    lcl_contract!(
        ctx_safe,
        count != 0,
        "program must have at least one line",
        CL_INVALID_VALUE,
        errcode_ret
    );
    lcl_contract!(
        ctx_safe,
        !strings.is_null(),
        "strings can't be NULL",
        CL_INVALID_VALUE,
        errcode_ret
    );

    let mut program_source: String = String::new();

    let strings_array = unsafe { std::slice::from_raw_parts(strings, count as usize) };

    if lengths.is_null() {
        for s in strings_array {
            lcl_contract!(
                ctx_safe,
                !s.is_null(),
                "neither of strings can be NULL",
                CL_INVALID_VALUE,
                errcode_ret
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
                "neither of strings can be NULL",
                CL_INVALID_VALUE,
                errcode_ret
            );

            let cur_string = unsafe {
                String::from_raw_parts(*s as *const u8 as *mut u8, *l as usize, *l as usize)
            };
            std::mem::forget(&cur_string);

            program_source.push_str(&cur_string);
        }
    }

    let program = ctx_safe.create_program_with_source(program_source);
    unsafe { *errcode_ret = CL_SUCCESS };

    return _cl_program::wrap(program);
}

#[no_mangle]
pub unsafe extern "C" fn clBuildProgram(
    program: cl_program,
    num_devices: cl_uint,
    device_list: *const cl_device_id,
    options: *const libc::c_char,
    callback: cl_build_callback,
    user_data: *mut libc::c_void,
) -> cl_int {
    lcl_contract!(
        !program.is_null(),
        "program can't be NULL",
        CL_INVALID_PROGRAM
    );
    let mut program_safe = ProgramKind::try_from_cl(program).unwrap();

    let context = program_safe.get_context().upgrade().unwrap();

    lcl_contract!(context, (device_list.is_null() && num_devices == 0) || (!device_list.is_null() && num_devices > 0), "either num_devices is > 0 and device_list is not NULL, or num_devices == 0 and device_list is NULL", CL_INVALID_VALUE);

    // TODO support options
    let build_function = |devices: Vec<WeakPtr<DeviceKind>>,
                          mut program: SharedPtr<ProgramKind>| {
        if !program.compile_program(devices.as_slice()) {
            return CL_BUILD_PROGRAM_FAILURE;
        }
        if !program.link_programs(devices.as_slice()) {
            return CL_BUILD_PROGRAM_FAILURE;
        }

        return CL_SUCCESS;
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
        let safe_data = unsafe { user_data.as_ref() };
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

        return CL_SUCCESS;
    } else {
        return build_function(devices_array, program_safe);
    }
}

#[no_mangle]
pub unsafe extern "C" fn clRetainProgram(program: cl_program) -> cl_int {
    lcl_contract!(
        !program.is_null(),
        "program can't be NULL",
        CL_INVALID_PROGRAM
    );

    let program_ref = &mut *program;

    program_ref.retain();

    return CL_SUCCESS;
}

#[no_mangle]
pub unsafe extern "C" fn clReleaseProgram(program: cl_program) -> cl_int {
    lcl_contract!(
        !program.is_null(),
        "program can't be NULL",
        CL_INVALID_PROGRAM
    );

    let program_ref = &mut *program;

    if program_ref.release() == 1 {
        // Intentionally ignore value to destroy pointer and its content
        Box::from_raw(program);
    }

    return CL_SUCCESS;
}
