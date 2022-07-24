use crate::common::context::Context;
use crate::{
    common::{cl_types::*, program},
    format_error, lcl_contract,
};
use enum_dispatch::enum_dispatch;

#[cfg(feature = "vulkan")]
use crate::vulkan::Program as VkProgram;

#[cfg(feature = "metal")]
use crate::metal::Program as MTLProgram;

#[enum_dispatch(ClProgram)]
pub trait Program {}

#[enum_dispatch]
#[repr(C)]
pub enum ClProgram {
    #[cfg(feature = "vulkan")]
    Vulkan(VkProgram),
    #[cfg(feature = "metal")]
    Metal(MTLProgram),
}

#[no_mangle]
pub extern "C" fn clCreateProgramWithSource(
    context: cl_context,
    count: cl_uint,
    strings: *const *const libc::c_char,
    lengths: *const libc::size_t,
    errcode_ret: *mut cl_int,
) -> cl_program {
    lcl_contract!(
        !context.is_null(),
        "context must not be NULL",
        CL_INVALID_CONTEXT,
        errcode_ret
    );

    let ctx_safe = unsafe { context.as_ref() }.unwrap();

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

            let cur_string = unsafe { String::from_raw_parts(*s as *const u8 as *mut u8, *l, *l) };
            std::mem::forget(&cur_string);

            program_source.push_str(&cur_string);
        }
    }

    let program = ctx_safe.create_program_with_source(program_source);
    unsafe { *errcode_ret = CL_SUCCESS };

    return program;
}

#[no_mangle]
pub extern "C" fn clBuildProgram(
    program: cl_program,
    num_devices: cl_uint,
    device_list: *const cl_device_id,
    options: *const libc::c_char,
    callback: cl_build_callback,
    user_data: *mut libc::c_void,
) -> cl_int {
    unimplemented!()
}
