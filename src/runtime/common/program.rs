use crate::common::context::ClContext;
use crate::common::context::Context;
use crate::common::device::ClDevice;
use crate::{common::cl_types::*, format_error, lcl_contract};
use enum_dispatch::enum_dispatch;

#[cfg(feature = "vulkan")]
use crate::vulkan::Program as VkProgram;

#[cfg(feature = "metal")]
use crate::metal::Program as MTLProgram;

#[enum_dispatch(ClProgram)]
pub trait Program {
    fn get_context(&self) -> cl_context;
    fn get_safe_context_mut<'a, 'b>(&'a mut self) -> &'b mut ClContext;

    // TODO allow options
    fn compile_program(&mut self, devices: &[&ClDevice]) -> bool;
    // TODO allow options and multiple programs
    fn link_programs(&mut self, devices: &[&ClDevice]) -> bool;

    fn create_kernel(&self, program: cl_program, kernel_name: &str) -> cl_kernel;
}

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

    let program = ctx_safe.create_program_with_source(context, program_source);
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
    lcl_contract!(
        !program.is_null(),
        "program can't be NULL",
        CL_INVALID_PROGRAM
    );
    let mut program_safe = unsafe { program.as_mut() }.unwrap();

    let context = unsafe { program_safe.get_context().as_mut() }.unwrap();

    lcl_contract!(context, (device_list.is_null() && num_devices == 0) || (!device_list.is_null() && num_devices > 0), "either num_devices is > 0 and device_list is not NULL, or num_devices == 0 and device_list is NULL", CL_INVALID_VALUE);

    // TODO support options
    let build_function = |devices: &[&ClDevice], program: &mut ClProgram| {
        if !program.compile_program(devices) {
            return CL_BUILD_PROGRAM_FAILURE;
        }
        if !program.link_programs(devices) {
            return CL_BUILD_PROGRAM_FAILURE;
        }

        return CL_SUCCESS;
    };

    let devices_array: Vec<&ClDevice> = if num_devices > 0 {
        unsafe { std::slice::from_raw_parts(device_list, num_devices as usize) }
    } else {
        context.get_associated_devices()
    }
    .iter()
    .map(|d| unsafe { d.as_ref() }.unwrap())
    .collect();

    if callback.is_some() {
        let _guard = context.get_threading_runtime().enter();
        let safe_data = unsafe { user_data.as_ref() };
        tokio::spawn(async move {
            build_function(devices_array.as_slice(), program_safe);
            let user_data_unwrapped = if safe_data.is_some() {
                safe_data.unwrap() as *const libc::c_void as *mut libc::c_void
            } else {
                std::ptr::null_mut()
            };
            callback.unwrap()(program_safe as *mut ClProgram, user_data_unwrapped);
        });

        return CL_SUCCESS;
    } else {
        return build_function(devices_array.as_slice(), program_safe);
    }
}
