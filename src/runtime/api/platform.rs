use once_cell::sync::Lazy;
use std::sync::Arc;
use crate::interface::PlatformKind;
use crate::api::cl_types::*;
use crate::{lcl_contract, format_error, return_error, set_info_str};

static mut GLOBAL_PLATFORMS: Lazy<Vec<Arc<PlatformKind>>> = Lazy::new(|| {
    let mut platforms: Vec<Arc<PlatformKind>> = vec![];

    cfg_if::cfg_if! {
        if #[cfg(not(test))] {
        #[cfg(feature = "vulkan")]
        crate::vulkan::Platform::create_platforms(platforms.as_mut());

        #[cfg(feature = "metal")]
        crate::metal::Platform::create_platforms(platforms.as_mut());
        } else {
        #[cfg(test)]
        crate::mock::Platform::create_platforms(platforms.as_mut());
        }
    }

    return platforms;
});

#[no_mangle]
pub(crate) extern "C" fn clGetPlatformIDs(
    num_entries: cl_uint,
    platforms_raw: *mut cl_platform_id,
    num_platforms_raw: *mut cl_uint,
) -> cl_int {
    let num_platforms = unsafe { num_platforms_raw.as_ref() };
    let platforms = unsafe { platforms_raw.as_ref() };

    lcl_contract!(
        num_entries != 0 || !num_platforms.is_none(),
        "either num_platforms is not NULL or num_entries is not 0",
        CL_INVALID_VALUE
    );

    lcl_contract!(
        !platforms.is_none() || !num_platforms.is_none(),
        "num_platforms and platforms can not be NULL at the same time",
        CL_INVALID_VALUE
    );

    if !platforms.is_none() {
        let platforms_array = unsafe {
            std::slice::from_raw_parts_mut(
                platforms_raw as *mut cl_platform_id,
                num_entries as usize,
            )
        };
        for i in 0..num_entries {
            use std::ops::Deref;
            platforms_array[i as usize] = unsafe {
                (GLOBAL_PLATFORMS[i as usize].deref()) as *const _cl_platform_id as cl_platform_id
            };
        }
    }

    if !num_platforms.is_none() {
        unsafe {
            *num_platforms_raw = GLOBAL_PLATFORMS.len() as u32;
        };
    }
    return CL_SUCCESS;
}

#[no_mangle]
pub(crate) extern "C" fn clGetPlatformInfo(
    platform: cl_platform_id,
    param_name_num: u32,
    param_value_size: libc::size_t,
    param_value: *mut libc::c_void,
    param_size_ret: *mut libc::size_t,
) -> libc::c_int {
    let platform_safe = unsafe { platform.as_ref() };
    lcl_contract!(
        !platform.is_null(),
        "platfrom can't be NULL",
        CL_INVALID_PLATFORM
    );

    let param_name = PlatformInfoNames::try_from(param_name_num).map_err(|err| {
        ClError::new(
            ClErrorCode::InvalidValue,
            format!("Unknown param_name value {}", param_name_num),
        )
    });

    lcl_contract!(
        param_name.is_ok(),
        "invalid param_name value",
        CL_INVALID_VALUE
    );

    return_error!(param_name);

    match param_name {
        Ok(PlatformInfoNames::CL_PLATFORM_NAME) => {
            let platform_name = platform_safe.unwrap().get_platform_name();
            // TODO check param value size
            set_info_str!(platform_name, param_value, param_size_ret);
        }
        // Error has been handled before
        Err(err) => {},
        _ => {}
    };
    return CL_SUCCESS;
}

#[cfg(test)]
mod tests {
    use crate::api::cl_types::*;
    use crate::api::platform::*;
    #[test]
    fn all_null_pointers() {
        let err = clGetPlatformIDs(0, std::ptr::null_mut(), std::ptr::null_mut());
        assert_eq!(err, CL_INVALID_VALUE);
    }
    #[test]
    fn all_zeros() {
        let mut platforms: Vec<cl_platform_id> = vec![];

        let err = clGetPlatformIDs(0, platforms.as_mut_ptr(), std::ptr::null_mut());
        assert_eq!(err, CL_INVALID_VALUE);
    }
    #[test]
    fn returns_same_platforms() {
        let mut num_platforms: cl_uint = 0;
        let mut err = clGetPlatformIDs(0, std::ptr::null_mut(), &mut num_platforms);
        assert_eq!(err, CL_SUCCESS);

        let mut platforms1: Vec<cl_platform_id> = vec![];
        platforms1.resize(num_platforms as usize, std::ptr::null_mut());
        err = clGetPlatformIDs(num_platforms, platforms1.as_mut_ptr(), std::ptr::null_mut());
        assert_eq!(err, CL_SUCCESS);

        let mut platforms2: Vec<cl_platform_id> = vec![];
        platforms2.resize(num_platforms as usize, std::ptr::null_mut());
        err = clGetPlatformIDs(num_platforms, platforms2.as_mut_ptr(), std::ptr::null_mut());
        assert_eq!(err, CL_SUCCESS);

        for (a, b) in std::iter::zip(platforms1.iter(), platforms2.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn invalid_platform_info_name() {
        let mut num_platforms: cl_uint = 0;
        let mut err = clGetPlatformIDs(0, std::ptr::null_mut(), &mut num_platforms);
        assert_eq!(err, CL_SUCCESS);

        let mut platforms: Vec<cl_platform_id> = vec![];
        platforms.resize(num_platforms as usize, std::ptr::null_mut());
        err = clGetPlatformIDs(num_platforms, platforms.as_mut_ptr(), std::ptr::null_mut());
        assert_eq!(err, CL_SUCCESS);

        let mut size_ret: usize = 0;
        err = clGetPlatformInfo(platforms[0], 1000000, 0, std::ptr::null_mut(), &mut size_ret);
        assert_eq!(err, CL_INVALID_VALUE);
    }
}
