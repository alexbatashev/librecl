use super::error_handling::ClError;
use crate::api::cl_types::*;
use crate::interface::{PlatformImpl, PlatformKind};
use crate::sync::SharedPtr;
use crate::{format_error, lcl_contract, return_error, set_info_array, set_info_int, set_info_str};
use once_cell::sync::Lazy;
use std::ops::Deref;

static mut GLOBAL_PLATFORMS: Lazy<Vec<SharedPtr<PlatformKind>>> = Lazy::new(|| {
    let mut platforms: Vec<SharedPtr<PlatformKind>> = vec![];

    cfg_if::cfg_if! {
        if #[cfg(not(test))] {
        #[cfg(feature = "vulkan")]
        crate::vulkan::Platform::create_platforms(platforms.as_mut());

        #[cfg(feature = "metal")]
        crate::metal::Platform::create_platforms(platforms.as_mut());

        crate::cpu::Platform::create_platforms(platforms.as_mut());
        } else {
        #[cfg(test)]
        crate::mock::Platform::create_platforms(platforms.as_mut());
        }
    }

    return platforms;
});

#[no_mangle]
pub(crate) unsafe extern "C" fn clGetPlatformIDs(
    num_entries: cl_uint,
    platforms_raw: *mut cl_platform_id,
    num_platforms_raw: *mut cl_uint,
) -> cl_int {
    let num_platforms = num_platforms_raw.as_ref();
    let platforms = platforms_raw.as_ref();

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
        let platforms_array = std::slice::from_raw_parts_mut(
            platforms_raw as *mut cl_platform_id,
            num_entries as usize,
        );
        for i in 0..num_entries {
            platforms_array[i as usize] = GLOBAL_PLATFORMS[i as usize].deref().get_cl_handle();
        }
    }

    if !num_platforms.is_none() {
        *num_platforms_raw = GLOBAL_PLATFORMS.len() as u32;
    }
    return CL_SUCCESS;
}

#[no_mangle]
pub(crate) unsafe extern "C" fn clGetPlatformInfo(
    platform: cl_platform_id,
    param_name_num: u32,
    _param_value_size: cl_size_t,
    param_value: *mut libc::c_void,
    param_size_ret: *mut cl_size_t,
) -> libc::c_int {
    lcl_contract!(
        !platform.is_null(),
        "platfrom can't be NULL",
        CL_INVALID_PLATFORM
    );

    let platform_safe = PlatformKind::try_from_cl(platform).unwrap();

    let param_name = PlatformInfoNames::try_from(param_name_num).map_err(|_err| {
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

    let result: Result<(), ClError> = match param_name {
        Ok(PlatformInfoNames::CL_PLATFORM_PROFILE) => {
            let profile = platform_safe.get_profile();
            // TODO check param value size
            set_info_str!(profile, param_value, param_size_ret)
        }
        Ok(PlatformInfoNames::CL_PLATFORM_VERSION) => {
            let version = String::from("OpenCL 3.0 ") + platform_safe.get_platform_version_info();
            set_info_str!(version, param_value, param_size_ret)
        }
        Ok(PlatformInfoNames::CL_PLATFORM_NUMERIC_VERSION) => {
            let version = make_version(3, 0, 0);
            set_info_int!(cl_version, version, param_value, param_size_ret)
        }
        Ok(PlatformInfoNames::CL_PLATFORM_NAME) => {
            let platform_name = platform_safe.get_platform_name();
            set_info_str!(platform_name, param_value, param_size_ret)
        }
        Ok(PlatformInfoNames::CL_PLATFORM_VENDOR) => {
            let platform_vendor = "LibreCL";
            set_info_str!(platform_vendor, param_value, param_size_ret)
        }
        Ok(PlatformInfoNames::CL_PLATFORM_EXTENSIONS) => {
            let extensions_vec = platform_safe.get_extension_names().to_vec();
            let extensions = extensions_vec.join(" ");
            set_info_str!(extensions, param_value, param_size_ret)
        }
        Ok(PlatformInfoNames::CL_PLATFORM_EXTENSIONS_WITH_VERSION) => {
            let names = platform_safe.get_extension_names();
            let versions = platform_safe.get_extension_versions();

            let extensions: Vec<_> = names
                .iter()
                .zip(versions.iter())
                .map(|(&n, &v)| cl_name_version {
                    version: v,
                    name: n.as_bytes().try_into().expect("failed to convert to array"),
                })
                .collect();
            set_info_array!(cl_name_version, extensions, param_value, param_size_ret)
        }
        Ok(PlatformInfoNames::CL_PLATFORM_HOST_TIMER_RESOLUTION) => {
            let resolution = platform_safe.get_host_timer_resolution();
            set_info_int!(cl_ulong, resolution, param_value, param_size_ret)
        }
        // Error has been handled before
        Err(err) => Err(err),
    };

    // TODO log error message
    return if result.is_ok() {
        CL_SUCCESS
    } else {
        match result {
            Err(err) => err.error_code.value,
            _ => panic!("unexpected"),
        }
    };
}

#[no_mangle]
pub(crate) unsafe extern "C" fn clIcdGetPlatformIDsKHR(
    num_entries: cl_uint,
    platforms_raw: *mut cl_platform_id,
    num_platforms_raw: *mut cl_uint,
) -> cl_int {
    return clGetPlatformIDs(num_entries, platforms_raw, num_platforms_raw);
}

#[cfg(test)]
mod tests {
    use crate::api::cl_types::*;
    use crate::api::platform::*;
    #[test]
    fn all_null_pointers() {
        let err = unsafe { clGetPlatformIDs(0, std::ptr::null_mut(), std::ptr::null_mut()) };
        assert_eq!(err, CL_INVALID_VALUE);
    }
    #[test]
    fn all_zeros() {
        let mut platforms: Vec<cl_platform_id> = vec![];

        let err = unsafe { clGetPlatformIDs(0, platforms.as_mut_ptr(), std::ptr::null_mut()) };
        assert_eq!(err, CL_INVALID_VALUE);
    }
    #[test]
    fn returns_same_platforms() {
        let mut num_platforms: cl_uint = 0;
        let mut err = unsafe { clGetPlatformIDs(0, std::ptr::null_mut(), &mut num_platforms) };
        assert_eq!(err, CL_SUCCESS);

        let mut platforms1: Vec<cl_platform_id> = vec![];
        platforms1.resize(num_platforms as usize, std::ptr::null_mut());
        err = unsafe {
            clGetPlatformIDs(num_platforms, platforms1.as_mut_ptr(), std::ptr::null_mut())
        };
        assert_eq!(err, CL_SUCCESS);

        let mut platforms2: Vec<cl_platform_id> = vec![];
        platforms2.resize(num_platforms as usize, std::ptr::null_mut());
        err = unsafe {
            clGetPlatformIDs(num_platforms, platforms2.as_mut_ptr(), std::ptr::null_mut())
        };
        assert_eq!(err, CL_SUCCESS);

        for (a, b) in std::iter::zip(platforms1.iter(), platforms2.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn invalid_platform_info_name() {
        let mut num_platforms: cl_uint = 0;
        let mut err = unsafe { clGetPlatformIDs(0, std::ptr::null_mut(), &mut num_platforms) };
        assert_eq!(err, CL_SUCCESS);

        let mut platforms: Vec<cl_platform_id> = vec![];
        platforms.resize(num_platforms as usize, std::ptr::null_mut());
        err = unsafe {
            clGetPlatformIDs(num_platforms, platforms.as_mut_ptr(), std::ptr::null_mut())
        };
        assert_eq!(err, CL_SUCCESS);

        let size_ret: usize = 0;
        err = unsafe {
            clGetPlatformInfo(
                platforms[0],
                1000000,
                0,
                std::ptr::null_mut(),
                &mut (size_ret as cl_size_t),
            )
        };
        assert_eq!(err, CL_INVALID_VALUE);
    }
}
