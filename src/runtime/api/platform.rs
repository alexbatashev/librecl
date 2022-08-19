use super::error_handling::ClError;
use crate::api::cl_types::*;
use crate::interface::{PlatformImpl, PlatformKind};
use crate::sync::SharedPtr;
use crate::{lcl_contract, set_info_array, set_info_int, set_info_str, success};
use ocl_type_wrapper::cl_api;
use once_cell::sync::Lazy;
use std::ops::Deref;
use tracing::dispatcher::{self, Dispatch};
use tracing::info;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::Registry;

static mut GLOBAL_PLATFORMS: Lazy<Vec<SharedPtr<PlatformKind>>> = Lazy::new(|| {
    let has_tracing = std::env::var("LIBRECL_TRACE");

    if has_tracing.is_ok() {
        let subscriber = Registry::default().with(tracing_logfmt::layer());

        dispatcher::set_global_default(Dispatch::new(subscriber))
            .expect("Global logger has already been set!");
    }

    let mut platforms: Vec<SharedPtr<PlatformKind>> = vec![];

    cfg_if::cfg_if! {
        if #[cfg(not(test))] {
            let filter = std::env::var("LIBRECL_PLATFORM_FILTER").unwrap_or("all".to_owned());
            #[cfg(feature = "vulkan")]
            if filter.contains("vulkan") || filter.contains("all") {
                info!("Searching for Vulkan devices");
                crate::vulkan::Platform::create_platforms(platforms.as_mut());
            }

            #[cfg(feature = "metal")]
            if filter.contains("metal") || filter.contains("all") {
                info!("Searching for Apple Metal devices");
                crate::metal::Platform::create_platforms(platforms.as_mut());
            }

            if filter.contains("cpu") || filter.contains("all") {
                crate::cpu::Platform::create_platforms(platforms.as_mut());
            }
        } else {
            #[cfg(test)]
            crate::mock::Platform::create_platforms(platforms.as_mut());
        }
    }

    return platforms;
});

#[cl_api]
fn clGetPlatformIDs(
    num_entries: cl_uint,
    platforms_raw: *mut cl_platform_id,
    num_platforms_raw: *mut cl_uint,
) -> Result<(), ClError> {
    let num_platforms = unsafe { num_platforms_raw.as_ref() };
    let platforms = unsafe { platforms_raw.as_ref() };

    lcl_contract!(
        num_entries != 0 || !num_platforms.is_none(),
        ClError::InvalidValue,
        "either num_platforms is not NULL or num_entries is not 0"
    );

    lcl_contract!(
        !platforms.is_none() || !num_platforms.is_none(),
        ClError::InvalidValue,
        "num_platforms and platforms can not be NULL at the same time"
    );

    if !platforms.is_none() {
        let platforms_array = unsafe {
            std::slice::from_raw_parts_mut(
                platforms_raw as *mut cl_platform_id,
                num_entries as usize,
            )
        };
        for i in 0..num_entries {
            platforms_array[i as usize] =
                unsafe { GLOBAL_PLATFORMS[i as usize].deref().get_cl_handle() };
        }
    }

    if !num_platforms.is_none() {
        unsafe { *num_platforms_raw = GLOBAL_PLATFORMS.len() as u32 };
    }

    // TODO return error if there are no platforms
    return success!();
}

#[cl_api]
fn clGetPlatformInfo(
    platform: cl_platform_id,
    param_name_num: u32,
    _param_value_size: cl_size_t,
    param_value: *mut libc::c_void,
    param_size_ret: *mut cl_size_t,
) -> Result<(), ClError> {
    lcl_contract!(
        !platform.is_null(),
        ClError::InvalidPlatform,
        "platfrom can't be NULL"
    );

    let platform_safe = PlatformKind::try_from_cl(platform)
        .map_err(|_| ClError::InvalidPlatform("not a LibreCL platform".into()))?;

    let param_name = PlatformInfoNames::try_from(param_name_num).map_err(|_err| {
        ClError::InvalidValue(format!("Unknown param_name value {}", param_name_num).into())
    });

    match param_name {
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
        Ok(PlatformInfoNames::CL_PLATFORM_ICD_SUFFIX_KHR) => {
            let suffix = "LCL";
            set_info_str!(suffix, param_value, param_size_ret)
        }
        Err(err) => Err(err),
    }
}

#[no_mangle]
pub(crate) unsafe extern "C" fn clIcdGetPlatformIDsLCL(
    num_entries: cl_uint,
    platforms_raw: *mut cl_platform_id,
    num_platforms_raw: *mut cl_uint,
) -> cl_int {
    let result = cl_get_platform_ids_impl(num_entries, platforms_raw, num_platforms_raw);

    match result {
        Ok(_) => 0,
        Err(err) => err.error_code(),
    }
}

#[cfg(test)]
mod tests {
    use crate::api::cl_types::*;
    use crate::api::error_handling::error_codes::CL_SUCCESS;
    use crate::api::error_handling::error_codes::*;
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
