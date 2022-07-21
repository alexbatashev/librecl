use libc::c_int;
use libc::c_uint;
use libc::c_void;
use libc::size_t;

pub trait Platform: Sync {
    fn create_platforms() -> Vec<Box<dyn Platform>>
    where
        Self: Sized;
}

#[no_mangle]
pub extern "C" fn clGetPlatformInfo(
    platform: *const dyn Platform,
    param_name: libc::c_uint,
    param_value_size: libc::size_t,
    param_value: *mut libc::c_void,
    param_size_ret: *mut libc::size_t,
) -> libc::c_int {
    return 0;
}
