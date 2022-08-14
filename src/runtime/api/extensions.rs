use super::platform::clIcdGetPlatformIDsLCL;

#[no_mangle]
unsafe extern "C" fn clGetExtensionFunctionAddress(
    funcname: *const libc::c_char,
) -> *mut libc::c_void {
    let name = std::ffi::CStr::from_ptr(funcname).to_str().unwrap_or("");
    match name {
        "clIcdGetPlatformIDsKHR" => clIcdGetPlatformIDsLCL as *mut libc::c_void,
        _ => std::ptr::null_mut(),
    }
}
