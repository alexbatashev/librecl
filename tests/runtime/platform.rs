use opencl::c_cl::{self, CL_INVALID_VALUE, CL_SUCCESS};

#[test]
fn test_lcl_platforms_positive() {
    unsafe {
        let mut num_platforms: u32 = 0;
        let err = c_cl::clGetPlatformIDs(0, std::ptr::null_mut(), &mut num_platforms as *mut u32);
        assert_eq!(err, CL_SUCCESS);

        let mut platforms: Vec<c_cl::cl_platform_id> = vec![];
        platforms.resize(num_platforms as usize, std::ptr::null_mut());
        c_cl::clGetPlatformIDs(num_platforms, platforms.as_mut_ptr(), std::ptr::null_mut());
    }
}

#[test]
fn test_lcl_platforms_negative() {
    unsafe {
        let err = c_cl::clGetPlatformIDs(0, std::ptr::null_mut(), std::ptr::null_mut());
        assert_eq!(err, CL_INVALID_VALUE);
    }
}
