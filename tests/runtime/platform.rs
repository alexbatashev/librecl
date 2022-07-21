use opencl::c_cl::{self, CL_INVALID_VALUE, CL_PLATFORM_NAME, CL_SUCCESS};

#[test]
fn test_lcl_platforms_positive() {
    unsafe {
        let mut num_platforms: u32 = 0;
        let mut err = c_cl::clGetPlatformIDs(0, std::ptr::null_mut(), &mut num_platforms as *mut u32);
        assert_eq!(err, CL_SUCCESS);

        let mut platforms: Vec<c_cl::cl_platform_id> = vec![];
        platforms.resize(num_platforms as usize, std::ptr::null_mut());
        err = c_cl::clGetPlatformIDs(num_platforms, platforms.as_mut_ptr(), std::ptr::null_mut());
        assert_eq!(err, CL_SUCCESS);

        eprintln!("-----------------\nQuering param {}", CL_PLATFORM_NAME);
        println!("-----------------\nQuering param {}", CL_PLATFORM_NAME);
        println!("-----------------\nFOOOOOOOOOOOOOOOOOO");

        for p in platforms {


            let mut name_len: u64 = 0;
            let mut err = c_cl::clGetPlatformInfo(
                p,
                CL_PLATFORM_NAME,
                0,
                std::ptr::null_mut(),
                &mut name_len as *mut u64,
            );
            assert_eq!(err, CL_SUCCESS);
            let mut name_bytes: Vec<u8> = vec![];
            name_bytes.resize(name_len as usize, 0);
            err = c_cl::clGetPlatformInfo(
                p,
                CL_PLATFORM_NAME,
                name_len,
                name_bytes.as_mut_ptr() as *mut libc::c_void,
                std::ptr::null_mut(),
            );
            assert_eq!(err, CL_SUCCESS);
            let name = String::from_raw_parts(name_bytes.as_mut_ptr(), name_bytes.len(), name_bytes.capacity());

            // assert!(name.is_ok());
            println!("Platform name len is {}, name is {}", name_len, name.as_str());
            assert!(name.as_str().starts_with("LibreCL"));
        }
    }
}

#[test]
fn test_lcl_platforms_negative() {
    unsafe {
        let err = c_cl::clGetPlatformIDs(0, std::ptr::null_mut(), std::ptr::null_mut());
        assert_eq!(err, CL_INVALID_VALUE);
    }
}
