use super::test_utils::{create_devices, create_platform};
use lcl_icd_runtime::c_cl::{self, CL_SUCCESS};

#[test]
fn creation() {
    let platform = create_platform();
    let devices = create_devices(platform);

    let mut err: c_cl::cl_int = 0;

    let context = c_cl::clCreateContext(
        std::ptr::null(),
        devices.len() as c_cl::cl_uint,
        devices.as_ptr(),
        None,
        std::ptr::null_mut(),
        &mut err,
    );
    assert_eq!(err, CL_SUCCESS);
}
