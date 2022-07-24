use super::test_utils::create_platform;
use lcl_icd_runtime::c_cl::{
    self, CL_DEVICE_TYPE_ALL, CL_INVALID_VALUE, CL_PLATFORM_NAME, CL_SUCCESS,
};

#[test]
fn creation() {
    let platform = create_platform();

    let mut num_devices: c_cl::cl_uint = 0;

    let mut err = c_cl::clGetDeviceIDs(
        platform,
        c_cl::cl_device_type::all(), // todo this needs to match the spec
        0,
        std::ptr::null_mut(),
        &mut num_devices as *mut c_cl::cl_uint,
    );
    assert_eq!(err, CL_SUCCESS);

    let mut devices: Vec<c_cl::cl_device_id> = vec![];
    devices.resize(num_devices as usize, std::ptr::null_mut());
    err = c_cl::clGetDeviceIDs(
        platform,
        c_cl::cl_device_type::all(),
        num_devices,
        devices.as_mut_ptr(),
        std::ptr::null_mut(),
    );
    assert_eq!(err, CL_SUCCESS);
}
