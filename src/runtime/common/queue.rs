use crate::common::cl_types::*;

pub trait Queue {}

#[no_mangle]
pub extern "C" fn clCreateCommandQueueWithProperties(
    context: cl_context,
    device: cl_device_id,
    properties: *const cl_queue_properties,
    errorcode_ret: *mut cl_int,
) -> cl_command_queue {
    unimplemented!();
}
