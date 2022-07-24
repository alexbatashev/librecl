use crate::common::context::Context as CommonContext;
use crate::common::device::Device as CommonDevice;
use crate::{common::cl_types::*, format_error, lcl_contract};
use enum_dispatch::enum_dispatch;

#[cfg(feature = "vulkan")]
use crate::vulkan::InOrderQueue as VkInOrderQueue;

#[cfg(feature = "metal")]
use crate::metal::Queue as MTLQueue;

#[enum_dispatch(ClQueue)]
pub trait Queue {}

#[enum_dispatch]
#[repr(C)]
pub enum ClQueue {
    #[cfg(feature = "vulkan")]
    VulkanInOrder(VkInOrderQueue),
    #[cfg(feature = "metal")]
    Metal(MTLQueue),
}

#[no_mangle]
pub extern "C" fn clCreateCommandQueueWithProperties(
    context: cl_context,
    device: cl_device_id,
    _properties: *const cl_queue_properties,
    errcode_ret: *mut cl_int,
) -> cl_command_queue {
    // TODO do not ignore properties

    lcl_contract!(
        !context.is_null(),
        "context can't be NULL",
        CL_INVALID_CONTEXT,
        errcode_ret
    );

    let context_safe = unsafe { context.as_ref() }.unwrap();

    lcl_contract!(
        context_safe,
        !device.is_null(),
        "device can't be NULL",
        CL_INVALID_DEVICE,
        errcode_ret
    );
    lcl_contract!(
        context_safe,
        context_safe.has_device(device),
        "device must belong to the provided context",
        CL_INVALID_DEVICE,
        errcode_ret
    );

    let device_safe = unsafe { device.as_ref() }.unwrap();

    let queue = device_safe.create_queue(context, device);

    unsafe { *errcode_ret = CL_SUCCESS };

    return queue;
}
