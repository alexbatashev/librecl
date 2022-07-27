use crate::common::context::Context as CommonContext;
use crate::common::device::Device as CommonDevice;
use crate::{common::cl_types::*, format_error, lcl_contract};
use enum_dispatch::enum_dispatch;

#[cfg(feature = "vulkan")]
use crate::vulkan::InOrderQueue as VkInOrderQueue;

#[cfg(feature = "metal")]
use crate::metal::Queue as MTLQueue;

use super::memory::ClMem;

#[enum_dispatch(ClQueue)]
pub trait Queue {
    // TODO return event, todo async
    fn enqueue_buffer_write(&self, src: *const libc::c_void, dst: cl_mem);
    fn enqueue_buffer_read(&self, src: cl_mem, dst: *mut libc::c_void);
    fn submit(
        &self,
        kernel: cl_kernel,
        offset: [u32; 3],
        global_size: [u32; 3],
        local_size: [u32; 3],
    );
    fn finish(&self);
}

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

#[no_mangle]
pub extern "C" fn clEnqueueWriteBuffer(
    command_queue: cl_command_queue,
    buffer: cl_mem,
    blocking_write: cl_bool,
    offset: libc::size_t,
    cb: libc::size_t,
    ptr: *const libc::c_void,
    num_events_in_wait_list: cl_uint,
    // TODO support events
    event_wait_list: *const libc::c_void,
    event: *const libc::c_void,
) -> cl_int {
    lcl_contract!(
        !command_queue.is_null(),
        "queue can't be null",
        CL_INVALID_COMMAND_QUEUE
    );

    let queue = unsafe { command_queue.as_ref() }.unwrap();

    lcl_contract!(
        !buffer.is_null(),
        "buffer can't be NULL",
        CL_INVALID_MEM_OBJECT
    );

    // TODO proper error handling

    // TODO blocking - non-blocking
    queue.enqueue_buffer_write(ptr, buffer);

    return CL_SUCCESS;
}

#[no_mangle]
pub extern "C" fn clEnqueueReadBuffer(
    command_queue: cl_command_queue,
    buffer: cl_mem,
    blocking_read: cl_bool,
    offset: libc::size_t,
    cb: libc::size_t,
    ptr: *mut libc::c_void,
    num_events_in_wait_list: cl_uint,
    // TODO support events
    event_wait_list: *const libc::c_void,
    event: *const libc::c_void,
) -> cl_int {
    lcl_contract!(
        !command_queue.is_null(),
        "queue can't be null",
        CL_INVALID_COMMAND_QUEUE
    );

    let queue = unsafe { command_queue.as_ref() }.unwrap();

    lcl_contract!(
        !buffer.is_null(),
        "buffer can't be NULL",
        CL_INVALID_MEM_OBJECT
    );

    // TODO proper error handling

    // TODO blocking - non-blocking
    queue.enqueue_buffer_read(buffer, ptr);

    return CL_SUCCESS;
}

#[no_mangle]
pub extern "C" fn clEnqueueNDRangeKernel(
    command_queue: cl_command_queue,
    kernel: cl_kernel,
    work_dim: cl_uint,
    global_work_offset: *const libc::size_t,
    global_work_size: *const libc::size_t,
    local_work_size: *const libc::size_t,
    num_events_in_wait_list: cl_uint,
    event_wait_list: *const libc::c_void,
    event: *const libc::c_void,
) -> cl_int {
    lcl_contract!(
        !command_queue.is_null(),
        "queue can't be null",
        CL_INVALID_COMMAND_QUEUE
    );

    lcl_contract!(
        work_dim > 0 && work_dim <= 4,
        "invalid work_dim",
        CL_INVALID_VALUE
    );

    let queue = unsafe { command_queue.as_ref() }.unwrap();

    let offset = [0u32, 0, 0];

    let global_size_slice =
        unsafe { std::slice::from_raw_parts(global_work_size, work_dim as usize) };

    let global_size = match work_dim {
        1 => [global_size_slice[0] as u32, 1, 1],
        2 => [global_size_slice[0] as u32, global_size_slice[1] as u32, 1],
        3 => [
            global_size_slice[0] as u32,
            global_size_slice[1] as u32,
            global_size_slice[2] as u32,
        ],
        _ => panic!(),
    };

    // TODO fix local size
    let local_size = [1u32, 1, 1];

    queue.submit(kernel, offset, global_size, local_size);

    return CL_SUCCESS;
}

#[no_mangle]
pub extern "C" fn clFinish(command_queue: cl_command_queue) -> cl_int {
    lcl_contract!(
        !command_queue.is_null(),
        "queue can't be null",
        CL_INVALID_COMMAND_QUEUE
    );
    let queue = unsafe { command_queue.as_ref() }.unwrap();
    queue.finish();

    return CL_SUCCESS;
}
