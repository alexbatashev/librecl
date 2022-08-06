use std::ops::Deref;

use super::cl_types::*;
use crate::{
    format_error,
    interface::{
        ContextImpl, ContextKind, DeviceImpl, DeviceKind, KernelKind, MemKind, QueueImpl, QueueKind,
    },
    lcl_contract,
    sync::SharedPtr,
};

#[no_mangle]
pub unsafe extern "C" fn clCreateCommandQueueWithProperties(
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

    let context_safe = ContextKind::try_from_cl(context).unwrap();

    lcl_contract!(
        context_safe,
        !device.is_null(),
        "device can't be NULL",
        CL_INVALID_DEVICE,
        errcode_ret
    );

    let device_safe = DeviceKind::try_from_cl(device).unwrap();

    lcl_contract!(
        context_safe,
        context_safe
            .deref()
            .has_device(SharedPtr::downgrade(&device_safe)),
        "device must belong to the provided context",
        CL_INVALID_DEVICE,
        errcode_ret
    );

    let queue = device_safe.create_queue(context_safe, device_safe.clone());

    *errcode_ret = CL_SUCCESS;

    return _cl_command_queue::wrap(queue);
}

#[no_mangle]
pub unsafe extern "C" fn clEnqueueWriteBuffer(
    command_queue: cl_command_queue,
    buffer: cl_mem,
    _blocking_write: cl_bool,
    _offset: cl_size_t,
    _cb: cl_size_t,
    ptr: *const libc::c_void,
    _num_events_in_wait_list: cl_uint,
    // TODO support events
    _event_wait_list: *const cl_event,
    _event: *mut cl_event,
) -> cl_int {
    lcl_contract!(
        !command_queue.is_null(),
        "queue can't be null",
        CL_INVALID_COMMAND_QUEUE
    );

    let queue = QueueKind::try_from_cl(command_queue).unwrap();

    lcl_contract!(
        !buffer.is_null(),
        "buffer can't be NULL",
        CL_INVALID_MEM_OBJECT
    );

    let buffer_safe = MemKind::try_from_cl(buffer).unwrap();

    // TODO proper error handling

    // TODO blocking - non-blocking
    queue.enqueue_buffer_write(ptr, SharedPtr::downgrade(&buffer_safe));

    return CL_SUCCESS;
}

#[no_mangle]
pub unsafe extern "C" fn clEnqueueReadBuffer(
    command_queue: cl_command_queue,
    buffer: cl_mem,
    _blocking_read: cl_bool,
    _offset: cl_size_t,
    _cb: cl_size_t,
    ptr: *mut libc::c_void,
    _num_events_in_wait_list: cl_uint,
    // TODO support events
    _event_wait_list: *const cl_event,
    _event: *mut cl_event,
) -> cl_int {
    lcl_contract!(
        !command_queue.is_null(),
        "queue can't be null",
        CL_INVALID_COMMAND_QUEUE
    );

    let queue = QueueKind::try_from_cl(command_queue).unwrap();

    lcl_contract!(
        !buffer.is_null(),
        "buffer can't be NULL",
        CL_INVALID_MEM_OBJECT
    );

    // TODO proper error handling

    // TODO blocking - non-blocking
    let buffer_safe = MemKind::try_from_cl(buffer).unwrap();
    queue.enqueue_buffer_read(SharedPtr::downgrade(&buffer_safe), ptr);

    return CL_SUCCESS;
}

#[no_mangle]
pub unsafe extern "C" fn clEnqueueNDRangeKernel(
    command_queue: cl_command_queue,
    kernel: cl_kernel,
    work_dim: cl_uint,
    _global_work_offset: *const cl_size_t,
    global_work_size: *const cl_size_t,
    _local_work_size: *const cl_size_t,
    _num_events_in_wait_list: cl_uint,
    _event_wait_list: *const cl_event,
    _event: *mut cl_event,
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

    let queue = QueueKind::try_from_cl(command_queue).unwrap();
    let kernel_safe = KernelKind::try_from_cl(kernel).unwrap();

    let offset = [0u32, 0, 0];

    let global_size_slice = std::slice::from_raw_parts(global_work_size, work_dim as usize);

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

    queue.submit(
        SharedPtr::downgrade(&kernel_safe),
        offset,
        global_size,
        local_size,
    );

    return CL_SUCCESS;
}

#[no_mangle]
pub unsafe extern "C" fn clFinish(command_queue: cl_command_queue) -> cl_int {
    lcl_contract!(
        !command_queue.is_null(),
        "queue can't be null",
        CL_INVALID_COMMAND_QUEUE
    );
    let queue = QueueKind::try_from_cl(command_queue).unwrap();
    queue.finish();

    return CL_SUCCESS;
}

#[no_mangle]
pub unsafe extern "C" fn clRetainCommandQueue(queue: cl_command_queue) -> cl_int {
    lcl_contract!(
        !queue.is_null(),
        "queue can't be NULL",
        CL_INVALID_COMMAND_QUEUE
    );

    let queue_ref = &mut *queue;

    queue_ref.retain();

    return CL_SUCCESS;
}

#[no_mangle]
pub unsafe extern "C" fn clReleaseCommandQueue(queue: cl_command_queue) -> cl_int {
    lcl_contract!(
        !queue.is_null(),
        "queue can't be NULL",
        CL_INVALID_COMMAND_QUEUE
    );

    let queue_ref = &mut *queue;

    if queue_ref.release() == 1 {
        // Intentionally ignore value to destroy pointer and its content
        Box::from_raw(queue);
    }

    return CL_SUCCESS;
}
