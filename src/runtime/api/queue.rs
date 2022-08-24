use super::cl_types::*;
use super::error_handling::{
    map_invalid_context, map_invalid_device, map_invalid_kernel, map_invalid_mem,
    map_invalid_queue, ClError,
};
use crate::{
    interface::{
        ContextImpl, ContextKind, DeviceImpl, DeviceKind, KernelKind, MemKind, QueueImpl, QueueKind,
    },
    lcl_contract, success,
    sync::SharedPtr,
};
use lcl_derive::cl_api;
use std::ops::Deref;

#[cl_api]
fn clCreateCommandQueueWithProperties(
    context: cl_context,
    device: cl_device_id,
    _properties: *const cl_queue_properties,
) -> Result<cl_command_queue, ClError> {
    // TODO do not ignore properties

    let context_safe = ContextKind::try_from_cl(context).map_err(map_invalid_context)?;

    let device_safe = DeviceKind::try_from_cl(device).map_err(map_invalid_device)?;

    lcl_contract!(
        context_safe,
        context_safe
            .deref()
            .has_device(SharedPtr::downgrade(&device_safe)),
        ClError::InvalidDevice,
        "device must belong to the provided context"
    );

    let queue = device_safe.create_queue(context_safe, device_safe.clone());

    Ok(_cl_command_queue::wrap(queue))
}

#[cl_api]
fn clEnqueueWriteBuffer(
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
) -> Result<(), ClError> {
    let queue = QueueKind::try_from_cl(command_queue).map_err(map_invalid_queue)?;

    let buffer_safe = MemKind::try_from_cl(buffer).map_err(map_invalid_mem)?;

    // TODO proper error handling

    // TODO blocking - non-blocking
    queue.enqueue_buffer_write(ptr, SharedPtr::downgrade(&buffer_safe));

    return success!();
}

#[cl_api]
fn clEnqueueReadBuffer(
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
) -> Result<(), ClError> {
    let queue = QueueKind::try_from_cl(command_queue).map_err(map_invalid_queue)?;

    // TODO proper error handling

    // TODO blocking - non-blocking
    let buffer_safe = MemKind::try_from_cl(buffer).map_err(map_invalid_mem)?;
    queue.enqueue_buffer_read(SharedPtr::downgrade(&buffer_safe), ptr);

    return success!();
}

#[cl_api]
fn clEnqueueNDRangeKernel(
    command_queue: cl_command_queue,
    kernel: cl_kernel,
    work_dim: cl_uint,
    _global_work_offset: *const cl_size_t,
    global_work_size: *const cl_size_t,
    _local_work_size: *const cl_size_t,
    _num_events_in_wait_list: cl_uint,
    _event_wait_list: *const cl_event,
    _event: *mut cl_event,
) -> Result<(), ClError> {
    lcl_contract!(
        work_dim > 0 && work_dim <= 3,
        ClError::InvalidValue,
        "invalid work_dim"
    );

    let queue = QueueKind::try_from_cl(command_queue).map_err(map_invalid_queue)?;
    let kernel_safe = KernelKind::try_from_cl(kernel).map_err(map_invalid_kernel)?;

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

    queue.submit(
        SharedPtr::downgrade(&kernel_safe),
        offset,
        global_size,
        local_size,
    );

    return success!();
}

#[cl_api]
fn clFinish(command_queue: cl_command_queue) -> Result<(), ClError> {
    let queue = QueueKind::try_from_cl(command_queue).map_err(map_invalid_queue)?;
    queue.finish();

    return success!();
}

#[cl_api]
fn clRetainCommandQueue(queue: cl_command_queue) -> Result<(), ClError> {
    lcl_contract!(
        !queue.is_null(),
        ClError::InvalidCommandQueue,
        "queue can't be NULL"
    );

    let queue_ref = unsafe { &mut *queue };

    queue_ref.retain();

    return success!();
}

#[cl_api]
fn clReleaseCommandQueue(queue: cl_command_queue) -> Result<(), ClError> {
    lcl_contract!(
        !queue.is_null(),
        ClError::InvalidCommandQueue,
        "queue can't be NULL"
    );

    let queue_ref = unsafe { &mut *queue };

    if queue_ref.release() == 1 {
        // Intentionally ignore value to destroy pointer and its content
        unsafe { Box::from_raw(queue) };
    }

    return success!();
}
