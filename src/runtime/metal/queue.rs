use crate::api::cl_types::*;
use crate::interface::{ContextKind, DeviceKind, KernelKind, MemKind, QueueImpl, QueueKind};
use crate::sync::{self, *};
use metal_api::CommandQueue;
use metal_api::MTLSize;
use ocl_type_wrapper::ClObjImpl;
use std::ops::Deref;
use std::sync::{Arc, Mutex};

#[derive(ClObjImpl)]
pub struct InOrderQueue {
    context: WeakPtr<ContextKind>,
    device: WeakPtr<DeviceKind>,
    queue: Arc<Mutex<UnsafeHandle<CommandQueue>>>,
    handle: UnsafeHandle<cl_command_queue>,
}

impl InOrderQueue {
    pub fn new(context: WeakPtr<ContextKind>, device: WeakPtr<DeviceKind>) -> QueueKind {
        let owned_device = device.upgrade().unwrap();
        let device_safe = match owned_device.deref() {
            DeviceKind::Metal(device) => device,
            _ => panic!(),
        };

        let device_lock = device_safe.get_native_device().lock().unwrap();

        let queue = Arc::new(Mutex::new(UnsafeHandle::new(
            device_lock.value().new_command_queue(),
        )));

        InOrderQueue {
            context,
            device,
            queue,
            handle: UnsafeHandle::null(),
        }
        .into()
    }
}

impl QueueImpl for InOrderQueue {
    fn enqueue_buffer_write(&self, src: *const libc::c_void, dst: WeakPtr<MemKind>) {
        let owned_buffer = dst.upgrade().unwrap();
        let transfer_fn = match owned_buffer.deref() {
            MemKind::MetalSDBuffer(ref buffer) => || {
                buffer.write(src);
            },
            _ => panic!("Unexpected"),
        };

        // TODO events and concurrency
        transfer_fn();
    }
    fn enqueue_buffer_read(&self, src: WeakPtr<MemKind>, dst: *mut libc::c_void) {
        let owned_buffer = src.upgrade().unwrap();
        let transfer_fn = match owned_buffer.deref() {
            MemKind::MetalSDBuffer(ref buffer) => || {
                buffer.read(dst);
            },
            _ => panic!("Unexpected"),
        };

        // TODO events and concurrency
        transfer_fn();
    }
    fn submit(
        &self,
        kernel: WeakPtr<KernelKind>,
        offset: [u32; 3],
        global_size: [u32; 3],
        local_size: [u32; 3],
    ) {
        let owned_kernel = kernel.upgrade().unwrap();
        let kernel_safe = match owned_kernel.deref() {
            KernelKind::Metal(kernel) => kernel,
            _ => panic!(),
        };

        let locked_queue = self.queue.lock().unwrap();

        let command_buffer = locked_queue.value().new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        let pso = kernel_safe.prepare_pso(self.device.clone());

        compute_encoder.set_compute_pipeline_state(&pso);

        kernel_safe.encode_arguments(compute_encoder);

        let grid_size = MTLSize::new(
            (global_size[0] / local_size[0]) as u64,
            (global_size[1] / local_size[1]) as u64,
            (global_size[2] / local_size[2]) as u64,
        );

        let threadgroup_size = MTLSize::new(
            local_size[0] as u64,
            local_size[1] as u64,
            local_size[2] as u64,
        );

        compute_encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        compute_encoder.end_encoding();

        command_buffer.commit();

        // TODO super sketchy
        command_buffer.wait_until_completed();
    }

    fn finish(&self) {
        // TODO nop for now
    }
}
