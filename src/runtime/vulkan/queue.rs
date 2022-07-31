use crate::sync::{self, UnsafeHandle, WeakPtr};
use ocl_type_wrapper::ClObjImpl;
use std::ops::Deref;
use std::sync::Arc;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::device::Queue as VkQueue;
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::sync::{self as sync_vk, GpuFuture};

use crate::api::cl_types::*;
use crate::interface::{ContextKind, DeviceKind, KernelKind, MemKind, QueueImpl, QueueKind};

#[derive(ClObjImpl)]
pub struct InOrderQueue {
    context: WeakPtr<ContextKind>,
    device: WeakPtr<DeviceKind>,
    queue: Arc<VkQueue>,
    #[cl_handle]
    handle: UnsafeHandle<cl_command_queue>,
}

impl InOrderQueue {
    pub fn new(context: WeakPtr<ContextKind>, device: WeakPtr<DeviceKind>) -> QueueKind {
        let owned_device = device.upgrade().unwrap();
        let queue = match owned_device.deref() {
            DeviceKind::Vulkan(device) => device.get_queue(),
            _ => panic!(),
        };
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
            MemKind::VulkanSDBuffer(ref buffer) => || {
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
            MemKind::VulkanSDBuffer(ref buffer) => || {
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
            KernelKind::Vulkan(kernel) => kernel,
            _ => panic!(),
        };

        let owned_device = self.device.upgrade().unwrap();
        let device = match owned_device.deref() {
            DeviceKind::Vulkan(device) => device,
            _ => panic!(),
        };

        let (pipeline, sets) = kernel_safe.build_pipeline(device.get_logical_device().clone());

        let mut builder = AutoCommandBufferBuilder::primary(
            device.get_logical_device().clone(),
            self.queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .bind_pipeline_compute(pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                sets,
            );

        builder.dispatch([
            global_size[0] / local_size[0],
            global_size[1] / local_size[1],
            global_size[2] / local_size[2],
        ]);

        let command_buffer = builder.build().unwrap();

        sync_vk::now(device.get_logical_device().clone())
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        // TODO return events.
    }

    fn finish(&self) {
        // TODO do we have better ideas?
        self.queue.wait().unwrap();
    }
}
