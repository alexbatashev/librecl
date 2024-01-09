use super::{Event, HostToGPUEvent};
use crate::api::cl_types::*;
use crate::api::error_handling::ClError;
use crate::interface::{
    ContextImpl, ContextKind, DeviceKind, EventKind, KernelKind, MemKind, QueueImpl, QueueKind,
};
use crate::sync::{self, SharedPtr, UnsafeHandle, WeakPtr};
use ocl_type_wrapper::ClObjImpl;
use std::ops::Deref;
use std::sync::Arc;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::device::Queue as VkQueue;
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::sync::{self as sync_vk, GpuFuture};

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
            #[allow(unreachable_patterns)]
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
    fn enqueue_buffer_write(
        &self,
        src: *const libc::c_void,
        dst: WeakPtr<MemKind>,
    ) -> Result<EventKind, ClError> {
        let owned_buffer = dst.upgrade().ok_or(ClError::InvalidMemObject(
            "failed to acquire owning reference for buffer".into(),
        ))?;
        let unsafe_src = UnsafeHandle::new(src);
        let transfer_fn =
            |owned_buffer: sync::SharedPtr<MemKind>,
             unsafe_src: UnsafeHandle<*const libc::c_void>| {
                match owned_buffer.deref() {
                    MemKind::VulkanSDBuffer(ref buffer) => buffer.write(*unsafe_src.value()),
                    #[allow(unreachable_patterns)]
                    _ => panic!("Unexpected"),
                }
            };

        let context = self.context.upgrade().ok_or(ClError::InvalidCommandQueue(
            "failed to acquire owning reference to a context associated with the queue".into(),
        ))?;

        let _guard = context.get_threading_runtime().enter();

        // TODO events and concurrency
        let join_handle = tokio::spawn(async move {
            transfer_fn(owned_buffer, unsafe_src);
        });

        Ok(HostToGPUEvent::new(self.context.clone(), join_handle).into())
    }

    fn enqueue_buffer_read(
        &self,
        src: WeakPtr<MemKind>,
        dst: *mut libc::c_void,
    ) -> Result<EventKind, ClError> {
        let owned_buffer = src.upgrade().ok_or(ClError::InvalidMemObject(
            "failed to acquire owning reference for buffer".into(),
        ))?;
        let unsafe_dst = UnsafeHandle::new(dst);
        let transfer_fn =
            |owned_buffer: sync::SharedPtr<MemKind>,
             mut unsafe_dst: UnsafeHandle<*mut libc::c_void>| {
                match owned_buffer.deref() {
                    MemKind::VulkanSDBuffer(ref buffer) => buffer.read(*unsafe_dst.value_mut()),
                    #[allow(unreachable_patterns)]
                    _ => panic!("Unexpected"),
                }
            };

        let context = self.context.upgrade().ok_or(ClError::InvalidCommandQueue(
            "failed to acquire owning reference to a context associated with the queue".into(),
        ))?;

        let _guard = context.get_threading_runtime().enter();

        // TODO events and concurrency
        let join_handle = tokio::spawn(async move {
            transfer_fn(owned_buffer, unsafe_dst);
        });

        Ok(HostToGPUEvent::new(self.context.clone(), join_handle).into())
    }

    fn submit(
        &self,
        kernel: WeakPtr<KernelKind>,
        _offset: [u32; 3],
        global_size: [u32; 3],
        local_size: [u32; 3],
    ) -> Result<EventKind, ClError> {
        let owned_kernel = kernel.upgrade().unwrap();
        let kernel_safe = match owned_kernel.deref() {
            KernelKind::Vulkan(kernel) => kernel,
            #[allow(unreachable_patterns)]
            _ => panic!(),
        };

        let owned_device = self.device.upgrade().unwrap();
        let device = match owned_device.deref() {
            DeviceKind::Vulkan(device) => device,
            #[allow(unreachable_patterns)]
            _ => panic!(),
        };

        let (pipeline, sets) = kernel_safe.build_pipeline(device.get_logical_device().clone());

        let mut builder = AutoCommandBufferBuilder::primary(
            device.get_logical_device().clone(),
            self.queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|_| ClError::OutOfResources("failed to create command buffer".into()))?;

        builder
            .bind_pipeline_compute(pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                sets,
            );

        // TODO process errors correctly
        builder
            .dispatch([
                global_size[0] / local_size[0],
                global_size[1] / local_size[1],
                global_size[2] / local_size[2],
            ])
            .map_err(|_| ClError::OutOfResources("failed to dispatch command buffer".into()))?;

        let command_buffer = builder.build().unwrap();

        // TODO add more details about queue and kernel
        let future = sync_vk::now(device.get_logical_device().clone())
            .then_execute(self.queue.clone(), command_buffer)
            .map_err(|_| ClError::OutOfResources("failed to submit Vulkan command".into()))?;
        // TODO figure out semaphores
        // let semaphore = SharedPtr::new(future.then_signal_semaphore());
        let fence = future
            .then_signal_fence_and_flush()
            .map_err(|_| ClError::OutOfResources("failed to submit Vulkan command".into()))?;

        Ok(Event::new(self.context.clone(), SharedPtr::new(fence)))
    }

    fn finish(&self) {
        // TODO do we have better ideas?
        self.queue.wait().unwrap();
    }
}
