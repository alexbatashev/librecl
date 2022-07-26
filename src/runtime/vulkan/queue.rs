use crate::common::cl_types::*;
use crate::common::device::ClDevice;
use crate::common::kernel::ClKernel;
use crate::common::memory::ClMem;
use crate::common::queue::Queue as CommonQueue;
use std::sync::Arc;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::device::Queue as VkQueue;
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::sync::{self, GpuFuture};

pub struct InOrderQueue {
    context: cl_context,
    device: cl_device_id,
    queue: Arc<VkQueue>,
}

impl InOrderQueue {
    pub fn new(context: cl_context, device: cl_device_id) -> cl_command_queue {
        let queue = match unsafe { device.as_ref() }.unwrap() {
            ClDevice::Vulkan(device) => device.get_queue(),
            _ => panic!(),
        };
        return Box::leak(Box::new(
            InOrderQueue {
                context,
                device,
                queue,
            }
            .into(),
        ));
    }
}

impl CommonQueue for InOrderQueue {
    fn enqueue_buffer_write(&self, src: *const libc::c_void, dst: cl_mem) {
        let transfer_fn = match unsafe { dst.as_mut() }.unwrap() {
            ClMem::VulkanSDBuffer(ref buffer) => || {
                buffer.write(src);
            },
            _ => panic!("Unexpected"),
        };

        // TODO events and concurrency
        transfer_fn();
    }
    fn enqueue_buffer_read(&self, src: cl_mem, dst: *mut libc::c_void) {
        let transfer_fn = match unsafe { src.as_ref() }.unwrap() {
            ClMem::VulkanSDBuffer(ref buffer) => || {
                buffer.read(dst);
            },
            _ => panic!("Unexpected"),
        };

        // TODO events and concurrency
        transfer_fn();
    }

    fn submit(
        &self,
        kernel: cl_kernel,
        offset: [u32; 3],
        global_size: [u32; 3],
        local_size: [u32; 3],
    ) {
        let kernel_safe = match unsafe { kernel.as_ref() }.unwrap() {
            ClKernel::Vulkan(kernel) => kernel,
            _ => panic!(),
        };

        let device = match unsafe { self.device.as_ref() }.unwrap() {
            ClDevice::Vulkan(device) => device,
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

        sync::now(device.get_logical_device().clone())
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
