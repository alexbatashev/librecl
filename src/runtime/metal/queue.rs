use crate::common::cl_types::*;
use crate::common::device::ClDevice;
use crate::common::kernel::ClKernel;
use crate::common::memory::ClMem;
use crate::common::queue::Queue as CommonQueue;
use metal_api::CommandQueue;
use metal_api::MTLSize;
use std::sync::Arc;

pub struct InOrderQueue {
    context: cl_context,
    device: cl_device_id,
    queue: Arc<CommandQueue>,
}

impl InOrderQueue {
    pub fn new(context: cl_context, device: cl_device_id) -> cl_command_queue {
        let device_safe = match unsafe { device.as_ref() }.unwrap() {
            ClDevice::Metal(device) => device,
            _ => panic!(),
        };
        let queue = Arc::new(device_safe.get_native_device().new_command_queue());

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
        let transfer_fn = match unsafe { dst.as_ref() }.unwrap() {
            ClMem::MetalSDBuffer(ref buffer) => || {
                buffer.write(src);
            },
            _ => panic!("Unexpected"),
        };

        // TODO events and concurrency
        transfer_fn();
    }
    fn enqueue_buffer_read(&self, src: cl_mem, dst: *mut libc::c_void) {
        let transfer_fn = match unsafe { src.as_ref() }.unwrap() {
            ClMem::MetalSDBuffer(ref buffer) => || {
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
            ClKernel::Metal(kernel) => kernel,
            _ => panic!(),
        };

        let command_buffer = self.queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        let pso = kernel_safe.prepare_pso(self.device);

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
